# Launching vLLM as a server and talking to its OpenAI-compatible endpoint.
#
# Standard library only (urllib + threads): the evaluation host has no
# internet, extra wheels would be a risk.

import http.client
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .console import log

PROGRESS_INTERVAL_SECONDS = 15
PROGRESS_BAR_WIDTH = 30


class VllmServer:
    def __init__(self, model_path, port=8000, served_name="m",
                 max_model_len=131072, gpu_memory_utilization=0.90,
                 log_path=None, extra_arguments=None):
        self.model_path = model_path
        self.port = port
        self.served_name = served_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.log_path = log_path
        self.extra_arguments = list(extra_arguments) if extra_arguments else []
        self.process = None

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.port

    def _command(self):
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_path),
            "--served-model-name", self.served_name,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--language-model-only",
            "--enable-prefix-caching",
            *self.extra_arguments,
        ]

    def start(self):
        cmd = self._command()
        log("старт vLLM: " + " ".join(cmd[3:]))
        if self.log_path:
            sink = open(self.log_path, "w", encoding="utf-8")
            self.process = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT)
        else:
            self.process = subprocess.Popen(cmd)

    def wait_until_ready(self, deadline):
        health = "%s/models" % self.base_url
        begin = time.time()
        last = 0.0
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                log("vLLM завершился с кодом %d" % self.process.returncode)
                self._dump_tail()
                return False
            try:
                with urllib.request.urlopen(health, timeout=5) as resp:
                    if resp.status == 200:
                        log("vLLM готов за %.0f с" % (time.time() - begin))
                        return True
            except (OSError, http.client.HTTPException):
                pass
            now = time.time()
            if now - last > 30:
                log("жду vLLM… %.0f с" % (now - begin))
                last = now
            time.sleep(2)
        log("vLLM не поднялся до дедлайна")
        self._dump_tail()
        return False

    def _dump_tail(self, lines=60):
        if not self.log_path:
            return
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, encoding="utf-8", errors="replace") as fh:
            content = fh.read().splitlines()
        print("\n".join(content[-lines:]), flush=True)

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()


class CompletionClient:
    def __init__(self, base_url, model="m", timeout=900.0, retries=3):
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.failed_requests = 0
        self.hard_deadline = None
        self._lock = threading.Lock()

    def _remaining(self):
        if self.hard_deadline is None:
            return self.timeout
        return min(self.timeout, self.hard_deadline - time.time())

    def _count_failure(self):
        with self._lock:
            self.failed_requests += 1

    def complete(self, prompt, **sampling):
        """Server response; ``{"error": ...}`` on HTTP 400 (e.g. prompt too
        long); ``None`` on a transport failure."""
        body = {"model": self.model, "prompt": prompt}
        body.update(sampling)
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for attempt in range(self.retries):
            budget = self._remaining()
            if budget < 2:
                return None
            request = urllib.request.Request(self.url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=budget) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                blob = error.read().decode("utf-8", errors="replace")[:300]
                self._count_failure()
                if error.code == 400:
                    log("HTTP 400: %s" % blob)
                    return {"error": blob}
            except (OSError, ValueError, http.client.HTTPException) as error:
                self._count_failure()
                if attempt == self.retries - 1:
                    log("запрос не удался: %r" % error)
            time.sleep(2 ** attempt)
        return None


def first_token_alternatives(response):
    """top-logprobs of the first generated token: ``{token: logprob}``."""
    if not response or "error" in response:
        return None
    lp = response["choices"][0].get("logprobs") or {}
    alt = lp.get("top_logprobs") or []
    if not alt:
        return None
    return alt[0]


def _best_logprob(alternatives, needle):
    best = None
    for token, lp in alternatives.items():
        if (token or "").strip() != needle:
            continue
        if best is None or lp > best:
            best = lp
    return best


def yes_probability(alternatives, yes="1", no="0"):
    """P(yes) as a softmax between the best "yes" and best "no" tokens."""
    if not alternatives:
        return None
    y = _best_logprob(alternatives, yes)
    n = _best_logprob(alternatives, no)
    if y is None and n is None:
        return None
    if n is None:
        return 1.0 - 1e-6
    if y is None:
        return 1e-6
    peak = max(y, n)
    yw = math.exp(y - peak)
    nw = math.exp(n - peak)
    return yw / (yw + nw)


def digit_probability(alternatives):
    """P('1') via top-logprobs, but never a hard 0/1 verdict: when only one
    of the digits is in top-k, we use its own weight so confident calls stay
    separable at ranking time."""
    if not alternatives:
        return None
    y = _best_logprob(alternatives, "1")
    n = _best_logprob(alternatives, "0")
    if y is None and n is None:
        return None
    if n is None:
        return min(1.0 - 1e-6, max(0.501, math.exp(y)))
    if y is None:
        return max(1e-6, min(0.499, 1.0 - math.exp(n)))
    peak = max(y, n)
    yw = math.exp(y - peak)
    nw = math.exp(n - peak)
    return yw / (yw + nw)


def map_concurrently(function, items, workers, deadline, title):
    """Run ``function`` over ``items`` with a thread pool.

    Returns ``{position: value}`` for successful entries. After the deadline
    is reached no new items are started."""
    results = {}
    if not items:
        return results

    started_at = time.time()
    completed = 0
    last_report = 0.0

    def _run(pos, item):
        if time.time() > deadline:
            return pos, None
        try:
            return pos, function(item)
        except Exception as error:  # noqa: BLE001
            log("%s: ошибка на элементе %d: %r" % (title, pos, error))
            return pos, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for pos, item in enumerate(items):
            futures.append(pool.submit(_run, pos, item))
        for fut in as_completed(futures):
            pos, value = fut.result()
            if value is not None:
                results[pos] = value
            completed += 1
            now = time.time()
            if now - last_report > PROGRESS_INTERVAL_SECONDS or completed == len(items):
                elapsed = now - started_at
                remaining = elapsed / completed * (len(items) - completed)
                bar_fill = int(PROGRESS_BAR_WIDTH * completed / len(items))
                bar = "█" * bar_fill
                log("%s |%s| %d/%d за %.0f с, осталось ≈%.0f с" % (
                    title, bar.ljust(PROGRESS_BAR_WIDTH),
                    completed, len(items), elapsed, remaining,
                ))
                last_report = now
    return results
