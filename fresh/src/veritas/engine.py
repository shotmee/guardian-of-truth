"""vLLM: запуск OpenAI-совместимого сервера и параллельные запросы к /v1/completions.

Только стандартная библиотека (urllib + потоки): на платформе нет интернета, лишние зависимости — риск.
"""
from __future__ import annotations

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
from typing import Callable, Optional

from .console import log

PROGRESS_INTERVAL_SECONDS = 15
PROGRESS_BAR_WIDTH = 30


class VllmServer:
    def __init__(self, model_path: str, port: int = 8000, served_name: str = "m", max_model_len: int = 131072,
                 gpu_memory_utilization: float = 0.90, log_path: Optional[str] = None,
                 extra_arguments: Optional[list] = None):
        self.model_path = model_path
        self.port = port
        self.served_name = served_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.log_path = log_path
        self.extra_arguments = list(extra_arguments or [])
        self.process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", str(self.model_path), "--served-model-name", self.served_name,
                   "--port", str(self.port), "--max-model-len", str(self.max_model_len),
                   "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                   "--language-model-only", "--enable-prefix-caching", *self.extra_arguments]
        log("старт vLLM: " + " ".join(command[3:]))
        output = open(self.log_path, "w", encoding="utf-8") if self.log_path else None
        self.process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT if output else None)

    def wait_until_ready(self, deadline: float) -> bool:
        health_url = f"{self.base_url}/models"
        started = time.time()
        last_report = 0.0
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                log(f"vLLM завершился с кодом {self.process.returncode}")
                self._print_log_tail()
                return False
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        log(f"vLLM готов за {time.time() - started:.0f} с")
                        return True
            except (OSError, http.client.HTTPException):   # сервер ещё поднимается
                pass
            if time.time() - last_report > 30:
                log(f"жду vLLM… {time.time() - started:.0f} с")
                last_report = time.time()
            time.sleep(2)
        log("vLLM не поднялся до дедлайна")
        self._print_log_tail()
        return False

    def _print_log_tail(self, lines: int = 60) -> None:
        if not self.log_path or not os.path.exists(self.log_path):
            return
        with open(self.log_path, encoding="utf-8", errors="replace") as source:
            print("\n".join(source.read().splitlines()[-lines:]), flush=True)

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()


class CompletionClient:
    def __init__(self, base_url: str, model: str = "m", timeout: float = 900.0, retries: int = 3):
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.failed_requests = 0
        self.hard_deadline: Optional[float] = None   # после него запросы не отправляются и обрываются
        self._lock = threading.Lock()

    def _remaining_timeout(self) -> float:
        if self.hard_deadline is None:
            return self.timeout
        return min(self.timeout, self.hard_deadline - time.time())

    def _count_failure(self) -> None:
        with self._lock:
            self.failed_requests += 1

    def complete(self, prompt: str, **sampling) -> Optional[dict]:
        """Ответ сервера; {"error": ...} при HTTP 400 (например, слишком длинный промпт); None при сбое."""
        payload = json.dumps({"model": self.model, "prompt": prompt, **sampling}).encode("utf-8")
        for attempt in range(self.retries):
            timeout = self._remaining_timeout()
            if timeout < 2:
                return None
            request = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")[:300]
                self._count_failure()
                if error.code == 400:                 # повтор не поможет
                    log(f"HTTP 400: {details}")
                    return {"error": details}
            except (OSError, ValueError, http.client.HTTPException) as error:
                self._count_failure()
                if attempt == self.retries - 1:
                    log(f"запрос не удался: {error!r}")
            time.sleep(2 ** attempt)
        return None


def first_token_alternatives(response: Optional[dict]) -> Optional[dict]:
    """top-logprobs первого сгенерированного токена: {токен: logprob}."""
    if not response or "error" in response:
        return None
    logprobs = response["choices"][0].get("logprobs") or {}
    alternatives = logprobs.get("top_logprobs") or []
    return alternatives[0] if alternatives else None


def yes_probability(alternatives: Optional[dict], yes: str = "1", no: str = "0") -> Optional[float]:
    """P(yes) — softmax между лучшими вариантами токенов yes и no на одной позиции."""
    if not alternatives:
        return None
    yes_logprob = max((lp for token, lp in alternatives.items() if token.strip() == yes), default=None)
    no_logprob = max((lp for token, lp in alternatives.items() if token.strip() == no), default=None)
    if yes_logprob is None and no_logprob is None:
        return None
    if no_logprob is None:
        return 1.0 - 1e-6
    if yes_logprob is None:
        return 1e-6
    top = max(yes_logprob, no_logprob)
    yes_weight, no_weight = math.exp(yes_logprob - top), math.exp(no_logprob - top)
    return yes_weight / (yes_weight + no_weight)


def digit_probability(alternatives: Optional[dict]) -> Optional[float]:
    """P('1') по top-logprobs без жёстких 0/1: если в топе только одна цифра, оценка по ней самой
    (P = exp(lp1) или 1 − exp(lp0)), чтобы уверенные ответы оставались различимыми при ранжировании."""
    if not alternatives:
        return None
    yes_logprob = max((lp for token, lp in alternatives.items() if (token or "").strip() == "1"), default=None)
    no_logprob = max((lp for token, lp in alternatives.items() if (token or "").strip() == "0"), default=None)
    if yes_logprob is None and no_logprob is None:
        return None
    if no_logprob is None:
        return min(1.0 - 1e-6, max(0.501, math.exp(yes_logprob)))
    if yes_logprob is None:
        return max(1e-6, min(0.499, 1.0 - math.exp(no_logprob)))
    top = max(yes_logprob, no_logprob)
    yes_weight, no_weight = math.exp(yes_logprob - top), math.exp(no_logprob - top)
    return yes_weight / (yes_weight + no_weight)


def map_concurrently(function: Callable, items: list, workers: int, deadline: float, title: str) -> dict:
    """{позиция: function(item)} для успешных элементов; после дедлайна новые элементы не запускаются."""
    results: dict = {}
    if not items:
        return results
    started = time.time()
    finished = 0
    last_report = 0.0

    def run_one(position: int, item):
        if time.time() > deadline:
            return position, None
        try:
            return position, function(item)
        except Exception as error:  # noqa: BLE001 — сбой одного элемента не должен ронять весь этап
            log(f"{title}: ошибка на элементе {position}: {error!r}")
            return position, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, position, item) for position, item in enumerate(items)]
        for future in as_completed(futures):
            position, result = future.result()
            if result is not None:
                results[position] = result
            finished += 1
            now = time.time()
            if now - last_report > PROGRESS_INTERVAL_SECONDS or finished == len(items):
                elapsed = now - started
                remaining = elapsed / finished * (len(items) - finished)
                bar = "█" * int(PROGRESS_BAR_WIDTH * finished / len(items))
                log(f"{title} |{bar:<{PROGRESS_BAR_WIDTH}}| {finished}/{len(items)} за {elapsed:.0f} с, "
                    f"осталось ≈{remaining:.0f} с")
                last_report = now
    return results
