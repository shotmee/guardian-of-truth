"""Фейковый vLLM (/v1/models, /v1/completions) для e2e-проверки конвейера на CPU.

Отвечает правдоподобно по форме: вызовы инструментов в формате Qwen (<tool_call><function=…>),
рассуждения, однотокенные ответы с top_logprobs. Детерминирован (сид из хэша промпта).
Запуск: python dev/mock_vllm.py --port 8011 [--delay 0.01]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"requests": 0, "max_prompt": 0}


def rng_for(prompt: str, salt: str = "") -> random.Random:
    return random.Random(int(hashlib.md5((prompt[-4000:] + salt).encode()).hexdigest()[:8], 16))


TEXT_ONLY = False


def fake_call(rng: random.Random, prompt: str) -> str:
    if TEXT_ONLY:
        return "Пожалуйста, подтвердите детали."
    tools = re.findall(r'"name": "([A-Za-z_][\w]*)", "description"', prompt[:60000])
    if not tools or rng.random() < 0.25:
        return "Пожалуйста, подтвердите детали."
    name = rng.choice(tools)
    # иногда повторяем последний вызов из истории, чтобы проверить совпадения
    hist = re.findall(r"<function=([^>]+)>\n(.*?)</function>", prompt[-20000:], re.S)
    if hist and rng.random() < 0.5:
        name, body = hist[-1]
        return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"
    return f"<tool_call>\n<function={name}>\n<parameter=x>\n{rng.randint(1, 9)}\n</parameter>\n</function>\n</tool_call>"


class H(BaseHTTPRequestHandler):
    delay = 0.0

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send({"data": [{"id": "m"}]})
        self._send({"error": "nf"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        if self.path.endswith("/chat/completions"):          # chat API (ветка GLM в dev/score_v6.py): один токен
            text = json.dumps(body.get("messages"), ensure_ascii=False)
            STATS["requests"] += 1
            if len(text) > 600000:
                return self._send({"error": {"message": "maximum context length"}}, 400)
            p = rng_for(text).random()
            top = [{"token": "1", "logprob": -0.1 - 3 * p}, {"token": "0", "logprob": -0.1 - 3 * (1 - p)}]
            return self._send({"choices": [{"index": 0, "message": {"role": "assistant", "content": "1"},
                                            "logprobs": {"content": [{"token": "1", "logprob": -0.1,
                                                                      "top_logprobs": top}]}}]})
        prompt = body["prompt"]
        STATS["requests"] += 1
        STATS["max_prompt"] = max(STATS["max_prompt"], len(prompt))
        if len(prompt) > 600000:
            return self._send({"error": {"message": "This model's maximum context length is 131072"}}, 400)
        time.sleep(self.delay)
        k = int(body.get("n", 1))
        rng = rng_for(prompt)
        choices = []
        for j in range(k):
            r = rng_for(prompt, str(j))
            if body.get("max_tokens") == 1:
                p = r.random()
                top = {"1": -0.1 - 3 * p, "0": -0.1 - 3 * (1 - p), "\n": -6.0}
                choices.append({"index": j, "text": "1" if p < 0.5 else "0", "finish_reason": "length",
                                "logprobs": {"tokens": ["1"], "token_logprobs": [-0.1], "top_logprobs": [top]}})
                continue
            thinking = prompt.endswith("<think>\n")
            stop = body.get("stop") or []
            if "list the specific factual claims" in prompt:          # выписка фактов (veritas.claims)
                text = "\n".join(f"{i}. Факт номер {i} про сумму {r.randint(10, 999)}$" for i in range(1, r.randint(1, 5)))
                text = text or "NONE"
            elif thinking and "</think>" in stop:
                text = "Let me check the policy and the facts. " * r.randint(3, 30)
            elif thinking:
                text = "Reasoning about the next step. " * r.randint(2, 10) + "</think>\n\n" + fake_call(r, prompt)
            else:
                text = fake_call(r, prompt)
            choices.append({"index": j, "text": text, "finish_reason": "stop", "logprobs": None})
        self._send({"id": "x", "object": "text_completion", "choices": choices})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--text-only", action="store_true", help="никогда не выдавать вызовов (проверка защиты)")
    a = ap.parse_args()
    global TEXT_ONLY
    TEXT_ONLY = a.text_only
    H.delay = a.delay
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"mock vLLM on :{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
