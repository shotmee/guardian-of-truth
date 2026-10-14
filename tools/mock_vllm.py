"""Фейковый OpenAI-совместимый vLLM для локальной проверки пайплайна без GPU.
Запуск: python tools/mock_vllm.py --port 8011 [--delay 0.05] [--think-delay 0.3]"""
import argparse
import hashlib
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8011)
ap.add_argument("--delay", type=float, default=0.02)
ap.add_argument("--think-delay", type=float, default=0.2)
ap.add_argument("--truncate-rate", type=float, default=0.1)
A = ap.parse_args()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "guardian", "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["prompt"]
        h = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        rnd = random.Random(h + int(req.get("seed") or 0))
        p1 = rnd.random()
        top = {"1": float(-0.01 - (1 - p1) * 3), "0": float(-0.01 - p1 * 3)}
        thinking = prompt.endswith("<think>\n")
        if not thinking:
            time.sleep(A.delay)
            tok = "1" if p1 > 0.5 else "0"
            ch = {"index": 0, "text": tok, "finish_reason": "length",
                  "logprobs": {"tokens": [tok], "token_logprobs": [top[tok]], "top_logprobs": [top], "text_offset": [0]}}
            usage = 1
        else:
            time.sleep(A.think_delay)
            if rnd.random() < A.truncate_rate:
                toks = ["Анализ", " вызовов", " продолжается"] * 5
                fin = "length"
            else:
                toks = ["Анализ", " вызовов", ".", "</think>", "\n\n", "1" if p1 > 0.5 else "0"]
                fin = "stop"
            tops = [{t: -0.1} for t in toks]
            if fin == "stop":
                tops[-1] = top
            ch = {"index": 0, "text": "".join(toks), "finish_reason": fin,
                  "logprobs": {"tokens": toks, "token_logprobs": [-0.1] * len(toks), "top_logprobs": tops,
                               "text_offset": list(range(len(toks)))}}
            usage = len(toks)
        self._send({"id": "cmpl-mock", "object": "text_completion", "created": 0, "model": "guardian",
                    "choices": [ch], "usage": {"prompt_tokens": 10, "completion_tokens": usage, "total_tokens": 10 + usage}})


print(f"mock vLLM on :{A.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
