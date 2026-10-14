"""Сигналы v6 на размеченных ходах (Colab, один поднятый vLLM): base / meta / shots для Qwen или GLM.

  base  — промпт бейзлайна организаторов (как P_base в решении);
  meta  — он же + метаданные задачи τ² (veritas.meta_judge; ходы без найденной задачи пропускаются);
  shots — он же + примеры разметки организаторов из valid (prompts/organizer_examples.md).
Qwen: запрос собирается нашим шаблоном чата и уходит в /v1/completions — побайтно как в решении.
GLM: /v1/chat/completions (шаблон применяет сервер), рассуждение выключено, logprobs первого токена.
Каждый (набор, модель, сигнал) — свой jsonl-чекпоинт; повторный запуск досчитывает недостающее.
python dev/score_v6.py --input x.parquet --name tau2 --model qwen|glm --base-url URL --out DIR [--signals base,meta,shots]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from veritas import baseline, meta_judge  # noqa: E402
from veritas.analysis import analyze_examples  # noqa: E402
from veritas.chat import ChatTemplate, judge_prompt  # noqa: E402
from veritas.engine import CompletionClient, digit_probability, first_token_alternatives, map_concurrently  # noqa: E402
from veritas.task_meta import find_spec, render_spec  # noqa: E402
from veritas import policy_slice  # noqa: E402
from veritas.transcript import parse_agent_turn, parse_context  # noqa: E402

LIMIT = baseline.MAX_PROMPT_CHARS
EXAMPLES = ROOT / "src" / "veritas" / "prompts" / "organizer_examples.md"


class Checkpoint:
    def __init__(self, path: Path):
        self.path, self.values = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.values[record["id"]] = record["value"]

    def add(self, results: dict) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            for key, value in results.items():
                self.values[key] = value
                output.write(json.dumps({"id": key, "value": value}, ensure_ascii=False) + "\n")


ERRORS: list = []                      # первые сбои запросов — печатаются после каждого сигнала


def chat_top_logprobs(base_url: str, model: str, system: str, user: str, timeout: float = 600,
                      overrides: Optional[dict] = None) -> Optional[dict]:
    """top-logprobs первого токена ответа через chat API; {"error": …} при HTTP 400; None при сбое."""
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    for key, value in (overrides or {}).items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    payload = json.dumps(body).encode("utf-8")
    for attempt in range(3):
        request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = ((body["choices"][0].get("logprobs") or {}).get("content") or [])
            return {item["token"]: item["logprob"] for item in content[0]["top_logprobs"]} if content else {}
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", "replace")[:300]
            if len(ERRORS) < 5:
                ERRORS.append(f"HTTP {error.code}: {details}")
            if error.code == 400:
                return {"error": details}
        except Exception as error:  # noqa: BLE001 — сеть/таймаут/разбор ответа: повтор
            if len(ERRORS) < 5:
                ERRORS.append(repr(error)[:300])
        time.sleep(2 ** attempt)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", choices=["qwen", "glm"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served", default="m")
    parser.add_argument("--out", required=True)
    parser.add_argument("--signals", default="base,meta,shots")
    parser.add_argument("--ids", default=None)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--api", choices=["completions", "chat"], default=None,
                        help="по умолчанию: qwen — completions (нашим шаблоном), glm — chat")
    parser.add_argument("--template", default=None, help="шаблон чата для completions (по умолчанию — Qwen из assets)")
    parser.add_argument("--chat-params", default=None, help="JSON: поля запроса chat API поверх стандартных (null — убрать)")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).reset_index(drop=True)
    if args.ids:
        wanted = set(json.loads(Path(args.ids).read_text(encoding="utf-8")))
        frame = frame[frame.id.isin(wanted)].reset_index(drop=True)
    analysis = analyze_examples(frame)
    kinds = ["text" if i not in analysis.call_turns else ("hard" if analysis.call_turns[i].violations else "call")
             for i in range(len(frame))]
    specs = [find_spec(example_id, prompt) for example_id, prompt in zip(frame.id, frame.prompt)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = args.api or ("completions" if args.model == "qwen" else "chat")
    template = ChatTemplate(Path(args.template) if args.template else ROOT / "src" / "veritas" / "assets" / "chat_template.jinja")
    overrides = json.loads(args.chat_params) if args.chat_params else None
    client = CompletionClient(args.base_url, model=args.served)
    systems = {"base": (baseline.PROMPTS_DIR / "baseline.md").read_text(encoding="utf-8").strip(),
               "meta": meta_judge.system_prompt()}
    systems["shots"] = systems["base"] + "\n\n" + EXAMPLES.read_text(encoding="utf-8")
    systems["slice"], systems["window"] = policy_slice.SLICE_SYSTEM, policy_slice.WINDOW_SYSTEM
    systems["focus"] = systems["base"] + policy_slice.FOCUS_NOTE
    fragments: dict = {}

    def fragment(i: int) -> tuple:
        """(фрагмент политики, последняя реплика клиента) для строки i — считается один раз."""
        if i not in fragments:
            transcript = parse_context(frame.prompt[i], frame.id[i])
            turn = parse_agent_turn(frame.response[i])
            fragments[i] = (policy_slice.relevant_policy(transcript, turn.text, tuple(c.name for c in turn.tool_calls)),
                            policy_slice.last_customer_message(transcript))
        return fragments[i]
    print(f"{args.name}/{args.model}: ходов {len(frame)} (текст {kinds.count('text')}, вызовы {kinds.count('call')}, "
          f"hard {kinds.count('hard')}); метаданные: "
          f"{pd.Series([how for _, how in specs]).value_counts().to_dict()}", flush=True)

    def user_text(signal: str, i: int, limit: int) -> Optional[str]:
        if signal == "meta":
            spec = specs[i][0]
            return None if spec is None else meta_judge.meta_message(render_spec(spec), frame.prompt[i], frame.response[i], limit)
        if signal in ("slice", "window", "focus"):
            text, customer = fragment(i)
            if signal == "slice":
                return policy_slice.slice_message(text, customer, frame.response[i])
            if signal == "window":
                return policy_slice.window_message(text, policy_slice.recent_window(frame.prompt[i]), frame.response[i])
            return policy_slice.focus_message(baseline.user_message(frame.prompt[i], frame.response[i], limit), text)
        return baseline.user_message(frame.prompt[i], frame.response[i], limit)

    stats = {}
    for signal in args.signals.split(","):
        checkpoint = Checkpoint(out / f"{args.name}_{args.model}_{signal}.jsonl")
        todo = [i for i in range(len(frame)) if frame.id[i] not in checkpoint.values and user_text(signal, i, 100) is not None]
        missing_digits = [0]

        def score(i: int) -> Optional[float]:
            for limit in (LIMIT, LIMIT // 2, LIMIT // 4):
                if api == "completions":
                    response = client.complete(judge_prompt(template, systems[signal], user_text(signal, i, limit), thinking=False),
                                               max_tokens=1, temperature=0.0, logprobs=20, seed=1, skip_special_tokens=False)
                    if response is None:
                        return None
                    if "error" in response:
                        continue
                    top = first_token_alternatives(response)
                else:
                    top = chat_top_logprobs(args.base_url, args.served, systems[signal], user_text(signal, i, limit),
                                            overrides=overrides)
                    if top is None:
                        return None
                    if "error" in top:
                        continue
                probability = digit_probability(top)
                if probability is None:
                    missing_digits[0] += 1
                return 0.5 if probability is None else probability
            return None

        started = time.time()
        results = map_concurrently(score, todo, args.workers, time.time() + 6 * 3600, f"{args.model}·{signal}")
        checkpoint.add({frame.id[todo[position]]: value for position, value in results.items()})
        stats[signal] = {"seconds": round(time.time() - started, 1), "scored": len(results), "todo": len(todo),
                         "no_digit_in_top": missing_digits[0], "api": api, "first_errors": ERRORS[:3]}
        ERRORS.clear()
        print(f"  {signal}: {stats[signal]}", flush=True)

    table = pd.DataFrame({"id": frame.id, "kind": kinds, "meta_how": [how for _, how in specs]})
    for signal in args.signals.split(","):
        values = Checkpoint(out / f"{args.name}_{args.model}_{signal}.jsonl").values
        table[f"{args.model}_{signal}"] = table.id.map(values)
    table.to_parquet(out / f"{args.name}_{args.model}_signals.parquet", index=False)
    (out / f"{args.name}_{args.model}_timings.json").write_text(json.dumps(stats), encoding="utf-8")


if __name__ == "__main__":
    main()
