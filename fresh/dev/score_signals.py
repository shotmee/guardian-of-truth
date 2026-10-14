"""Оценки всех сигналов на dev-наборе для выбора текстового ранжировщика (Colab, один уже поднятый vLLM).

Сигналы на каждый ход:
  p_base — промпт бейзлайна организаторов (все ходы);
  p_v1, p_v2 — старые судьи v1 и v2 (все ходы, запросы собирает код v2.1 из legacy_v2: подсказки
               [АВТОПРОВЕРКИ], у v2 ещё фрагменты контекста) — значения совпадают со старыми дампами;
  text — текстовые атомы veritas.text_atoms (только ходы без вызовов);
  calls — атомы по вызовам, стратегия C veritas.atoms (только ходы с вызовами без срабатываний проверок).
Детерминированные сигналы (hard-проверки, text_checks) считаются отдельно, локально.
Каждый сигнал пишется в свой jsonl (чекпоинт): повторный запуск досчитывает только недостающее.

python dev/score_signals.py --input examples.parquet --name tau2 --base-url http://127.0.0.1:8000/v1 \\
    --out <папка> [--signals base,v1,v2,text,calls]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))
from legacy_v2 import evidence as legacy_evidence  # noqa: E402
from legacy_v2 import prompts as legacy_prompts  # noqa: E402
from legacy_v2 import rules as legacy_rules  # noqa: E402
from veritas import atoms, claims, deliberate, text_atoms  # noqa: E402
from veritas.analysis import analyze_examples  # noqa: E402
from veritas.chat import ChatTemplate  # noqa: E402
from veritas.engine import CompletionClient, first_token_alternatives, map_concurrently  # noqa: E402
from veritas.transcript import parse_agent_turn  # noqa: E402

LEGACY_MAX_CHARS = 300000              # как в конфиге v2.1
NO_DEADLINE = 6 * 3600
CONFIG = {"workers": 96, "max_context_chars": 330000, "atoms": {"views": ["args", "policy", "step"], "max_calls": 6}}


def legacy_probability(top: Optional[dict]) -> Optional[float]:
    """P('1') по top-logprobs ровно как в v2.1 (pipeline.p_from_top): одна цифра в top — оценка по ней самой."""
    top = dict(top or {})
    yes = max([lp for token, lp in top.items() if (token or "").strip() == "1"], default=None)
    no = max([lp for token, lp in top.items() if (token or "").strip() == "0"], default=None)
    if yes is None and no is None:
        return None
    if no is None:
        return min(1.0 - 1e-6, max(0.501, math.exp(yes)))
    if yes is None:
        return max(1e-6, min(0.499, 1.0 - math.exp(no)))
    top_lp = max(yes, no)
    yes_weight, no_weight = math.exp(yes - top_lp), math.exp(no - top_lp)
    return yes_weight / (yes_weight + no_weight)


class Checkpoint:
    """jsonl {id, value} на один (набор, сигнал)."""

    def __init__(self, path: Path):
        self.path = path
        self.values: dict = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.values[record["id"]] = record["value"]

    def missing(self, ids: list) -> list:
        return [example_id for example_id in ids if example_id not in self.values]

    def add(self, results: dict) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            for example_id, value in results.items():
                self.values[example_id] = value
                output.write(json.dumps({"id": example_id, "value": value}, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- старые судьи (v2.1)
LEGACY_SIGNALS = {
    "base": ("baseline_prompt.md", "baseline"),
    "v1": ("judge_system_v1.md", "judge"),
    "v2": ("judge_system.md", "judge+evidence"),
}


def legacy_request(kind: str, row, limit: int) -> str:
    system_file, user_kind = LEGACY_SIGNALS[kind]
    system = legacy_prompts.load_text(system_file)
    if user_kind == "baseline":
        user = legacy_prompts.baseline_user_content(row.prompt, row.response, limit)
    else:
        analysis = legacy_rules.analyze(row.id, row.prompt, row.response)
        context = legacy_evidence.build(row.prompt, row.response) if user_kind == "judge+evidence" else ""
        user = legacy_prompts.user_content(row.prompt, row.response, analysis, limit, context)
    return legacy_prompts.chat(system, user, thinking=False)


def score_legacy(kind: str, frame: pd.DataFrame, client: CompletionClient, checkpoint: Checkpoint) -> None:
    todo = set(checkpoint.missing(frame.id.tolist()))
    rows = [row for row in frame.itertuples() if row.id in todo]
    rows.sort(key=lambda row: (row.id.split("::")[0], len(row.prompt)))   # соседние ходы диалога — общий префикс

    def answer(row) -> Optional[float]:
        for limit in (LEGACY_MAX_CHARS, LEGACY_MAX_CHARS // 2):
            response = client.complete(legacy_request(kind, row, limit), max_tokens=1, temperature=0.0, logprobs=20,
                                       seed=1, skip_special_tokens=False)
            if response is None:
                return None
            if "error" not in response:
                probability = legacy_probability(first_token_alternatives(response))
                return 0.5 if probability is None else probability
        return None

    results = map_concurrently(answer, rows, int(CONFIG["workers"]), time.time() + NO_DEADLINE, f"{kind}")
    checkpoint.add({rows[position].id: value for position, value in results.items()})


# ----------------------------------------------------------------------------- атомы veritas
def score_text_atoms(frame: pd.DataFrame, text_indices: list, client: CompletionClient, template: ChatTemplate,
                     checkpoint: Checkpoint) -> None:
    todo = set(checkpoint.missing([frame.id[i] for i in text_indices]))
    turns = [text_atoms.TextTurn(index=i, prompt=frame.prompt[i], text=parse_agent_turn(frame.response[i]).text)
             for i in text_indices if frame.id[i] in todo]
    turns.sort(key=lambda turn: (frame.id[turn.index].split("::")[0], len(turn.prompt)))
    results = text_atoms.run(turns, CONFIG, client, template, time.time() + NO_DEADLINE)
    checkpoint.add({frame.id[index]: result["answers"] for index, result in results.items()})


def _pending_text_turns(frame: pd.DataFrame, text_indices: list, checkpoint: Checkpoint) -> list:
    todo = set(checkpoint.missing([frame.id[i] for i in text_indices]))
    turns = [text_atoms.TextTurn(index=i, prompt=frame.prompt[i], text=parse_agent_turn(frame.response[i]).text)
             for i in text_indices if frame.id[i] in todo]
    return sorted(turns, key=lambda turn: (frame.id[turn.index].split("::")[0], len(turn.prompt)))


def score_claims(frame: pd.DataFrame, text_indices: list, client: CompletionClient, template: ChatTemplate,
                 checkpoint: Checkpoint) -> None:
    results = claims.run(_pending_text_turns(frame, text_indices, checkpoint), CONFIG, client, template,
                         time.time() + NO_DEADLINE)
    checkpoint.add({frame.id[index]: result for index, result in results.items()})


def score_deliberate(frame: pd.DataFrame, text_indices: list, client: CompletionClient, template: ChatTemplate,
                     checkpoint: Checkpoint) -> None:
    results = deliberate.run(_pending_text_turns(frame, text_indices, checkpoint), CONFIG, client, template,
                             time.time() + NO_DEADLINE)
    checkpoint.add({frame.id[index]: result for index, result in results.items()})


def score_call_atoms(frame: pd.DataFrame, candidates: list, client: CompletionClient, template: ChatTemplate,
                     checkpoint: Checkpoint) -> None:
    todo = set(checkpoint.missing([item.example_id for item in candidates]))
    pending = [item for item in candidates if item.example_id in todo]
    results = atoms.run(pending, CONFIG, client, template, time.time() + NO_DEADLINE)
    checkpoint.add({frame.id[index]: result for index, result in results.items()})


# ----------------------------------------------------------------------------- сборка
def turn_kinds(frame: pd.DataFrame, analysis) -> list:
    kinds = []
    for index in range(len(frame)):
        item = analysis.call_turns.get(index)
        kinds.append("text" if item is None else ("hard" if item.violations else "call"))
    return kinds


def assemble(frame: pd.DataFrame, kinds: list, checkpoints: dict) -> pd.DataFrame:
    table = pd.DataFrame({"id": frame.id, "kind": kinds})
    for name, checkpoint in checkpoints.items():
        values = table.id.map(checkpoint.values)
        if name == "text":
            for question in text_atoms.QUESTIONS:
                table[f"text_{question}"] = values.map(lambda v, q=question: v.get(q) if isinstance(v, dict) else None)
        elif name == "claims":
            for key in ("max", "mean", "count"):
                table[f"claims_{key}"] = values.map(lambda v, k=key: v.get(k) if isinstance(v, dict) else None)
            table["claims_detail"] = values.map(lambda v: json.dumps(v.get("claims"), ensure_ascii=False)
                                                if isinstance(v, dict) else None)
        elif name == "think":
            for key in ("p", "tokens", "truncated"):
                table[f"think_{key}"] = values.map(lambda v, k=key: v.get(k) if isinstance(v, dict) else None)
        elif name == "calls":
            table["calls_score"] = values.map(lambda v: v.get("score") if isinstance(v, dict) else None)
            table["calls_detail"] = values.map(lambda v: json.dumps(v.get("calls")) if isinstance(v, dict) else None)
        else:
            table[f"p_{name}"] = values
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True, help="имя набора для файлов чекпоинтов")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served", default="m")
    parser.add_argument("--template", default=str(ROOT / "src" / "veritas" / "assets" / "chat_template.jinja"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--signals", default="base,v1,v2,text,calls")
    parser.add_argument("--ids", default=None, help="json со списком id: считать только эти ходы")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).reset_index(drop=True)
    if args.ids:
        wanted = set(json.loads(Path(args.ids).read_text(encoding="utf-8")))
        frame = frame[frame.id.isin(wanted)].reset_index(drop=True)
    analysis = analyze_examples(frame)
    kinds = turn_kinds(frame, analysis)
    text_indices = [i for i, kind in enumerate(kinds) if kind == "text"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = CompletionClient(args.base_url, model=args.served)
    template = ChatTemplate(Path(args.template))
    print(f"{args.name}: ходов {len(frame)}, без вызовов {len(text_indices)}, с вызовами "
          f"{kinds.count('call')}, сработали проверки {kinds.count('hard')}", flush=True)

    timings, checkpoints = {}, {}
    for signal in args.signals.split(","):
        checkpoint = Checkpoint(out / f"{args.name}_{signal}.jsonl")
        checkpoints[signal] = checkpoint
        started = time.time()
        if signal in LEGACY_SIGNALS:
            score_legacy(signal, frame, client, checkpoint)
        elif signal == "text":
            score_text_atoms(frame, text_indices, client, template, checkpoint)
        elif signal == "claims":
            score_claims(frame, text_indices, client, template, checkpoint)
        elif signal == "think":
            score_deliberate(frame, text_indices, client, template, checkpoint)
        elif signal == "calls":
            score_call_atoms(frame, analysis.ranking_candidates, client, template, checkpoint)
        timings[signal] = round(time.time() - started, 1)
        print(f"  {signal}: {len(checkpoint.values)} оценок, {timings[signal]} с, ошибок HTTP {client.failed_requests}",
              flush=True)

    table = assemble(frame, kinds, checkpoints)
    table.to_parquet(out / f"{args.name}_signals.parquet", index=False)
    (out / f"{args.name}_timings.json").write_text(json.dumps({"n": len(frame), "text": len(text_indices),
                                                               "seconds": timings}, ensure_ascii=False))
    print(table.drop(columns=["calls_detail", "claims_detail"], errors="ignore").describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
