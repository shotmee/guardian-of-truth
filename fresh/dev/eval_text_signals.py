"""Оценка детерминированных текстовых сигналов (veritas.text_checks) на размеченных текстовых ходах.

Метки: разметка v5 (labeling_v5/out) для ходов пула, метки организаторов для ходов valid.
Для каждого сигнала: срабатывания, сколько из них ошибки, и как сработавшие ошибки распределены по P_base
(промпт бейзлайна, из дампов v2_main) — новый сигнал полезен там, где P_base ниже 0.8.

python dev/eval_text_signals.py [--show имя_сигнала]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from veritas.text_checks import SIGNAL_NAMES, find_text_signals  # noqa: E402
from veritas.transcript import parse_agent_turn, parse_context  # noqa: E402


def load_labeled_text_turns() -> pd.DataFrame:
    units = pd.DataFrame(json.loads((PROJECT / "labeling_v5" / "units.json").read_text(encoding="utf-8")))
    records = [json.loads(line) for path in sorted((PROJECT / "labeling_v5" / "out").glob("*.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = pd.DataFrame(records).drop_duplicates("id", keep="last")[["id", "label", "category"]]
    dev = pd.read_parquet(ROOT / "dev" / "data" / "dev.parquet")[["id", "prompt", "response"]]
    frame = units[units.kind == "text"].merge(labels, on="id").merge(dev, on="id")
    organizer = pd.read_parquet(PROJECT / "valid.parquet").set_index("id")["label"]
    frame["label"] = [int(organizer[i]) if source == "valid" else int(label)
                      for i, source, label in zip(frame.id, frame.src, frame.label)]
    return frame


def baseline_scores() -> dict:
    scores = {}
    for name in ("v2_pool", "v2_valid"):
        dump = pd.read_parquet(PROJECT / "colab_results" / "v2_main" / "results" / f"{name}.parquet")
        for example_id, extra in zip(dump["id"], dump["p_extra"]):
            extra = extra if isinstance(extra, dict) else json.loads(extra or "{}")
            scores[example_id] = extra.get("base")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default=None, help="показать срабатывания сигнала")
    args = parser.parse_args()

    frame = load_labeled_text_turns()
    frame["p_base"] = frame.id.map(baseline_scores())
    evidence = {}
    for row in frame.itertuples():
        signals = find_text_signals(parse_context(row.prompt, row.id), parse_agent_turn(row.response))
        for name in SIGNAL_NAMES:
            frame.loc[row.Index, name] = any(signal.name == name for signal in signals)
        evidence[row.id] = {signal.name: signal.evidence for signal in signals}
    frame["any_process"] = frame[[n for n in SIGNAL_NAMES if n != "offers_transfer"]].any(axis=1)
    frame["any_signal"] = frame[list(SIGNAL_NAMES)].any(axis=1)

    positives = int(frame.label.sum())
    print(f"текстовых ходов {len(frame)}, ошибок {positives} (база {frame.label.mean():.2f})\n")
    print(f"{'сигнал':24s} {'сраб.':>5s} {'ошибок':>6s} {'точн.':>5s}   ошибки по P_base: ≥0.8 / 0.5–0.8 / <0.5")
    for name in (*SIGNAL_NAMES, "any_process", "any_signal"):
        fired = frame[frame[name]]
        hits = fired[fired.label == 1]
        bins = (int((hits.p_base >= 0.8).sum()), int(((hits.p_base >= 0.5) & (hits.p_base < 0.8)).sum()),
                int((hits.p_base < 0.5).sum()))
        precision = hits.shape[0] / fired.shape[0] if len(fired) else float("nan")
        print(f"{name:24s} {len(fired):5d} {len(hits):6d} {precision:5.2f}   {bins[0]} / {bins[1]} / {bins[2]}")

    if args.show:
        print(f"\nсрабатывания {args.show}:")
        for row in frame[frame[args.show]].itertuples():
            print(f"  [{row.label}] {row.id} (P_base {row.p_base}): {evidence[row.id].get(args.show, '')}")


if __name__ == "__main__":
    main()
