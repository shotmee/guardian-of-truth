"""Точка входа: python scripts/predict.py --input <test> --output <predictions> [--config ...].

Стратегия берётся из config.json: детерминированный зонд (без модели) или ранжирование ходов моделью.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .analysis import analyze_examples
from .console import log
from .probes import is_probe, probe_flags
from .tables import read_examples, write_predictions

PACKAGE_DIR = Path(__file__).resolve().parent
PROCESS_START = time.time()


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guardian of Truth: детектор ошибок агента")
    parser.add_argument("--input", required=True, help="тестовый датасет (parquet или csv)")
    parser.add_argument("--output", required=True, help="куда записать предсказания")
    parser.add_argument("--config", default=str(PACKAGE_DIR / "config.json"))
    parser.add_argument("--model-path", default=str(PACKAGE_DIR.parents[1] / "model"))
    parser.add_argument("--base-url", default=None, help="уже поднятый vLLM (для разработки)")
    parser.add_argument("--strategy", default=None, help="переопределить стратегию из конфига")
    parser.add_argument("--dump", default=None, help="json с подробностями прогона (для разработки)")
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N примеров")
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace, strategy: str) -> None:
    frame = read_examples(args.input)
    if args.limit:
        frame = frame.head(args.limit).reset_index(drop=True)
    analysis = analyze_examples(frame)
    flags = probe_flags(strategy, analysis)
    write_predictions(analysis.example_ids, [int(i in flags) for i in range(analysis.size)], args.output)
    log(f"зонд {strategy}: записано {analysis.size} ответов, флагов {len(flags)} "
        f"(ходов без вызовов {len(analysis.text_turn_indices)}, с вызовами {len(analysis.call_turns)}, "
        f"сработали проверки {len(analysis.violation_indices)})")
    if args.dump:
        details = {
            "strategy": strategy,
            "n": analysis.size,
            "flags": [analysis.example_ids[i] for i in sorted(flags)],
            "hard": {item.example_id: [(v.rule, v.message) for v in item.violations]
                     for item in analysis.call_turns.values() if item.violations},
        }
        Path(args.dump).write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> None:
    args = parse_arguments(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    strategy = args.strategy or config["strategy"]
    log(f"стратегия {strategy}, конфиг {args.config}")
    if is_probe(strategy):
        run_probe(args, strategy)
        return
    if strategy == "final":
        from .final import run_final   # модули с моделью нужны только веткам с ранжированием
        run_final(args, config, PROCESS_START)
        return
    from .ranking import run_ranking
    run_ranking(args, config, strategy, PROCESS_START)


if __name__ == "__main__":
    main()
