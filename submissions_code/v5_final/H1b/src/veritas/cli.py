# Entry point used by the platform.
#
#   python scripts/predict.py --input <test> --output <predictions> [--config ...]
#
# The strategy name is taken from config.json: either a probe (no model) or a
# full model-based ranking.

import argparse
import json
import pathlib
import time

from .analysis import analyze_examples
from .console import log
from .probes import is_probe, probe_flags
from .tables import read_examples, write_predictions

_HERE = pathlib.Path(__file__).resolve().parent
_T0 = time.time()


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Guardian of Truth: LLM agent error detector",
    )
    parser.add_argument("--input", required=True,
                        help="test dataset (parquet or csv)")
    parser.add_argument("--output", required=True,
                        help="where to write the predictions")
    parser.add_argument("--config", default=str(_HERE / "config.json"))
    parser.add_argument("--model-path", default=str(_HERE.parents[1] / "model"))
    parser.add_argument("--base-url", default=None,
                        help="an already running vLLM (for development)")
    parser.add_argument("--strategy", default=None,
                        help="override the strategy from the config")
    parser.add_argument("--dump", default=None,
                        help="json with run details (for development)")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N rows")
    return parser


def parse_arguments(argv=None):
    return _build_parser().parse_args(argv)


def _apply_limit(frame, limit):
    if not limit:
        return frame
    return frame.head(limit).reset_index(drop=True)


def _dump_probe_run(args, strategy, analysis, flags):
    if not args.dump:
        return
    hard_detail = {}
    for item in analysis.call_turns.values():
        if item.violations:
            hard_detail[item.example_id] = [(v.rule, v.message) for v in item.violations]
    payload = {
        "strategy": strategy,
        "n": analysis.size,
        "flags": [analysis.example_ids[i] for i in sorted(flags)],
        "hard": hard_detail,
    }
    pathlib.Path(args.dump).write_text(json.dumps(payload, ensure_ascii=False),
                                       encoding="utf-8")


def run_probe(args, strategy):
    frame = _apply_limit(read_examples(args.input), args.limit)
    analysis = analyze_examples(frame)
    flags = probe_flags(strategy, analysis)
    labels = [int(i in flags) for i in range(analysis.size)]
    write_predictions(analysis.example_ids, labels, args.output)

    log("зонд %s: записано %d ответов, флагов %d "
        "(ходов без вызовов %d, с вызовами %d, "
        "сработали проверки %d)" % (
            strategy, analysis.size, len(flags),
            len(analysis.text_turn_indices), len(analysis.call_turns),
            len(analysis.violation_indices),
        ))
    _dump_probe_run(args, strategy, analysis, flags)


def main(argv=None):
    args = parse_arguments(argv)
    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    strategy = args.strategy if args.strategy else config["strategy"]
    log("стратегия %s, конфиг %s" % (strategy, args.config))

    if is_probe(strategy):
        run_probe(args, strategy)
        return

    if strategy == "final":
        from .final import run_final
        run_final(args, config, _T0)
        return

    from .ranking import run_ranking
    run_ranking(args, config, strategy, _T0)


if __name__ == "__main__":
    main()
