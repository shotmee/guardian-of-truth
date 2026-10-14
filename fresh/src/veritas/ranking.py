"""Ранжирование ходов моделью: сервер vLLM -> стратегия (A/B/C) -> бюджет флагов -> запись.

Решение одинаково для всех стратегий:
  1. ход без вызовов инструментов -> 0;
  2. сработала детерминированная проверка -> 1;
  3. остальные ходы с вызовами ранжируются по оценке стратегии (ходы только с чтением × read_weight)
     и помечаются лучшие, пока общее число флагов не достигнет round(budget_frac · N).

Защита попыток: если vLLM не поднялся, стратегия оценила слишком мало ходов или оценки вырождены,
процесс завершается с ошибкой без файла ответа (упавшее решение попытку не расходует).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .analysis import DatasetAnalysis, analyze_examples
from .chat import ChatTemplate
from .console import log
from .engine import CompletionClient, VllmServer
from .tables import read_examples, write_predictions

PACKAGE_DIR = Path(__file__).resolve().parent
EXIT_SERVER_FAILED, EXIT_LOW_COVERAGE, EXIT_DEGENERATE_SCORES, EXIT_NO_NATIVE_CALLS = 2, 3, 4, 5
MIN_SCORES_FOR_DEGENERACY_CHECK = 20
MIN_DISTINCT_SCORES = 3
MIN_NATIVE_CALL_SHARE = 0.15


def _tie_breaker(example_id: str) -> float:
    """Детерминированный порядок равных оценок, не зависящий от позиции в файле."""
    return int(hashlib.md5(example_id.encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def select_flags_by_budget(analysis: DatasetAnalysis, scores: dict, config: dict) -> set:
    budget = int(round(float(config.get("budget_frac", 0.185)) * analysis.size))
    read_weight = float(config.get("read_weight", 1.0))
    flags = set(analysis.violation_indices)
    ranked = []
    for index, score in scores.items():
        if index in flags or score is None:
            continue
        has_side_effects = analysis.call_turns[index].has_side_effects
        weighted = round(score * (1.0 if has_side_effects else read_weight), 9)
        ranked.append((weighted, has_side_effects, _tie_breaker(analysis.example_ids[index]), index))
    ranked.sort(reverse=True)
    for *_, index in ranked:
        if len(flags) >= budget:
            break
        flags.add(index)
    return flags


def _run_strategy(strategy: str, candidates: list, config: dict, client: CompletionClient,
                  template: ChatTemplate, deadline) -> dict:
    if strategy == "replay":
        from . import replay
        return replay.run(candidates, config, client, template, deadline("fast", 1150), deadline("llm", 1500))
    if strategy == "audit":
        from . import audit
        return audit.run(candidates, config, client, template, deadline("llm", 1450))
    if strategy == "atoms":
        from . import atoms
        return atoms.run(candidates, config, client, template, deadline("llm", 1550))
    raise SystemExit(f"неизвестная стратегия {strategy}")


def _check_results_or_exit(strategy: str, results: dict, candidates: list, config: dict) -> None:
    coverage = len(results) / max(1, len(candidates))
    if candidates and coverage < float(config.get("min_coverage", 0.5)):
        log(f"покрытие {coverage:.2f} ниже порога — выходим с ошибкой, попытка не расходуется")
        sys.exit(EXIT_LOW_COVERAGE)
    if len(results) >= MIN_SCORES_FOR_DEGENERACY_CHECK:
        distinct = len({round(float(r["score"]), 4) for r in results.values()})
        if distinct < MIN_DISTINCT_SCORES:
            log(f"оценки вырождены ({distinct} различных значений) — выходим с ошибкой, попытка не расходуется")
            sys.exit(EXIT_DEGENERATE_SCORES)
    if strategy == "replay" and results:
        samples = sum(len(r["q_fast"]) + len(r["q_think"]) for r in results.values())
        with_calls = sum(r.get("n_ref_calls", 0) for r in results.values())
        log(f"A: сэмплов {samples}, из них с вызовами инструментов {with_calls}")
        if samples and with_calls / samples < MIN_NATIVE_CALL_SHARE:
            log("модель почти не выдаёт вызовов в нативном формате — выходим с ошибкой")
            sys.exit(EXIT_NO_NATIVE_CALLS)


def run_ranking(args: argparse.Namespace, config: dict, strategy: str, process_start: float) -> None:
    deadlines = config.get("deadlines", {})

    def deadline(name: str, default_seconds: float) -> float:
        return process_start + float(deadlines.get(name, default_seconds))

    server = None
    base_url = args.base_url
    if base_url is None:                # сервер стартует первым: пока он грузит веса, разбираем данные
        server = VllmServer(args.model_path, port=int(config.get("port", 8000)),
                            max_model_len=int(config.get("max_model_len", 131072)),
                            gpu_memory_utilization=float(config.get("gpu_util", 0.90)),
                            log_path=os.path.join(tempfile.gettempdir(), "veritas_vllm.log"),
                            extra_arguments=config.get("vllm_extra", []))
        server.start()
        base_url = server.base_url
    try:
        frame = read_examples(args.input)
        if args.limit:
            frame = frame.head(args.limit).reset_index(drop=True)
        analysis = analyze_examples(frame)
        candidates = analysis.ranking_candidates
        budget = int(round(float(config.get("budget_frac", 0.185)) * analysis.size))
        log(f"примеров {analysis.size}, с вызовами {len(analysis.call_turns)}, сработали проверки "
            f"{len(analysis.violation_indices)}, к оценке {len(candidates)}, бюджет флагов {budget}")

        if server is not None and not server.wait_until_ready(deadline("server", 1000)):
            log("vLLM не поднялся — выходим с ошибкой, попытка не расходуется")
            sys.exit(EXIT_SERVER_FAILED)

        template_path = Path(args.model_path) / "chat_template.jinja"
        template = ChatTemplate(template_path if template_path.exists() else PACKAGE_DIR / "assets" / "chat_template.jinja")
        client = CompletionClient(base_url, model=config.get("served", "m"))
        client.hard_deadline = deadline("hard", 1590)

        strategy_start = time.time()
        results = _run_strategy(strategy, candidates, config, client, template, deadline)
        log(f"стратегия заняла {time.time() - strategy_start:.0f} с, оценено {len(results)}/{len(candidates)}, "
            f"ошибок HTTP {client.failed_requests}")
        _check_results_or_exit(strategy, results, candidates, config)

        flags = select_flags_by_budget(analysis, {i: r["score"] for i, r in results.items()}, config)
        write_predictions(analysis.example_ids, [int(i in flags) for i in range(analysis.size)], args.output)
        log(f"записано {analysis.size} ответов, флагов {len(flags)} (проверки {len(analysis.violation_indices)}), "
            f"{time.time() - process_start:.0f} с с начала")
        if args.dump:
            _write_dump(args.dump, strategy, analysis, results, flags, process_start)
    finally:
        if server is not None:
            server.stop()


def _write_dump(path: str, strategy: str, analysis: DatasetAnalysis, results: dict, flags: set,
                process_start: float) -> None:
    ids = analysis.example_ids
    details = {
        "strategy": strategy,
        "n": analysis.size,
        "hard": {ids[i]: [(v.rule, v.message) for v in analysis.call_turns[i].violations]
                 for i in sorted(analysis.violation_indices)},
        "conseq": {ids[i]: item.has_side_effects for i, item in analysis.call_turns.items()},
        "scores": {ids[i]: result for i, result in results.items()},
        "flags": [ids[i] for i in sorted(flags)],
        "seconds": time.time() - process_start,
    }
    Path(path).write_text(json.dumps(details, ensure_ascii=False, default=str), encoding="utf-8")
