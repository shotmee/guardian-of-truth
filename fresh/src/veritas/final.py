"""Финальная стратегия: hard-проверки + «Атомы» на ходах с вызовами + ранжирование текстовых ходов.

Флаги:
  1. hard — на ходе с вызовами сработала детерминированная проверка;
  2. остальные ходы с вызовами — лучшие call_frac по «Атомам» (atoms.py);
  3. ходы без вызовов — лучшие text_frac по оценке текста (config["text_ranker"]):
       "base"              — logit P_base (промпт бейзлайна организаторов);
       "base+atoms+groups" — logit P_base + logit(среднее текстовых атомов) + сдвиг группы зондов
                             (процессный сигнал в ходе / заметная ошибка раньше в диалоге).
Доли считаются от числа ходов своего типа (на паблике это 30 из 148 и 160 из 380).
Порядок оценок — по важности: P_base, затем «Атомы» на вызовах, затем текстовые атомы; если текстовые
атомы не успели, текст ранжируется по одному P_base.
Защита попыток: vLLM не поднялся или P_base оценил меньше половины текстовых ходов — выход с ошибкой без
файла ответа (упавший запуск попытку не расходует).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import atoms, baseline, text_atoms
from .analysis import DatasetAnalysis, analyze_examples
from .chat import ChatTemplate
from .console import log
from .engine import CompletionClient, VllmServer
from .tables import read_examples, write_predictions
from .text_checks import earlier_error_in_history, has_process_signal

PACKAGE_DIR = Path(__file__).resolve().parent
EXIT_SERVER_FAILED, EXIT_LOW_COVERAGE = 2, 3
MIN_TEXT_COVERAGE = 0.5
MIN_ATOMS_COVERAGE = 0.9               # текстовые атомы учитываются, только если оценены почти все ходы


@dataclass(frozen=True)
class TextExample:
    index: int
    prompt: str
    response: str


def _tie_breaker(example_id: str) -> float:
    return int(hashlib.md5(example_id.encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def _logit(probability) -> float:
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _top(scores: dict, share: float, example_ids: list) -> set:
    """Лучшие round(share · n) элементов по оценке; равные упорядочены детерминированно по id."""
    count = int(round(share * len(scores)))
    ranked = sorted(scores, key=lambda index: (scores[index], _tie_breaker(example_ids[index])), reverse=True)
    return set(ranked[:count])


def group_shift(item, shifts: dict) -> float:
    if has_process_signal(item.transcript, item.turn):
        return float(shifts.get("process", 0.0))
    if earlier_error_in_history(item.transcript, item.tools):
        return float(shifts.get("history", 0.0))
    return 0.0


def text_scores(analysis: DatasetAnalysis, base: dict, atom_results: dict, config: dict) -> tuple:
    """({номер строки: оценка текста}, использованный ранжировщик)."""
    ranker = config.get("text_ranker", "base")
    scored = {index: _logit(p) for index, p in base.items() if p is not None}
    if ranker == "base":
        return scored, "base"
    coverage = len(atom_results) / max(1, len(scored))
    if coverage < MIN_ATOMS_COVERAGE:
        log(f"текстовые атомы оценили {coverage:.0%} ходов — ранжирую текст по одному P_base")
        return scored, "base (атомы не успели)"
    shifts = config.get("group_logit_shift", {})
    combined = {}
    for index, base_logit in scored.items():
        atom_score = atom_results.get(index, {}).get("score")
        atom_logit = _logit(atom_score) if atom_score is not None else 0.0
        combined[index] = base_logit + atom_logit + group_shift(analysis.text_turns[index], shifts)
    return combined, ranker


def select_flags(analysis: DatasetAnalysis, text: dict, calls: dict, config: dict) -> dict:
    hard = set(analysis.violation_indices)
    call_scores = {item.index: float(calls.get(item.index, {}).get("score", 0.0))
                   for item in analysis.ranking_candidates}
    call_flags = _top(call_scores, float(config.get("call_frac", 0.203)), analysis.example_ids)
    text_flags = _top(text, float(config.get("text_frac", 0.421)), analysis.example_ids)
    return {"hard": hard, "calls": call_flags, "text": text_flags}


def run_final(args: argparse.Namespace, config: dict, process_start: float) -> None:
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
        texts = [TextExample(index=i, prompt=item.prompt, response=str(frame["response"].iloc[i]))
                 for i, item in sorted(analysis.text_turns.items())]
        log(f"примеров {analysis.size}: текстовых {len(texts)}, с вызовами {len(analysis.call_turns)} "
            f"(сработали проверки {len(analysis.violation_indices)}); ранжировщик текста {config.get('text_ranker')}")

        if server is not None and not server.wait_until_ready(deadline("server", 1000)):
            log("vLLM не поднялся — выходим с ошибкой, попытка не расходуется")
            sys.exit(EXIT_SERVER_FAILED)
        template_path = Path(args.model_path) / "chat_template.jinja"
        template = ChatTemplate(template_path if template_path.exists() else PACKAGE_DIR / "assets" / "chat_template.jinja")
        client = CompletionClient(base_url, model=config.get("served", "m"))
        client.hard_deadline = deadline("hard", 1590)

        started = time.time()
        base = baseline.run(texts, config, client, template, deadline("base", 1250))
        log(f"P_base: {len(base)}/{len(texts)} за {time.time() - started:.0f} с")
        if texts and len(base) / len(texts) < MIN_TEXT_COVERAGE:
            log("P_base оценил слишком мало ходов — выходим с ошибкой, попытка не расходуется")
            sys.exit(EXIT_LOW_COVERAGE)

        started = time.time()
        calls = atoms.run(analysis.ranking_candidates, config, client, template, deadline("calls", 1380))
        log(f"атомы вызовов: {len(calls)}/{len(analysis.ranking_candidates)} за {time.time() - started:.0f} с")

        atom_results: dict = {}
        if config.get("text_ranker", "base") != "base":
            started = time.time()
            turns = [text_atoms.TextTurn(index=t.index, prompt=t.prompt, text=analysis.text_turns[t.index].turn.text)
                     for t in texts if t.index in base]
            atom_results = text_atoms.run(turns, config, client, template, deadline("text_atoms", 1500))
            log(f"текстовые атомы: {len(atom_results)}/{len(turns)} за {time.time() - started:.0f} с")

        scores, used_ranker = text_scores(analysis, base, atom_results, config)
        flags = select_flags(analysis, scores, calls, config)
        flagged = flags["hard"] | flags["calls"] | flags["text"]
        write_predictions(analysis.example_ids, [int(i in flagged) for i in range(analysis.size)], args.output)
        log(f"записано {analysis.size} ответов, флагов {len(flagged)}: hard {len(flags['hard'])}, "
            f"вызовы {len(flags['calls'])}, текст {len(flags['text'])} ({used_ranker}); "
            f"{time.time() - process_start:.0f} с с начала, ошибок HTTP {client.failed_requests}")
        if args.dump:
            ids = analysis.example_ids
            Path(args.dump).write_text(json.dumps({
                "text_ranker": used_ranker, "seconds": time.time() - process_start,
                "base": {ids[i]: p for i, p in base.items()},
                "text_atoms": {ids[i]: r for i, r in atom_results.items()},
                "calls": {ids[i]: r.get("score") for i, r in calls.items()},
                "text_scores": {ids[i]: s for i, s in scores.items()},
                "flags": {name: [ids[i] for i in sorted(group)] for name, group in flags.items()},
            }, ensure_ascii=False, default=str), encoding="utf-8")
    finally:
        if server is not None:
            server.stop()
