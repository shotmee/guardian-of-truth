# The "final" pipeline: hard rules + atoms over call turns + ranking over
# text-only turns.
#
# Flag sources:
#   1. hard   - a rule fired on the audited turn.
#   2. calls  - the remaining call turns, top ``call_frac`` by atoms score
#               (with ``config["call_shots"]``: logit of atoms + logit of
#               P('1') under the baseline prompt with the organisers'
#               labelled examples; this step runs last and is dropped if it
#               scored fewer than 90% of the call turns).
#   3. text   - the text-only turns, top ``text_frac`` by text score
#               (controlled by ``config["text_ranker"]``):
#                 "base"              -> logit of P_base alone;
#                 "base+atoms+groups" -> logit P_base + logit of text-atoms
#                                        mean + group bias (process signal /
#                                        earlier error).
# Shares are relative to the number of turns of each type (30/148 and 160/380
# on the public split).
# Importance order of signals: P_base, then atoms over calls, then text
# atoms; if text atoms didn't finish, text is ranked by P_base alone.
# Submission guards: if vLLM never comes up, or P_base scored fewer than
# half of the text turns, the process exits with an error and writes no
# output - a failed submission does not consume an attempt.

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from . import atoms, baseline, text_atoms
from .analysis import DatasetAnalysis, analyze_examples
from .chat import ChatTemplate
from .console import log
from .engine import CompletionClient, VllmServer
from .tables import read_examples, write_predictions
from .text_checks import earlier_error_in_history, has_process_signal


_HERE = Path(__file__).resolve().parent
EXIT_SERVER_FAILED = 2
EXIT_LOW_COVERAGE = 3
MIN_TEXT_COVERAGE = 0.5
MIN_ATOMS_COVERAGE = 0.9


class TextExample:
    __slots__ = ("index", "prompt", "response")

    def __init__(self, index, prompt, response):
        self.index = index
        self.prompt = prompt
        self.response = response


def _tie_breaker(example_id):
    digest = hashlib.md5(example_id.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16 ** 12)


def _logit(probability):
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _top(scores, share, example_ids):
    """Pick top ``round(share * n)`` indices by score; tie-break by hashed id
    so the result is deterministic."""
    want = int(round(share * len(scores)))
    sorted_indices = sorted(
        scores,
        key=lambda idx: (scores[idx], _tie_breaker(example_ids[idx])),
        reverse=True,
    )
    return set(sorted_indices[:want])


def group_shift(item, shifts):
    if has_process_signal(item.transcript, item.turn):
        return float(shifts.get("process", 0.0))
    if earlier_error_in_history(item.transcript, item.tools):
        return float(shifts.get("history", 0.0))
    return 0.0


def text_scores(analysis, base, atom_results, config):
    """Return ``({row_index: text_score}, ranker_name_used)``."""
    ranker = config.get("text_ranker", "base")
    base_logits = {i: _logit(p) for i, p in base.items() if p is not None}
    if ranker == "base":
        return base_logits, "base"

    coverage = len(atom_results) / max(1, len(base_logits))
    if coverage < MIN_ATOMS_COVERAGE:
        log("текстовые атомы оценили %.0f%% ходов — ранжирую текст по одному P_base" % (coverage * 100))
        return base_logits, "base (атомы не успели)"

    shifts = config.get("group_logit_shift", {})
    combined = {}
    for idx, bl in base_logits.items():
        atom_score = atom_results.get(idx, {}).get("score")
        atom_logit = _logit(atom_score) if atom_score is not None else 0.0
        combined[idx] = bl + atom_logit + group_shift(analysis.text_turns[idx], shifts)
    return combined, ranker


def select_flags(analysis, text, calls, config, call_examples=None):
    hard = set(analysis.violation_indices)
    call_scores = {}
    for item in analysis.ranking_candidates:
        score = _logit(float(calls.get(item.index, {}).get("score", 0.0)))
        if call_examples and item.index in call_examples:
            score += _logit(call_examples[item.index])
        call_scores[item.index] = score
    call_flags = _top(call_scores, float(config.get("call_frac", 0.203)), analysis.example_ids)
    text_flags = _top(text, float(config.get("text_frac", 0.421)), analysis.example_ids)
    return {"hard": hard, "calls": call_flags, "text": text_flags}


def _start_server(args, config):
    if args.base_url is not None:
        return None, args.base_url
    server = VllmServer(
        args.model_path,
        port=int(config.get("port", 8000)),
        max_model_len=int(config.get("max_model_len", 131072)),
        gpu_memory_utilization=float(config.get("gpu_util", 0.90)),
        log_path=os.path.join(tempfile.gettempdir(), "veritas_vllm.log"),
        extra_arguments=config.get("vllm_extra", []),
    )
    server.start()
    return server, server.base_url


def _write_dump(path, used_ranker, process_start, base, atom_results, calls, scores, flags, ids,
                call_examples=None):
    payload = {
        "text_ranker": used_ranker,
        "seconds": time.time() - process_start,
        "base": {ids[i]: p for i, p in base.items()},
        "text_atoms": {ids[i]: r for i, r in atom_results.items()},
        "calls": {ids[i]: r.get("score") for i, r in calls.items()},
        "call_examples": {ids[i]: p for i, p in (call_examples or {}).items()},
        "text_scores": {ids[i]: s for i, s in scores.items()},
        "flags": {name: [ids[i] for i in sorted(group)] for name, group in flags.items()},
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, default=str),
                          encoding="utf-8")


def run_final(args, config, process_start):
    deadlines = config.get("deadlines", {})

    def _deadline(name, default_seconds):
        return process_start + float(deadlines.get(name, default_seconds))

    server, base_url = _start_server(args, config)
    try:
        frame = read_examples(args.input)
        if args.limit:
            frame = frame.head(args.limit).reset_index(drop=True)
        analysis = analyze_examples(frame)

        texts = []
        for i, item in sorted(analysis.text_turns.items()):
            texts.append(TextExample(
                index=i, prompt=item.prompt,
                response=str(frame["response"].iloc[i]),
            ))

        log("примеров %d: текстовых %d, с вызовами %d "
            "(сработали проверки %d); ранжировщик текста %s" % (
                analysis.size, len(texts), len(analysis.call_turns),
                len(analysis.violation_indices), config.get("text_ranker"),
            ))

        if server is not None and not server.wait_until_ready(_deadline("server", 1000)):
            log("vLLM не поднялся — выходим с ошибкой, попытка не расходуется")
            sys.exit(EXIT_SERVER_FAILED)

        tpl_path = Path(args.model_path) / "chat_template.jinja"
        tpl_file = tpl_path if tpl_path.exists() else (_HERE / "assets" / "chat_template.jinja")
        template = ChatTemplate(tpl_file)

        client = CompletionClient(base_url, model=config.get("served", "m"))
        client.hard_deadline = _deadline("hard", 1590)

        step_begin = time.time()
        base = baseline.run(texts, config, client, template, _deadline("base", 1250))
        log("P_base: %d/%d за %.0f с" % (len(base), len(texts), time.time() - step_begin))

        if texts and len(base) / len(texts) < MIN_TEXT_COVERAGE:
            log("P_base оценил слишком мало ходов — выходим с ошибкой, попытка не расходуется")
            sys.exit(EXIT_LOW_COVERAGE)

        step_begin = time.time()
        calls = atoms.run(analysis.ranking_candidates, config, client, template, _deadline("calls", 1380))
        log("атомы вызовов: %d/%d за %.0f с" % (
            len(calls), len(analysis.ranking_candidates), time.time() - step_begin,
        ))

        atom_results = {}
        if config.get("text_ranker", "base") != "base":
            step_begin = time.time()
            turns = []
            for t in texts:
                if t.index not in base:
                    continue
                turns.append(text_atoms.TextTurn(
                    index=t.index, prompt=t.prompt,
                    text=analysis.text_turns[t.index].turn.text,
                ))
            atom_results = text_atoms.run(turns, config, client, template, _deadline("text_atoms", 1500))
            log("текстовые атомы: %d/%d за %.0f с" % (
                len(atom_results), len(turns), time.time() - step_begin,
            ))

        call_examples = {}
        if config.get("call_shots"):
            step_begin = time.time()
            call_items = []
            for item in analysis.ranking_candidates:
                call_items.append(TextExample(
                    index=item.index, prompt=item.prompt,
                    response=str(frame["response"].iloc[item.index]),
                ))
            call_examples = baseline.run(call_items, config, client, template,
                                         _deadline("call_shots", 1560), with_examples=True)
            log("примеры организаторов по вызовам: %d/%d за %.0f с" % (
                len(call_examples), len(call_items), time.time() - step_begin,
            ))
            if call_items and len(call_examples) / len(call_items) < MIN_ATOMS_COVERAGE:
                log("примеры по вызовам не успели — вызовы ранжируются по одним атомам")
                call_examples = {}

        scores, used_ranker = text_scores(analysis, base, atom_results, config)
        flags = select_flags(analysis, scores, calls, config, call_examples)
        flagged = flags["hard"] | flags["calls"] | flags["text"]
        labels = [int(i in flagged) for i in range(analysis.size)]
        write_predictions(analysis.example_ids, labels, args.output)
        log("записано %d ответов, флагов %d: hard %d, "
            "вызовы %d, текст %d (%s); %.0f с с начала, "
            "ошибок HTTP %d" % (
                analysis.size, len(flagged), len(flags["hard"]),
                len(flags["calls"]), len(flags["text"]), used_ranker,
                time.time() - process_start, client.failed_requests,
            ))

        if args.dump:
            _write_dump(args.dump, used_ranker, process_start,
                        base, atom_results, calls, scores, flags,
                        analysis.example_ids, call_examples)
    finally:
        if server is not None:
            server.stop()
