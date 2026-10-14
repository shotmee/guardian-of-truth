# Deterministic leaderboard probes: flag turns without using the model.
#
# These probe pipelines combine rule hits with turn shape to shed light on
# the public leaderboard composition:
#
#     probe_hard      -- only rule firings. Measures the precision of
#                        deterministic firings on the public set.
#     probe_hardtext  -- rule firings together with every turn that has no
#                        tool calls. Together with ``probe_hard`` this
#                        reveals how many positives sit in the text part.
#     probe_text      -- only turns without tool calls.
#     probe_process   -- rule firings plus text turns that carry a process
#                        signal (acting without a call, "connecting" without
#                        the transfer tool, promising a device-side change,
#                        re-asking a known fact). Gives signal precision.
#     probe_history   -- rule firings plus text turns that don't carry a
#                        process signal but follow a visible earlier mistake
#                        by the agent. Tells us whether the organisers mark
#                        the continuation of an earlier mistake as an error.
#
# Across every probe the hard set is the same (matches probe No. 8:
# 18 flags, 11 TP), so the text-side TP is TP minus 11.

from .text_checks import earlier_error_in_history, has_process_signal


def _carries_process_signal(analysis):
    out = set()
    for i, item in analysis.text_turns.items():
        if has_process_signal(item.transcript, item.turn):
            out.add(i)
    return out


def _follows_prior_error(analysis):
    out = set()
    for i, item in analysis.text_turns.items():
        if has_process_signal(item.transcript, item.turn):
            continue
        if earlier_error_in_history(item.transcript, item.tools):
            out.add(i)
    return out


PROBES = {
    "probe_hard":     lambda a: set(a.violation_indices),
    "probe_hardtext": lambda a: set(a.violation_indices) | set(a.text_turn_indices),
    "probe_text":     lambda a: set(a.text_turn_indices),
    "probe_process":  lambda a: set(a.violation_indices) | _carries_process_signal(a),
    "probe_history":  lambda a: set(a.violation_indices) | _follows_prior_error(a),
}


def is_probe(strategy):
    return strategy in PROBES


def probe_flags(strategy, analysis):
    return set(PROBES[strategy](analysis))
