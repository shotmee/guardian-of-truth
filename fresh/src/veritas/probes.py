"""Детерминированные зонды лидерборда: флаги без модели, по типу хода и срабатываниям проверок.

По TP и числу флагов на лидерборде они измеряют состав паблика:
    probe_hard      — только срабатывания проверок: их точность на паблике;
    probe_hardtext  — проверки + все ходы без вызовов: вместе с probe_hard даёт число позитивов в тексте;
    probe_text      — только ходы без вызовов;
    probe_process   — проверки + текстовые ходы с процессным сигналом (действие без вызова, «соединяю» без
                      перевода, обещание действия на устройстве, повторный запрос известного): точность сигналов;
    probe_history   — проверки + текстовые ходы без процессного сигнала, но с заметной ошибкой агента раньше
                      в диалоге: считают ли организаторы ошибкой продолжение прежней ошибки.
Hard-набор во всех зондах тот же, что в зонде №8 (18 флагов, 11 TP), поэтому TP текстовой части = TP − 11.
"""
from __future__ import annotations

from .analysis import DatasetAnalysis
from .text_checks import earlier_error_in_history, has_process_signal


def _process_turns(analysis: DatasetAnalysis) -> set:
    return {index for index, item in analysis.text_turns.items()
            if has_process_signal(item.transcript, item.turn)}


def _history_turns(analysis: DatasetAnalysis) -> set:
    return {index for index, item in analysis.text_turns.items()
            if not has_process_signal(item.transcript, item.turn)
            and earlier_error_in_history(item.transcript, item.tools)}


PROBES = {
    "probe_hard": lambda analysis: analysis.violation_indices,
    "probe_hardtext": lambda analysis: analysis.violation_indices | analysis.text_turn_indices,
    "probe_text": lambda analysis: analysis.text_turn_indices,
    "probe_process": lambda analysis: analysis.violation_indices | _process_turns(analysis),
    "probe_history": lambda analysis: analysis.violation_indices | _history_turns(analysis),
}


def is_probe(strategy: str) -> bool:
    return strategy in PROBES


def probe_flags(strategy: str, analysis: DatasetAnalysis) -> set:
    return set(PROBES[strategy](analysis))
