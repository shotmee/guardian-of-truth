"""Разбор всех примеров: ход агента, вызовы инструментов, срабатывания детерминированных проверок."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .checks import ConversationState, find_violations, has_side_effects
from .console import log
from .tool_schema import parse_available_tools
from .transcript import Transcript, Turn, parse_agent_turn, parse_context


@dataclass
class TurnAnalysis:
    index: int                         # номер строки во входном файле
    example_id: str
    prompt: str
    transcript: Transcript
    turn: Turn
    tools: dict
    has_side_effects: bool
    violations: list = field(default_factory=list)


@dataclass
class DatasetAnalysis:
    size: int
    example_ids: list
    call_turns: dict                   # номер строки -> TurnAnalysis, только ходы с вызовами инструментов
    text_turns: dict = field(default_factory=dict)    # номер строки -> TurnAnalysis, разобранные ходы без вызовов

    @property
    def text_turn_indices(self) -> set:
        """Ходы без вызовов инструментов (и строки, которые не удалось разобрать)."""
        return set(range(self.size)) - set(self.call_turns)

    @property
    def violation_indices(self) -> set:
        return {index for index, item in self.call_turns.items() if item.violations}

    @property
    def ranking_candidates(self) -> list:
        """Ходы с вызовами, на которых проверки ничего не нашли: их ранжирует стратегия."""
        return [item for item in self.call_turns.values() if not item.violations]


def analyze_examples(frame: pd.DataFrame) -> DatasetAnalysis:
    call_turns: dict = {}
    text_turns: dict = {}
    rows = zip(frame["id"], frame["prompt"], frame["response"])
    for index, (example_id, prompt, response) in enumerate(rows):
        try:
            transcript = parse_context(str(prompt), str(example_id))
            turn = parse_agent_turn(str(response))
        except Exception as error:  # noqa: BLE001 — один битый пример не должен ронять весь прогон
            log(f"не удалось разобрать {example_id}: {error!r}")
            continue
        tools = parse_available_tools(transcript.tools_text)
        if not turn.has_tool_calls:
            text_turns[index] = TurnAnalysis(index=index, example_id=str(example_id), prompt=str(prompt),
                                             transcript=transcript, turn=turn, tools=tools, has_side_effects=False)
            continue
        try:
            violations = find_violations(transcript, turn, tools, ConversationState(transcript))
        except Exception as error:  # noqa: BLE001
            log(f"проверки упали на {example_id}: {error!r}")
            violations = []
        call_turns[index] = TurnAnalysis(index=index, example_id=str(example_id), prompt=str(prompt),
                                         transcript=transcript, turn=turn, tools=tools,
                                         has_side_effects=has_side_effects(turn), violations=violations)
    return DatasetAnalysis(size=len(frame), example_ids=[str(x) for x in frame["id"]], call_turns=call_turns,
                           text_turns=text_turns)
