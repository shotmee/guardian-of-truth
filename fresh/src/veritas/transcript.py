"""Разбор транскрипта диалога агента.

Формат данных организаторов:
    ⟦SYSTEM⟧ / ⟦USER⟧ / ⟦ASSISTANT⟧ / ⟦ASSISTANT · ход N⟧   маркер реплики на отдельной строке
    \\t→ TOOL_CALL name: {json}                               вызов инструмента (агента или клиента)
    \\t← TOOL_RESPONSE name[ [ERROR]]: text                   ответ среды
    \\t  ...                                                  продолжение многострочного вызова/ответа

`parse_context(prompt)` разбирает всё, что агент видел до своего хода, `parse_agent_turn(response)` —
сам проверяемый ход.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

MARKER_LINE = re.compile(r"(?m)^(⟦[^⟧\n]*⟧)[ \t]*$")
TOOL_CALL_LINE = re.compile(r"^\t→ TOOL_CALL ([^:\s]+):\s?(.*)$")
TOOL_RESPONSE_LINE = re.compile(r"^\t← TOOL_RESPONSE ([^:\s\[]+)( \[ERROR\])?:\s?(.*)$")
CONTINUATION_PREFIX = "\t  "
TURN_NUMBER = re.compile(r"ход\s+(\d+)")
POLICY_TIME = re.compile(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TOOL_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
ERROR_RESULT = re.compile(r"^\s*(error\b|exception\b|tool '[^']*' not found)", re.IGNORECASE)
TOOLS_SECTION = "[AVAILABLE TOOLS]"
KNOWN_DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")

AGENT = "assistant"
CUSTOMER = "user"
SYSTEM = "system"
CHECKED_TURN_INDEX = -1


@dataclass
class ToolCall:
    name: str
    raw_arguments: str
    arguments: Optional[dict]          # None, если аргументы не разобрались как JSON-объект
    caller: str                        # AGENT или CUSTOMER
    turn_index: int                    # номер реплики в диалоге, CHECKED_TURN_INDEX для проверяемого хода
    result: Optional[str] = None
    marked_error: bool = False         # среда пометила ответ как [ERROR]

    @property
    def has_valid_arguments(self) -> bool:
        return self.arguments is not None

    @property
    def failed(self) -> bool:
        if self.marked_error:
            return True
        return self.result is not None and bool(ERROR_RESULT.match(self.result[:200]))

    def result_as_json(self) -> Any:
        if self.result is None:
            return None
        try:
            return json.loads(self.result)
        except ValueError:
            return None

    def describe(self, max_length: int = 600) -> str:
        text = f"{self.name}({self.raw_arguments})"
        return text if len(text) <= max_length else text[:max_length] + "…"


@dataclass
class Turn:
    role: str                          # SYSTEM, CUSTOMER или AGENT
    number: Optional[int]              # «ход N» у реплик агента
    text: str
    tool_calls: list = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class Transcript:
    system_text: str                   # весь текст ⟦SYSTEM⟧
    policy_text: str                   # системный текст без раздела [AVAILABLE TOOLS]
    tools_text: str                    # раздел [AVAILABLE TOOLS]
    turns: list                        # реплики после системной
    domain: str
    current_time: Optional[datetime]

    def agent_calls(self) -> list:
        return [call for turn in self.turns for call in turn.tool_calls if call.caller == AGENT]

    def all_calls(self) -> list:
        return [call for turn in self.turns for call in turn.tool_calls]


def _parse_arguments(raw: str) -> Optional[dict]:
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _parse_turn_body(role: str, body: str, turn_index: int) -> Turn:
    """Текст реплики + вызовы инструментов с их ответами (ответ привязывается к вызову по имени, FIFO)."""
    text_lines: list = []
    calls: list = []
    awaiting_result: dict = {}         # имя инструмента -> вызовы без ответа, в порядке появления
    last_call, last_line_was_result = None, False
    caller = AGENT if role == SYSTEM else role
    for line in body.split("\n"):
        call_match = TOOL_CALL_LINE.match(line)
        if call_match:
            raw = call_match.group(2).strip()
            last_call = ToolCall(name=call_match.group(1), raw_arguments=raw, arguments=_parse_arguments(raw),
                                 caller=caller, turn_index=turn_index)
            calls.append(last_call)
            awaiting_result.setdefault(last_call.name, []).append(last_call)
            last_line_was_result = False
            continue
        result_match = TOOL_RESPONSE_LINE.match(line)
        if result_match:
            name = result_match.group(1)
            queue = awaiting_result.get(name) or []
            target = queue.pop(0) if queue else None
            if target is None:         # ответ без видимого вызова — заводим вызов без аргументов
                target = ToolCall(name=name, raw_arguments="", arguments={}, caller=role, turn_index=turn_index)
                calls.append(target)
            target.result = result_match.group(3)
            target.marked_error = bool(result_match.group(2))
            last_call, last_line_was_result = target, True
            continue
        if line.startswith(CONTINUATION_PREFIX) and last_call is not None:
            continuation = line[len(CONTINUATION_PREFIX):]
            if last_line_was_result:
                last_call.result = (last_call.result or "") + "\n" + continuation
            else:
                last_call.raw_arguments += "\n" + continuation
                reparsed = _parse_arguments(last_call.raw_arguments)
                if reparsed is not None:
                    last_call.arguments = reparsed
            continue
        text_lines.append(line)
    return Turn(role=role, number=None, text="\n".join(text_lines).strip(), tool_calls=calls)


def _role_of(marker: str) -> str:
    label = marker.strip("⟦⟧").upper()
    if label.startswith("SYSTEM"):
        return SYSTEM
    if label.startswith("USER"):
        return CUSTOMER
    return AGENT


def _turn_number(marker: str) -> Optional[int]:
    match = TURN_NUMBER.search(marker)
    return int(match.group(1)) if match else None


def split_marked_blocks(text: str) -> list:
    """[(маркер, тело)] по маркерам ⟦…⟧ в начале строки."""
    parts = MARKER_LINE.split(text)    # [преамбула, маркер1, тело1, маркер2, тело2, ...]
    return [(parts[k], parts[k + 1].strip("\n")) for k in range(1, len(parts), 2)]


def detect_domain(system_text: str, example_id: str = "") -> str:
    prefix = (example_id or "").split("__")[0]
    if prefix in KNOWN_DOMAINS:
        return prefix
    head = system_text[:3000].lower()
    for keyword, domain in (("airline", "airline"), ("retail", "retail"), ("telecom", "telecom"),
                            ("bank", "banking_knowledge")):
        if keyword in head:
            return domain
    return "other"


def _current_time(system_text: str, calls: list) -> Optional[datetime]:
    """Время из policy (airline, telecom), иначе — из последнего ответа get_current_time (banking)."""
    match = POLICY_TIME.search(system_text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    latest = None
    for call in calls:
        if call.name == "get_current_time" and call.result:
            match = TOOL_TIME.search(call.result)
            if match:
                latest = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S")
    return latest


def parse_context(prompt: str, example_id: str = "") -> Transcript:
    system_text = ""
    turns: list = []
    for marker, body in split_marked_blocks(prompt):
        role = _role_of(marker)
        if role == SYSTEM and not system_text:
            system_text = body
            continue
        turn = _parse_turn_body(role, body, turn_index=len(turns))
        turn.number = _turn_number(marker)
        turns.append(turn)
    tools_start = system_text.find(TOOLS_SECTION)
    policy_text = system_text[:tools_start].rstrip() if tools_start >= 0 else system_text
    tools_text = system_text[tools_start:] if tools_start >= 0 else ""
    all_calls = [call for turn in turns for call in turn.tool_calls]
    return Transcript(system_text=system_text, policy_text=policy_text, tools_text=tools_text, turns=turns,
                      domain=detect_domain(system_text, example_id),
                      current_time=_current_time(system_text, all_calls))


def parse_agent_turn(response: str) -> Turn:
    blocks = split_marked_blocks(response)
    if not blocks:
        return _parse_turn_body(AGENT, response.strip("\n"), CHECKED_TURN_INDEX)
    marker, body = blocks[0]
    turn = _parse_turn_body(AGENT, body, CHECKED_TURN_INDEX)
    turn.number = _turn_number(marker)
    for call in turn.tool_calls:
        call.caller = AGENT
    return turn
