"""Промпты в формате шаблона чата модели (model/chat_template.jinja) — без transformers.

Окружение jinja повторяет transformers.apply_chat_template: ImmutableSandboxedEnvironment,
trim_blocks/lstrip_blocks, фильтр tojson(ensure_ascii=False), функция raise_exception.
Совпадение с tokenizer.apply_chat_template проверяет dev-ноутбук.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .tool_schema import parse_available_tools, to_function_schemas
from .transcript import AGENT, CUSTOMER, Transcript, Turn

KEPT_LEADING_TURNS = 2                  # при обрезке длинной истории начало диалога сохраняется
OMISSION_NOTE = "\n\n[… часть истории опущена из-за длины …]"


def _raise_template_error(message: str):
    raise ValueError(message)


def _to_json(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class ChatTemplate:
    def __init__(self, template_path: Path):
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                    extensions=[loopcontrols])
        environment.filters["tojson"] = _to_json
        environment.globals["raise_exception"] = _raise_template_error
        self._template = environment.from_string(Path(template_path).read_text(encoding="utf-8"))

    def render(self, messages: list, tools: Optional[list] = None, thinking: bool = False,
               add_generation_prompt: bool = True) -> str:
        variables = dict(messages=messages, add_generation_prompt=add_generation_prompt, enable_thinking=thinking)
        if tools:
            variables["tools"] = tools
        return self._template.render(**variables)


# ----------------------------------------------------------------------------- контекст агента
def _customer_message(turn: Turn) -> str:
    """Реплика клиента; действия клиента на устройстве (telecom) — отдельным списком."""
    parts = [turn.text] if turn.text else []
    if turn.tool_calls:
        actions = [f"- {call.name}({call.raw_arguments}) → {(call.result or '').strip()}" for call in turn.tool_calls]
        parts.append("[Клиент выполнил на своём устройстве]\n" + "\n".join(actions))
    return "\n\n".join(parts) if parts else "(пустое сообщение)"


def agent_messages(transcript: Transcript, dropped_turns: int = 0) -> list:
    """История диалога в нативном формате агента: assistant.tool_calls + сообщения role=tool."""
    messages: list = [{"role": "system", "content": transcript.policy_text}]
    turns = transcript.turns
    if dropped_turns:
        turns = turns[:KEPT_LEADING_TURNS] + turns[KEPT_LEADING_TURNS + dropped_turns:]
    for turn in turns:
        if turn.role == CUSTOMER:
            messages.append({"role": "user", "content": _customer_message(turn)})
            continue
        if turn.role != AGENT:
            continue
        message: dict = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            message["tool_calls"] = [{"type": "function", "function": {
                "name": call.name,
                "arguments": call.arguments if isinstance(call.arguments, dict) else {"__raw__": call.raw_arguments},
            }} for call in turn.tool_calls]
        messages.append(message)
        for call in turn.tool_calls:
            messages.append({"role": "tool", "content": call.result if call.result is not None else "(нет ответа)"})
    if not any(message["role"] == "user" for message in messages):
        messages.insert(1, {"role": "user", "content": "(начало диалога)"})
    return messages


def agent_prompt(template: ChatTemplate, transcript: Transcript, thinking: bool, dropped_turns: int = 0) -> str:
    tools = to_function_schemas(parse_available_tools(transcript.tools_text))
    return template.render(agent_messages(transcript, dropped_turns), tools=tools, thinking=thinking)


def judge_prompt(template: ChatTemplate, system: str, user: str, thinking: bool) -> str:
    return template.render([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           thinking=thinking)


# ----------------------------------------------------------------------------- текстовые представления
def describe_turn(turn: Turn) -> str:
    """Проверяемый ход для судьи: текст клиенту и пронумерованные вызовы."""
    lines = [turn.text] if turn.text else []
    lines += [f"[CALL {number}] {call.name} {call.raw_arguments}"
              for number, call in enumerate(turn.tool_calls, 1)]
    return "\n".join(lines) if lines else "(пусто)"


def shorten_transcript(prompt: str, max_chars: int) -> str:
    """Слишком длинный транскрипт: оставляем начало (до первой реплики после приветствия) и хвост."""
    if len(prompt) <= max_chars:
        return prompt
    head_end = prompt.find("\n\n⟦", prompt.find("⟦ASSISTANT"))
    head = prompt[:head_end] if head_end > 0 else prompt[: max_chars // 3]
    tail = prompt[-(max_chars - len(head) - 80):]
    tail_start = tail.find("\n\n⟦")
    return head + OMISSION_NOTE + (tail[tail_start:] if tail_start >= 0 else tail)
