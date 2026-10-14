# Rendering of chat-template prompts (model/chat_template.jinja) with no
# dependency on ``transformers``.
#
# The Jinja environment mirrors ``transformers.apply_chat_template``:
# ImmutableSandboxedEnvironment, trim_blocks + lstrip_blocks, a ``tojson``
# filter with ``ensure_ascii=False`` and a ``raise_exception`` global.
# Equivalence with ``tokenizer.apply_chat_template`` is verified in the
# development notebook.

import json as _json
from pathlib import Path as _Path

from jinja2.ext import loopcontrols as _loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment as _ImmSandboxEnv

from .tool_schema import parse_available_tools, to_function_schemas
from .transcript import AGENT, CUSTOMER

_KEEP_HEAD_TURNS = 2
_OMISSION = "\n\n[… часть истории опущена из-за длины …]"


def _raise(message):
    raise ValueError(message)


def _dumps(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return _json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                       separators=separators, sort_keys=sort_keys)


class ChatTemplate:
    def __init__(self, template_path):
        env = _ImmSandboxEnv(trim_blocks=True, lstrip_blocks=True,
                             extensions=[_loopcontrols])
        env.filters["tojson"] = _dumps
        env.globals["raise_exception"] = _raise
        self._tpl = env.from_string(_Path(template_path).read_text(encoding="utf-8"))

    def render(self, messages, tools=None, thinking=False, add_generation_prompt=True):
        ctx = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": thinking,
        }
        if tools:
            ctx["tools"] = tools
        return self._tpl.render(**ctx)


# ---------------- building the agent-visible message list ----------------
def _customer_message(turn):
    pieces = []
    if turn.text:
        pieces.append(turn.text)
    if turn.tool_calls:
        bullets = []
        for c in turn.tool_calls:
            tail = (c.result or "").strip()
            bullets.append("- %s(%s) → %s" % (c.name, c.raw_arguments, tail))
        pieces.append(
            "[Клиент выполнил на своём устройстве]\n" + "\n".join(bullets),
        )
    return "\n\n".join(pieces) if pieces else "(пустое сообщение)"


def agent_messages(transcript, dropped_turns=0):
    """Dialogue in the agent's native message format: assistant.tool_calls +
    role=tool entries for each result."""
    out = [{"role": "system", "content": transcript.policy_text}]
    turns = transcript.turns
    if dropped_turns:
        turns = turns[:_KEEP_HEAD_TURNS] + turns[_KEEP_HEAD_TURNS + dropped_turns:]
    for turn in turns:
        if turn.role == CUSTOMER:
            out.append({"role": "user", "content": _customer_message(turn)})
            continue
        if turn.role != AGENT:
            continue
        msg = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            packed = []
            for c in turn.tool_calls:
                if isinstance(c.arguments, dict):
                    arg_payload = c.arguments
                else:
                    arg_payload = {"__raw__": c.raw_arguments}
                packed.append({
                    "type": "function",
                    "function": {"name": c.name, "arguments": arg_payload},
                })
            msg["tool_calls"] = packed
        out.append(msg)
        for c in turn.tool_calls:
            out.append({
                "role": "tool",
                "content": c.result if c.result is not None else "(нет ответа)",
            })
    if not any(m["role"] == "user" for m in out):
        out.insert(1, {"role": "user", "content": "(начало диалога)"})
    return out


def agent_prompt(template, transcript, thinking, dropped_turns=0):
    specs = parse_available_tools(transcript.tools_text)
    tools = to_function_schemas(specs)
    return template.render(agent_messages(transcript, dropped_turns),
                           tools=tools, thinking=thinking)


def judge_prompt(template, system, user, thinking):
    pair = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return template.render(pair, thinking=thinking)


# ---------------- textual turn renderings ----------------
def describe_turn(turn):
    lines = []
    if turn.text:
        lines.append(turn.text)
    for n, c in enumerate(turn.tool_calls, 1):
        lines.append("[CALL %d] %s %s" % (n, c.name, c.raw_arguments))
    return "\n".join(lines) if lines else "(пусто)"


def shorten_transcript(prompt, max_chars):
    """Keep the start of the transcript (up to the first turn after the
    greeting) and the tail; drop the middle if the prompt is too long."""
    if len(prompt) <= max_chars:
        return prompt
    head_end = prompt.find("\n\n⟦", prompt.find("⟦ASSISTANT"))
    head = prompt[:head_end] if head_end > 0 else prompt[: max_chars // 3]
    budget_for_tail = max_chars - len(head) - 80
    tail = prompt[-budget_for_tail:]
    tail_start = tail.find("\n\n⟦")
    if tail_start >= 0:
        tail = tail[tail_start:]
    return head + _OMISSION + tail
