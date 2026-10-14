"""Парсер транскрипта Guardian of Truth.

Формат (как в valid.parquet):
    ⟦SYSTEM⟧ <instructions>…<policy>…</policy> [AVAILABLE TOOLS] - tool — desc / "    param: type! [enum: a|b] — desc"
    ⟦ASSISTANT⟧ / ⟦USER⟧ / ⟦ASSISTANT · ход N⟧
    \t→ TOOL_CALL name: {json}
    \t← TOOL_RESPONSE name: text        (или name [ERROR]: text), продолжение строк с отступом \t
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HEADER_RE = re.compile(r"^⟦(SYSTEM|USER|ASSISTANT)(?: · ход (\d+))?⟧[ \t]*$", re.M)
CALL_PREFIX = "→ TOOL_CALL "
RESP_PREFIX = "← TOOL_RESPONSE "
VALID_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}
TOOL_RE = re.compile(r"^- ([A-Za-z_][\w\-.]*)(?: — ?(.*))?$")
PARAM_RE = re.compile(r"^ {4}([A-Za-z_][\w\-]*): ([a-z]+)(!?)(?: \[enum: ([^\]]*)\])?(?:\s+—.*)?$")
FIELD_RE = re.compile(r"^ {8}· ([A-Za-z_][\w\-]*): ([a-z]+)(!?)(?: \[enum: ([^\]]*)\])?(?:\s+—.*)?$")
TIME_RE = re.compile(r"current time is (\d{4}-\d{2}-\d{2}[^\n]*?)\.?\s*$", re.M | re.I)
ERR_RE = re.compile(r"^\s*(error\b|tool '.*' not found)|not found|no (?:records?|results?|users?|matches) found|не найден", re.I)


@dataclass
class Param:
    name: str
    type: str
    required: bool
    enum: Optional[List[str]] = None
    fields: Dict[str, "Param"] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    desc: str
    params: Dict[str, Param] = field(default_factory=dict)


@dataclass
class Result:
    name: str
    text: str
    error: bool


@dataclass
class Call:
    name: str
    raw: str
    args: Any
    ok: bool
    result: Optional[Result] = None

    @property
    def key(self) -> str:
        if self.ok:
            return self.name + "|" + json.dumps(self.args, sort_keys=True, ensure_ascii=False)
        return self.name + "|" + self.raw.strip()


@dataclass
class Block:
    role: str
    turn: Optional[int]
    text: str
    calls: List[Call]
    results: List[Result]
    start: int
    end: int


def _parse_call(rest: str, following: List[str]) -> tuple:
    """rest = 'name: {json}'. Возвращает (Call, сколько доп. строк съедено)."""
    if ":" in rest:
        name, raw = rest.split(":", 1)
    else:
        name, raw = rest, ""
    name, raw = name.strip(), raw.strip()
    if not raw:
        return Call(name, raw, {}, True), 0
    try:
        return Call(name, raw, json.loads(raw), True), 0
    except Exception:  # noqa: BLE001 — пробуем склеить многострочный JSON
        acc = raw
        for k, ln in enumerate(following[:200]):
            s = ln.lstrip("\t ")
            if s.startswith(CALL_PREFIX) or s.startswith(RESP_PREFIX):
                break
            acc += "\n" + ln
            try:
                return Call(name, acc, json.loads(acc), True), k + 1
            except Exception:  # noqa: BLE001
                continue
        return Call(name, raw, None, False), 0


def _is_error(text: str, flag: bool) -> bool:
    return flag or bool(ERR_RE.search(text[:300]))


def parse_body(body: str):
    lines = body.split("\n")
    text_lines, calls, results = [], [], []
    cur: Optional[Result] = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.lstrip("\t ")
        if s.startswith(CALL_PREFIX):
            call, extra = _parse_call(s[len(CALL_PREFIX):], lines[i + 1:])
            calls.append(call)
            cur = None
            i += 1 + extra
            continue
        if s.startswith(RESP_PREFIX):
            rest = s[len(RESP_PREFIX):]
            m = re.match(r"([^\s:\[]+)( \[ERROR\])?: ?(.*)", rest, re.S)
            if m:
                cur = Result(m.group(1), m.group(3), bool(m.group(2)))
            else:
                cur = Result(rest.split(":", 1)[0].strip(), rest, False)
            results.append(cur)
            i += 1
            continue
        if cur is not None and ln.startswith("\t"):
            cur.text += "\n" + ln.strip()
        else:
            cur = None
            text_lines.append(ln)
        i += 1
    for r in results:
        r.error = _is_error(r.text, r.error)
    # спариваем результаты с вызовами по имени и порядку
    used = set()
    for r in results:
        for j, c in enumerate(calls):
            if j not in used and c.name == r.name:
                c.result = r
                used.add(j)
                break
    return "\n".join(text_lines).strip(), calls, results


def parse_blocks(text: str) -> List[Block]:
    heads = list(HEADER_RE.finditer(text))
    blocks: List[Block] = []
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        body = text[m.end():end]
        if m.group(1) == "SYSTEM":
            blocks.append(Block("SYSTEM", None, body.strip(), [], [], m.start(), end))
            continue
        t, calls, results = parse_body(body.strip("\n"))
        turn = int(m.group(2)) if m.group(2) else None
        blocks.append(Block(m.group(1), turn, t, calls, results, m.start(), end))
    return blocks


def parse_tools(prompt: str) -> Dict[str, Tool]:
    i = prompt.find("[AVAILABLE TOOLS]")
    if i < 0:
        return {}
    m = HEADER_RE.search(prompt, i)
    region = prompt[i: m.start() if m else len(prompt)]
    tools: Dict[str, Tool] = {}
    cur_tool: Optional[Tool] = None
    cur_param: Optional[Param] = None
    for ln in region.split("\n")[1:]:
        mt = TOOL_RE.match(ln)
        if mt:
            cur_tool = Tool(mt.group(1), (mt.group(2) or "").strip())
            tools[cur_tool.name] = cur_tool
            cur_param = None
            continue
        if cur_tool is None:
            continue
        mp = PARAM_RE.match(ln)
        if mp and mp.group(2) in VALID_TYPES:
            enum = [e.strip() for e in mp.group(4).split("|")] if mp.group(4) else None
            cur_param = Param(mp.group(1), mp.group(2), mp.group(3) == "!", enum)
            cur_tool.params[cur_param.name] = cur_param
            continue
        mf = FIELD_RE.match(ln)
        if mf and cur_param is not None and mf.group(2) in VALID_TYPES:
            enum = [e.strip() for e in mf.group(4).split("|")] if mf.group(4) else None
            cur_param.fields[mf.group(1)] = Param(mf.group(1), mf.group(2), mf.group(3) == "!", enum)
    return tools


def policy_time(prompt: str) -> Optional[str]:
    m = TIME_RE.search(prompt[:60000])
    return m.group(1).strip() if m else None


def domain_of(example_id: str, prompt: str) -> str:
    head = str(example_id).split("__", 1)[0].lower()
    if head in {"airline", "retail", "telecom", "banking_knowledge"}:
        return head
    p = prompt[:20000].lower()
    if "airline" in p:
        return "airline"
    if "knowledge base" in p or "kb_search" in p:
        return "banking_knowledge"
    if "telecom" in p or "mms" in p or "roaming" in p:
        return "telecom"
    if "retail" in p or "order" in p:
        return "retail"
    return "unknown"


def response_turn(response: str) -> Optional[int]:
    m = HEADER_RE.search(response)
    return int(m.group(2)) if m and m.group(2) else None


def split_head(prompt: str) -> int:
    """Позиция конца системной части (начало первого ⟦ASSISTANT⟧/⟦USER⟧ после SYSTEM)."""
    heads = [m for m in HEADER_RE.finditer(prompt) if m.group(1) != "SYSTEM"]
    return heads[0].start() if heads else len(prompt)
