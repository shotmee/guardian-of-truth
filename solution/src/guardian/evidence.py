"""Фрагменты контекста, связанные с ответом (лексический поиск без моделей).
Судья на 10–60k токенов контекста теряет нужные строки: выносим их рядом с ответом и отдельно
перечисляем термины ответа, которых в контексте нет вовсе (кандидаты в выдуманное)."""
from __future__ import annotations

import re
from typing import List, Tuple

from .transcript import split_head

NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?(?![\w])")
IDLIKE = re.compile(r"#?[A-Za-z]+[_\-]?\d[\w\-]*|\b[A-Z0-9]{6}\b|\b[a-z]+_[a-z]+_\d+\b")
PROPER = re.compile(r"\b(?:[A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+){0,3})\b")
QUOTED = re.compile(r"[«\"']([^«»\"'\n]{3,60})[»\"']")
STOP = {"The", "This", "You", "Your", "If", "For", "And", "But", "Yes", "No", "Please", "Thank", "Hello",
        "ASSISTANT", "USER", "TOOL", "CALL", "RESPONSE", "SYSTEM", "JSON", "API", "ID", "OK"}


def _terms(response: str) -> List[Tuple[str, float]]:
    out = {}
    for m in NUM.finditer(response):
        t = m.group(0).strip("$%")
        digits = re.sub(r"\D", "", t)
        if len(digits) >= 2 and not re.fullmatch(r"20\d\d", digits):
            out[t] = max(out.get(t, 0), 1.0)
    for m in IDLIKE.finditer(response):
        t = m.group(0)
        if len(t) >= 4 and any(ch.isdigit() for ch in t):
            out[t] = 2.0
    for m in PROPER.finditer(response):
        t = m.group(0).strip()
        if t not in STOP and len(t) >= 4 and t.upper() != t:
            out[t] = max(out.get(t, 0), 1.5)
    for m in QUOTED.finditer(response):
        t = m.group(1).strip()
        if len(t) >= 4:
            out[t] = max(out.get(t, 0), 1.0)
    return sorted(out.items(), key=lambda x: -x[1])[:25]


def build(prompt: str, response: str, max_lines: int = 12, line_chars: int = 380, budget: int = 4200) -> str:
    terms = _terms(response)
    if not terms:
        return ""
    h = split_head(prompt)
    conv = prompt[h:].split("\n")
    pol = prompt[:h].split("\n")
    low_prompt = prompt.lower()
    missing = [t for t, _ in terms if t.lower() not in low_prompt]
    present = [(t, w) for t, w in terms if t.lower() in low_prompt]
    scored = []
    for src, lines, bonus in (("диалог", conv, 1.0), ("policy", pol, 0.6)):
        for k, ln in enumerate(lines):
            l = ln.lower()
            if len(l) < 3:
                continue
            hit = [(t, w) for t, w in present if t.lower() in l]
            if hit:
                scored.append((sum(w for _, w in hit) * bonus, k, src, ln, hit))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    picked, used, total = [], set(), 0
    for sc, k, src, ln, hit in scored:
        key = (src, k)
        if key in used:
            continue
        t0 = hit[0][0].lower()
        pos = ln.lower().find(t0)
        start = max(0, pos - line_chars // 3)
        frag = ln.strip()[start:start + line_chars].strip()
        piece = f"[{src}] …{frag}…"
        if total + len(piece) > budget:
            break
        picked.append(piece)
        used.add(key)
        total += len(piece)
        if len(picked) >= max_lines:
            break
    parts = []
    if picked:
        parts.append("Строки контекста, где встречаются термины ответа:\n" + "\n".join(f"- {p}" for p in picked))
    if missing:
        parts.append("Термины ответа, которых НЕТ в контексте дословно (проверь, не выдуманы ли они; числа могут быть "
                     "результатом расчёта): " + ", ".join(missing[:15]))
    return "\n".join(parts)
