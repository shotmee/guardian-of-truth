"""Метаданные задачи τ² для хода: поиск задачи и текстовый блок для судьи.

Разметчик организаторов видел скрытые метаданные задачи (сценарий клиента, эталонное решение). Публичные задачи
tau2-bench лежат в assets/tau2_tasks.json (dev/build_task_meta.py). Поиск задачи:
  1) по id примера: airline__10::t19, retail__106::t3, banking_knowledge__task_003::t7 — точно;
     telecom__<обрезанный id задачи> — по префиксу; если подходят несколько задач, берётся только общее для всех;
  2) если по id не нашлось — по идентификаторам из реплик клиента (user id, номера заказов и броней, email,
     телефоны); принимается только однозначное совпадение.
Не нашлось — None: судья работает без метаданных.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

ASSETS = Path(__file__).resolve().parent / "assets" / "tau2_tasks.json"
MAX_SPEC_CHARS = 5000
TOKEN = re.compile(r"#?W\d{7}|\b[a-z]+_[a-z]+_\d{3,5}\b|\b[A-Z0-9]{6}\b|[\w.+-]+@[\w-]+\.[\w.]+"
                   r"|\b\d{3}-\d{3}-\d{4}\b|\b\d{9,16}\b|\b[A-Z]{3}\b")
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _tokens(text: str) -> set:
    return {token.lstrip("#") for token in TOKEN.findall(text) if not token.isalpha() or len(token) == 3}


@lru_cache(maxsize=1)
def _base() -> dict:
    if not ASSETS.exists():
        return {"tasks": {}, "telecom": [], "tokens": {}}
    tasks = json.loads(ASSETS.read_text(encoding="utf-8"))
    telecom = [(_alnum(key.split("::", 1)[1]), key) for key in tasks if key.startswith("telecom::")]
    tokens = {key: _tokens(json.dumps(spec.get("instructions"), ensure_ascii=False))
              for key, spec in tasks.items() if not key.startswith("telecom::")}
    return {"tasks": tasks, "telecom": telecom, "tokens": tokens}


def _common(specs: list) -> dict:
    """Только то, что одинаково у всех подходящих задач (обрезанный id telecom)."""
    first = specs[0]

    def shared(field):
        values = [json.dumps(spec.get(field), sort_keys=True, ensure_ascii=False) for spec in specs]
        return first.get(field) if len(set(values)) == 1 else None

    def shared_list(field):
        lists = [[json.dumps(item, sort_keys=True, ensure_ascii=False) for item in spec.get(field) or []] for spec in specs]
        common = set(lists[0]).intersection(*map(set, lists[1:]))
        return [json.loads(item) for item in lists[0] if item in common] or None

    instructions = {}
    for key in (first.get("instructions") or {}):
        values = {json.dumps((spec.get("instructions") or {}).get(key), ensure_ascii=False) for spec in specs}
        if len(values) == 1:
            instructions[key] = first["instructions"][key]
    return {"purpose": shared("purpose"), "persona": shared("persona"), "instructions": instructions or None,
            "expected_actions": shared_list("expected_actions"), "communicate_info": shared_list("communicate_info"),
            "nl_assertions": shared_list("nl_assertions"),
            "note": f"one of {len(specs)} similar tasks; only the parts shared by all of them are shown"}


def _user_text(prompt: str) -> str:
    return " ".join(part.split("⟦", 1)[0] for part in prompt.split("⟦USER⟧")[1:])


@lru_cache(maxsize=4096)
def _by_id(head: str) -> tuple:
    base = _base()
    domain, _, rest = head.partition("__")
    rest = re.sub(r"__g\d+$", "", rest)            # наши сгенерированные диалоги τ²: <задача>__g<номер>
    if domain in ("airline", "retail", "banking_knowledge"):
        key = f"{domain}::{rest}"
        return (base["tasks"][key],) if key in base["tasks"] else ()
    if domain == "telecom" and rest:
        prefix = _alnum(rest)
        return tuple(base["tasks"][key] for norm, key in base["telecom"] if norm.startswith(prefix))
    return ()


def _by_content(domain: str, prompt: str) -> Optional[dict]:
    base = _base()
    seen = _tokens(_user_text(prompt))
    if not seen:
        return None
    scores = [(len(tokens & seen), key) for key, tokens in base["tokens"].items() if key.startswith(domain + "::") and tokens]
    best = max((score for score, _ in scores), default=0)
    winners = [key for score, key in scores if score == best]
    return base["tasks"][winners[0]] if best > 0 and len(winners) == 1 else None


def find_spec(example_id: str, prompt: str) -> tuple:
    """(метаданные задачи или None, как найдены: id / id-common / content / none)."""
    head = str(example_id).split("::")[0]
    found = _by_id(head)
    if len(found) == 1:
        return found[0], "id"
    if len(found) > 1:
        return _common(list(found)), "id-common"
    domain = next((d for d in DOMAINS if head.startswith(d)), None)
    if domain is None:                              # id без домена: домен по политике в системном сообщении
        lowered = prompt[:20000].lower()
        domain = ("airline" if "airline" in lowered else "retail" if "retail" in lowered else
                  "banking_knowledge" if "bank" in lowered else None)
    if domain and domain != "telecom":
        spec = _by_content(domain, prompt)
        if spec is not None:
            return spec, "content"
    return None, "none"


def render_spec(spec: dict) -> str:
    clean = {key: value for key, value in spec.items() if value not in (None, [], {}, "")}
    text = json.dumps(clean, ensure_ascii=False, indent=1)
    return text if len(text) <= MAX_SPEC_CHARS else text[:MAX_SPEC_CHARS] + "\n…"
