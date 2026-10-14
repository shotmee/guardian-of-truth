"""Детерминированные проверки ответа агента.

hard  — нарушения, которые разметка стабильно считает галлюцинацией (метка 1 без LLM):
        вызов недоступного инструмента, аргументы не по схеме, невалидный JSON, заглушки,
        повтор вызова, уже вернувшего ошибку, discoverable-инструмент без unlock.
facts — текстовые подсказки для LLM-судьи (достоверные факты, вывод делает судья).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .transcript import (Block, Call, Param, Tool, domain_of, parse_blocks, parse_tools,
                         policy_time, response_turn)

FREE_TEXT_KEYS = {"summary", "reason", "expression", "query", "pattern", "content", "message",
                  "note", "notes", "description", "text", "arguments", "comment", "explanation",
                  "details", "issue", "feedback", "title", "body"}
PLACEHOLDER_RE = re.compile(r"placeholder|\bx{3,}\b|<[^<>\n]{1,40}>|\{\{|\}\}|\bTBD\b|\byour[_ ]\w+|"
                            r"\bdummy\b|\bunknown_|_unknown\b", re.I)
WRITE_PREFIXES = ("book_", "cancel_", "update_", "modify_", "exchange_", "return_", "send_",
                  "suspend_", "resume_", "enable_", "disable_", "refuel_", "change_", "log_",
                  "transfer_", "unlock_", "call_discoverable", "give_discoverable", "approve_",
                  "close_", "open_", "file_", "submit_", "order_", "freeze_", "unfreeze_")
READ_ONLY = ("get_", "search_", "find_", "list_", "calculate", "KB_search", "grep", "think")
CONFIRM_REQUIRED = ("book_", "cancel_", "update_", "modify_", "exchange_", "return_", "send_certificate")
CONFIRM_RE = re.compile(r"\b(да|yes|подтверждаю|подтвердить|подтверждено|верно|согласен|согласна|хорошо|ок|окей|"
                        r"ok|okay|давайте|давай|оформляйте|оформите|делайте|конечно|продолжайте|go ahead|proceed|"
                        r"sure|confirm|всё правильно|все правильно)\b", re.I)
CLAIM_RE = re.compile(r"\b(отменил[аи]?|оформил[аи]?|забронировал[аи]?|изменил[аи]?|обновил[аи]?|отправил[аи]?|"
                      r"вернул[аи]?|выполнил[аи]?|сбросил[аи]?|перев[её]л[аи]?|активировал[аи]?|"
                      r"разблокировал[аи]?|заблокировал[аи]?|отменен[аоы]?|отменён[аоы]?|оформлен[аоы]?|"
                      r"изменен[аоы]?|изменён[аоы]?|обновлен[аоы]?|обновлён[аоы]?|отправлен[аоы]?|"
                      r"выполнен[аоы]?|сброшен[аоы]?|заблокирован[аоы]?|разблокирован[аоы]?|возвращен[аоы]?|"
                      r"возвращён[аоы]?|забронирован[аоы]?)\b", re.I)
NUM_RE = re.compile(r"(?<![\w.])\d{1,3}(?:[ , ]\d{3})+(?:[.,]\d+)?(?![\w])|(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
HUMAN_RE = re.compile(r"оператор|живым человеком|живого человека|с человеком|специалист|сотрудник|менеджер|"
                      r"human|representative|real person", re.I)


def _type_ok(v: Any, t: str) -> bool:
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool) or (isinstance(v, float) and v.is_integer())
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return True


def _check_obj(obj: Dict[str, Any], schema: Dict[str, Param], path: str,
               hard: List[str], soft: List[str], nested: bool = False) -> None:
    unknown = [k for k in obj if k not in schema]
    missing = [p.name for p in schema.values() if p.required and p.name not in obj]
    # лишние поля внутри вложенных объектов (origin/destination/price в flights) среда игнорирует,
    # разметка их ошибкой не считает -> для вложенных объектов не сообщаем вовсе
    if unknown and not nested:
        hard.append(f"{path}: поля {unknown} отсутствуют в схеме инструмента")
    if missing:
        hard.append(f"{path}: нет обязательных полей {missing}")
    for k, v in obj.items():
        p = schema.get(k)
        if p is None or v is None:
            continue
        if not _type_ok(v, p.type):
            soft.append(f"{path}.{k}: тип {type(v).__name__} не совпадает со схемой ({p.type})")
        if p.enum and isinstance(v, str) and v not in p.enum:
            hard.append(f"{path}.{k}: значение '{v}' вне допустимого enum {p.enum}")
        if p.fields:
            items = v if isinstance(v, list) else [v] if isinstance(v, dict) else []
            for n, it in enumerate(items):
                if isinstance(it, dict):
                    _check_obj(it, p.fields, f"{path}.{k}[{n}]", hard, soft, nested=True)


def _leaves(obj: Any, key: str = "") -> List[Tuple[str, Any]]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _leaves(v, k)
    elif isinstance(obj, list):
        for v in obj:
            out += _leaves(v, key)
    else:
        out.append((key, obj))
    return out


def _norm_num(s: str):
    s = s.replace(" ", " ")
    if re.fullmatch(r"\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?", s):
        s = re.sub(r"[ ,](?=\d{3})", "", s)
    s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return round(v, 2)


def _numbers(text: str) -> set:
    out = set()
    for m in NUM_RE.finditer(text):
        v = _norm_num(m.group(0))
        if v is not None:
            out.add(v)
    return out


def _short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def analyze(example_id: str, prompt: str, response: str) -> Dict[str, Any]:
    tools: Dict[str, Tool] = parse_tools(prompt)
    hist: List[Block] = parse_blocks(prompt)
    resp_blocks: List[Block] = parse_blocks(response)
    calls: List[Call] = [c for b in resp_blocks for c in b.calls]
    resp_text = "\n".join(b.text for b in resp_blocks if b.text).strip()
    domain = domain_of(example_id, prompt)
    turn = response_turn(response)
    low_prompt = prompt.lower()

    hard: List[str] = []
    soft: List[str] = []
    facts: List[str] = []

    hist_calls = [(b.turn, c) for b in hist for c in b.calls]
    unlocked = {str((c.args or {}).get("agent_tool_name")) for _, c in hist_calls
                if c.name == "unlock_discoverable_agent_tool" and c.ok and not (c.result and c.result.error)}
    users_hist = [b for b in hist if b.role == "USER"]
    flags: Dict[str, Any] = {"ungrounded_values": 0, "unconfirmed_write": 0, "claimed_action_no_call": 0,
                             "missing_numbers": 0, "transfer": 0, "repeat_call": 0}

    for ci, c in enumerate(calls):
        tag = f"`{c.name}`"
        if not c.ok:
            hard.append(f"{tag}: аргументы не являются валидным JSON")
            continue
        if tools and c.name not in tools:
            prev_err = any(hc.name == c.name and hc.result is not None and hc.result.error for _, hc in hist_calls)
            hard.append(f"инструмент {tag} отсутствует в [AVAILABLE TOOLS] агента"
                        + (" (ранее его вызов уже вернул ошибку)" if prev_err else ""))
        elif tools and isinstance(c.args, dict):
            _check_obj(c.args, tools[c.name].params, tag, hard, soft)
        # повтор вызова, который уже вернул ошибку (если между ними не было изменяющих действий)
        for idx in range(len(hist_calls) - 1, -1, -1):
            t, hc = hist_calls[idx]
            if hc.key == c.key and hc.result is not None and hc.result.error:
                between = [x for _, x in hist_calls[idx + 1:]] + calls[:ci]
                changed = any(x.name.startswith(WRITE_PREFIXES) and x.name != "transfer_to_human_agents"
                              and not (x.result is not None and x.result.error)
                              for x in between)
                msg = (f"{tag} с теми же аргументами уже вызывался (ход {t}) и вернул ошибку: "
                       f"«{_short(hc.result.text, 120)}»")
                if changed:
                    soft.append(msg + "; после этого были изменяющие действия — проверь, изменилась ли ситуация")
                else:
                    hard.append(msg)
                break
        else:
            for t, hc in hist_calls:
                if hc.key == c.key and not c.name.startswith(("get_", "search_", "find_", "list_", "check_", "KB_", "grep")):
                    soft.append(f"{tag} с теми же аргументами уже выполнялся в ходе {t}")
                    break
        # discoverable-инструменты банка
        if c.name == "call_discoverable_agent_tool" and isinstance(c.args, dict):
            nm = str(c.args.get("agent_tool_name"))
            unlocked_now = any(x.name == "unlock_discoverable_agent_tool" and isinstance(x.args, dict)
                               and str(x.args.get("agent_tool_name")) == nm for x in calls)
            if nm not in unlocked and not unlocked_now:
                hard.append(f"{tag}: инструмент '{nm}' не был разблокирован через unlock_discoverable_agent_tool")
        # аргументы: заглушки и значения, которых нет в контексте
        args = c.args
        if c.name == "call_discoverable_agent_tool" and isinstance(args, dict) and isinstance(args.get("arguments"), str):
            try:
                import json
                args = dict(args)
                args["_inner"] = json.loads(args["arguments"])
            except Exception:  # noqa: BLE001
                pass
        ungrounded = []
        for k, v in _leaves(args):
            if k in FREE_TEXT_KEYS:
                continue
            if isinstance(v, str) and PLACEHOLDER_RE.search(v):
                hard.append(f"{tag}: значение аргумента {k}='{_short(v, 60)}' похоже на заглушку")
                continue
            if isinstance(v, str) and len(v) >= 3 and v.lower() not in low_prompt:
                ungrounded.append(f"{k}='{_short(v, 60)}'")
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and ("id" in k.lower()) and str(v) not in prompt:
                ungrounded.append(f"{k}={v}")
        if ungrounded:
            soft.append(f"{tag}: значения, которых нет в контексте дословно: " + ", ".join(ungrounded[:6]))
            flags["ungrounded_values"] += len(ungrounded)
        # замена товара на него же (retail: modify/exchange с new_item_id == item_id)
        if isinstance(c.args, dict) and isinstance(c.args.get("item_ids"), list) \
                and isinstance(c.args.get("new_item_ids"), list):
            same = [a for a, b in zip(c.args["item_ids"], c.args["new_item_ids"]) if str(a) == str(b)]
            if same:
                hard.append(f"{tag}: new_item_id совпадает со старым item_id {same[:3]} — это не замена")
        # изменяющее действие без явного «да» клиента (airline/retail policy требует подтверждения)
        if domain in ("airline", "retail") and c.name.startswith(CONFIRM_REQUIRED) and users_hist:
            last_user = users_hist[-1].text or ""
            if not CONFIRM_RE.search(last_user):
                soft.append(f"{tag}: изменяющее действие, а в последнем сообщении клиента нет явного подтверждения")
                flags["unconfirmed_write"] = 1

    # числа в тексте ответа, которых нет в контексте
    if resp_text:
        pn = _numbers(prompt)
        missing = sorted(v for v in _numbers(resp_text) if v >= 10 and v not in pn)
        if missing:
            soft.append("числа из текста ответа, которых нет в контексте дословно (могут быть корректным "
                        "результатом вычисления — проверь арифметику): "
                        + ", ".join(str(int(v)) if float(v).is_integer() else str(v) for v in missing[:10]))
            flags["missing_numbers"] = len(missing)
        # «я отменил / оформлено / отправлено», но изменяющих вызовов после последнего сообщения клиента нет
        if CLAIM_RE.search(resp_text):
            last_user_end = users_hist[-1].end if users_hist else 0
            done_after = [c for b in hist if b.start >= last_user_end for c in b.calls
                          if c.name.startswith(WRITE_PREFIXES) and not (c.result is not None and c.result.error)]
            done_now = [c for c in calls if c.name.startswith(WRITE_PREFIXES)]
            if not done_after and not done_now:
                soft.append("в тексте заявлено выполненное действие, но изменяющих вызовов после последнего "
                            "сообщения клиента нет — проверь, не выдумано ли действие")
                flags["claimed_action_no_call"] = 1

    # контекстные факты для судьи
    pt = policy_time(prompt)
    if pt:
        facts.append(f"Текущее время по policy: {pt}.")
    users = [b for b in hist if b.role == "USER"]
    if users:
        facts.append(f"Последнее сообщение клиента перед ходом: «{_short(users[-1].text, 700)}»")
    if calls:
        names = [c.name for c in calls]
        facts.append("Вызовы в проверяемом ходе: " + ", ".join(names) + ".")
        writes = [n for n in names if n.startswith(WRITE_PREFIXES)]
        if writes:
            facts.append("Изменяющие/значимые действия в ходе: " + ", ".join(writes)
                         + ". Проверь предусловия policy (подтверждение клиента, правила, верификация).")
        if any(n == "transfer_to_human_agents" for n in names):
            asks = sum(1 for b in users if HUMAN_RE.search(b.text or ""))
            facts.append(f"Клиент упоминал оператора/человека в {asks} сообщениях из {len(users)}.")
    else:
        facts.append("В ходе нет вызовов инструментов (только текст клиенту).")
    if tools:
        facts.append("Инструменты агента в [AVAILABLE TOOLS]: " + ", ".join(tools) + ".")
    if calls and not hard:
        facts.append("Автопроверка: все вызванные инструменты доступны агенту, аргументы соответствуют схеме.")
    hard = list(dict.fromkeys(hard))
    soft = list(dict.fromkeys(soft))
    for h in hard[:8]:
        facts.append("НАРУШЕНИЕ: " + h)
    for s in soft[:8]:
        facts.append("Внимание: " + s)

    flags["transfer"] = int(any(c.name == "transfer_to_human_agents" for c in calls))
    flags["repeat_call"] = int(any("уже выполнялся" in s or "уже вызывался" in s for s in soft + hard))
    flags["ro_only"] = int(bool(calls) and not resp_text and all(c.name.startswith(READ_ONLY) for c in calls))
    flags.update({"n_hard": len(hard), "n_soft": len(soft), "n_calls": len(calls), "has_text": int(bool(resp_text)),
                  "n_writes": sum(1 for c in calls if c.name.startswith(WRITE_PREFIXES)),
                  "resp_chars": len(response), "prompt_chars": len(prompt), "turn": turn or 0})
    return {
        "domain": domain,
        "turn": turn,
        "n_calls": len(calls),
        "has_text": bool(resp_text),
        "hard": hard,
        "soft": soft,
        "facts": facts,
        "n_tools": len(tools),
        "flags": flags,
    }
