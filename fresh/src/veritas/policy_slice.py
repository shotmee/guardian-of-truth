"""Фрагмент политики, относящийся к ходу, и короткие контексты для фокусной проверки регламента (эксперимент v6p).

Политика агента в системном сообщении размечена markdown-заголовками («## Cancel flight», «## Data Roaming» …).
Разделы ранжируются по совпадению с ключевыми словами ситуации: русские основы из последней реплики клиента и ответа
агента переводятся в английские слова политики, плюс части имён инструментов из недавних вызовов
(cancel_reservation → cancel, reservation). Берутся лучшие разделы в пределах лимита символов.
"""
from __future__ import annotations

import re

from .transcript import AGENT, CUSTOMER, Transcript

MAX_FRAGMENT_CHARS = 3000
WINDOW_CHARS = 6000
# русская основа → слова, которыми о том же говорит политика
RU_TO_POLICY = {
    "отмен": ("cancel",), "возврат": ("return", "refund"), "вернуть": ("return", "refund"), "верн": ("return", "refund"),
    "обмен": ("exchange",), "измен": ("modify", "change"), "поменя": ("modify", "change"), "перенес": ("change", "modify"),
    "замен": ("modify", "exchange"), "адрес": ("address",), "оплат": ("payment",), "платеж": ("payment",),
    "платёж": ("payment",), "карт": ("card", "payment"), "сертификат": ("certificate",), "подарочн": ("gift",),
    "багаж": ("baggage", "bag"), "чемодан": ("baggage", "bag"), "страхов": ("insurance",), "бизнес": ("business", "cabin"),
    "эконом": ("economy", "cabin"), "класс": ("cabin", "class"), "апгрейд": ("upgrade", "cabin"), "повыс": ("upgrade", "cabin"),
    "компенсац": ("compensation",), "задержк": ("delay", "compensation"), "брон": ("reservation", "book"),
    "рейс": ("flight",), "перелет": ("flight",), "перелёт": ("flight",), "пассажир": ("passenger",), "заказ": ("order",),
    "товар": ("item", "product"), "размер": ("item", "option"), "цвет": ("item", "option"), "доставк": ("delivered", "shipping"),
    "счет": ("bill", "account"), "счёт": ("bill", "account"), "долг": ("overdue", "bill"), "задолж": ("overdue", "bill"),
    "просроч": ("overdue",), "блокир": ("suspension", "suspend"), "приостанов": ("suspension", "suspend"),
    "трафик": ("data",), "интернет": ("data",), "мобильн": ("data", "mobile"), "роуминг": ("roaming",),
    "границ": ("roaming", "abroad"), "тариф": ("plan",), "план": ("plan",), "сим": ("sim",), "apn": ("apn",),
    "ммс": ("mms",), "mms": ("mms",), "смс": ("sms",), "sms": ("sms",), "wi-fi": ("wifi", "wi-fi"), "вайфай": ("wifi",),
    "скорост": ("speed",), "авиарежим": ("airplane",), "режим полет": ("airplane",), "подтвер": ("confirm", "confirmation"),
    "личност": ("authenticate", "identity", "verify"), "верифик": ("verify", "authenticate"),
    "аутентиф": ("authenticate",), "оператор": ("transfer", "human"), "специалист": ("transfer", "human"),
    "кредит": ("credit",), "комисси": ("fee",), "лимит": ("limit",), "пароль": ("password",), "аккаунт": ("account",),
    "профил": ("profile", "user"), "перевод": ("transfer",), "депозит": ("deposit",), "вклад": ("deposit", "savings"),
}
STOP_TOKENS = {"get", "details", "info", "information", "user", "id", "by", "the", "a", "and", "to", "of", "for", "list",
               "kb", "search", "bm25", "dense", "grep", "find", "calculate", "current", "time", "name", "can", "send",
               "enable", "phone", "customer", "direct", "think", "check", "status", "agent", "agents", "tool", "tools"}


def policy_sections(policy_text: str) -> list:
    """[(заголовок, текст раздела)] по markdown-заголовкам; текст до первого заголовка — раздел «intro»."""
    sections, title, buffer = [], "intro", []
    for line in policy_text.splitlines():
        if line.lstrip().startswith("#"):
            if any(part.strip() for part in buffer):
                sections.append((title, "\n".join(buffer).strip()))
            title, buffer = line.strip().lstrip("#").strip(), [line]
        else:
            buffer.append(line)
    if any(part.strip() for part in buffer):
        sections.append((title, "\n".join(buffer).strip()))
    return sections


def last_customer_message(transcript: Transcript) -> str:
    for turn in reversed(transcript.turns):
        if turn.role == CUSTOMER and (turn.text or "").strip():
            return turn.text.strip()
    return ""


def _recent_tool_tokens(transcript: Transcript, turns: int = 6) -> set:
    tokens = set()
    agent_turns = [turn for turn in transcript.turns if turn.role == AGENT]
    for turn in agent_turns[-turns:]:
        for call in turn.tool_calls:
            tokens.update(part for part in call.name.lower().split("_") if part and part not in STOP_TOKENS)
    return tokens


def query_terms(transcript: Transcript, response_text: str, response_tools: tuple = ()) -> set:
    text = (last_customer_message(transcript) + " " + (response_text or "")).lower()
    terms = {word for stem, words in RU_TO_POLICY.items() if stem in text for word in words}
    terms.update(re.findall(r"\b(cancel|refund|return|exchange|modify|baggage|insurance|upgrade|roaming|refuel|apn|mms)\b", text))
    terms.update(_recent_tool_tokens(transcript))
    for name in response_tools:
        terms.update(part for part in name.lower().split("_") if part and part not in STOP_TOKENS)
    return terms


def relevant_policy(transcript: Transcript, response_text: str, response_tools: tuple = (),
                    max_chars: int = MAX_FRAGMENT_CHARS) -> str:
    """Самые подходящие разделы политики (не больше max_chars); если совпадений нет — общие правила."""
    sections = [s for s in policy_sections(transcript.policy_text or "") if s[0].lower() != "intro" or len(s[1]) < 1500]
    if not sections:
        return ""
    terms = query_terms(transcript, response_text, response_tools)
    scored = []
    for position, (title, body) in enumerate(sections):
        lowered = (title + " " + body).lower()
        hits = sum(1 for term in terms if re.search(r"\b" + re.escape(term), lowered))
        title_hits = sum(1 for term in terms if term in title.lower())
        scored.append((hits + 2 * title_hits, -position, title, body))
    scored.sort(reverse=True)
    if scored[0][0] == 0:                              # ничего не совпало — общие правила (первые разделы)
        chosen = [(t, b) for _, _, t, b in sorted(scored, key=lambda x: -x[1])[:2]]
    else:
        chosen = [(t, b) for score, _, t, b in scored if score > 0][:3]
    out, used = [], 0
    for title, body in chosen:
        piece = body if used + len(body) <= max_chars else body[: max(0, max_chars - used)]
        if piece:
            out.append(piece)
            used += len(piece)
        if used >= max_chars:
            break
    return "\n\n".join(out)


def recent_window(prompt: str, max_chars: int = WINDOW_CHARS) -> str:
    """Хвост транскрипта (без системного сообщения), начиная с границы сообщения."""
    body = prompt.split("⟦USER⟧", 1)
    body = "⟦USER⟧" + body[1] if len(body) == 2 else prompt
    if len(body) <= max_chars:
        return body
    tail = body[-max_chars:]
    start = tail.find("\n⟦")
    return tail[start + 1:] if start >= 0 else tail


SLICE_SYSTEM = """Ты проверяешь один ответ агента службы поддержки на соответствие правилам компании.

Даны:
- `<policy_fragment>` — фрагмент политики компании, относящийся к ситуации;
- `<customer_message>` — последняя реплика клиента;
- `<response>` — проверяемый ответ агента (текст клиенту и/или вызовы инструментов).

Ошибка (1), если ответ нарушает правила фрагмента: сообщает неверные условия или правила; отказывает или предлагает
перевод на оператора там, где правила позволяют помочь; выполняет или обещает действие без шагов, которых требуют
правила (проверка личности, перечисление деталей и явное подтверждение клиента); выполняет запрещённое правилами
действие; запрашивает сведения, которые правила не требуют.
Не ошибка (0), если ответ соответствует правилам, является обычным промежуточным шагом (уточнить данные, найти запись,
запросить подтверждение) или фрагмент к ответу не относится.

Ответь одной цифрой: 1 — нарушает, 0 — не нарушает."""

WINDOW_SYSTEM = SLICE_SYSTEM.replace(
    "- `<customer_message>` — последняя реплика клиента;",
    "- `<recent_dialog>` — последние сообщения диалога, включая вызовы инструментов и их результаты (данные, которые видел "
    "агент); сверяй условия правил с этими данными;")

FOCUS_NOTE = """

## Фрагмент политики

Перед ответом агента дополнительно приведён фрагмент политики, относящийся к ситуации (`<relevant_policy>`).
Проверь ответ особенно внимательно на соответствие этому фрагменту с учётом данных из контекста."""


def slice_message(fragment: str, customer_message: str, response: str) -> str:
    return (f"<policy_fragment>\n{fragment}\n</policy_fragment>\n<customer_message>\n{customer_message}\n"
            f"</customer_message>\n<response>\n{response}\n</response>")


def window_message(fragment: str, window: str, response: str) -> str:
    return (f"<policy_fragment>\n{fragment}\n</policy_fragment>\n<recent_dialog>\n{window}\n</recent_dialog>\n"
            f"<response>\n{response}\n</response>")


def focus_message(baseline_message: str, fragment: str) -> str:
    """Сообщение P_base, в котором перед <response> вставлен фрагмент политики."""
    marker = "\n<response>\n"
    head, _, tail = baseline_message.rpartition(marker)
    return f"{head}\n<relevant_policy>\n{fragment}\n</relevant_policy>{marker}{tail}" if head else baseline_message
