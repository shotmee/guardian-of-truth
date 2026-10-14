"""Детерминированные сигналы для ходов без вызовов инструментов (агент только пишет клиенту).

В таких ходах слабый агент чаще ошибается процессно, а не в фактах:
  • рассказывает о сделанном действии, которого не было: ни одного подходящего вызова в истории;
  • описывает действие «прямо сейчас», хотя в ходе нет вызова («сбрасываю настройки…»);
  • говорит, что соединяет с оператором, не вызывая перевод;
  • обещает сам проверить или изменить то, что делается только на устройстве клиента;
  • предлагает перевод на оператора (часто преждевременный);
  • снова спрашивает то, что клиент уже сообщил.

Промпт бейзлайна такие ошибки видит плохо: он ищет выдуманные факты. Поэтому сигналы идут в ранжирование
текстовых ходов отдельными признаками. Это подсказки, а не вердикты.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .checks import CALL_DISCOVERABLE_TOOL, UNLOCK_TOOL, ConversationState, find_violations, is_read_only_call
from .transcript import AGENT, CUSTOMER, ToolCall, Transcript, Turn

SIGNAL_NAMES = ("claims_unbacked_action", "narrates_action", "transfer_without_call", "promises_device_action",
                "offers_transfer", "reasks_known_value")


@dataclass(frozen=True)
class TextSignal:
    name: str
    evidence: str                      # фрагмент текста агента, на котором сработал сигнал


# ----------------------------------------------------------------------------- заявленные действия
# корень глагола в тексте -> фрагменты имён инструментов, вызов которых подтверждает слова агента
ACTION_FAMILIES = (
    ("отмен|аннулир", ("cancel", "close", "void")),
    ("оформ|заказ|заброниров|созда|откры|выпуст|пода|зарегистр",
     ("book", "order", "create", "open", "issue", "file", "submit", "apply", "register", "dispute", "replace",
      "exchange", "return")),
    ("измен|обнов|перен[её]с|замен|исправ|настро", ("update", "modify", "change", "replace", "set", "edit", "exchange")),
    ("отправ", ("send", "request", "submit")),
    ("вернул|возврат", ("return", "refund")),
    ("сброс", ("reset",)),
    ("перев[её]л|соедин|переда", ("transfer",)),
    ("заблок|замороз|приостанов|закры", ("block", "freeze", "suspend", "lock", "close")),
    ("разблок|восстанов|возобнов|активир", ("unblock", "unfreeze", "resume", "unlock", "activate", "enable", "restore")),
    ("включ|подключ", ("enable", "activate", "add", "toggle")),
    ("выключ|отключ", ("disable", "deactivate", "remove", "toggle")),
    ("оплат|списа|начисл", ("pay", "charge", "refuel", "credit")),
    ("добав", ("add", "update")),
    ("удал", ("remove", "delete")),
)
PAST_ACTION_VERB = (r"(?:отменил|аннулировал|оформил|заказал|забронировал|создал|открыл|выпустил|подал|зарегистрировал|"
                    r"изменил|обновил|перен[её]с|заменил|исправил|настроил|отправил|вернул|сбросил|перев[её]л|соединил|"
                    r"передал|заблокировал|заморозил|приостановил|разблокировал|восстановил|возобновил|активировал|"
                    r"включил|подключил|выключил|отключил|оплатил|списал|начислил|добавил|удалил|закрыл|выполнил)\w*")
FIRST_PERSON_PAST_ACTION = re.compile(r"\bя\s+(?:\w+\s+){0,3}(" + PAST_ACTION_VERB + r")", re.IGNORECASE)
PRESENT_ACTION = re.compile(
    r"\b(сбрасываю|отправляю|соединяю|перевожу|переключаю|закрываю|блокирую|оформляю|отменяю|заказываю|"
    r"выполняю|запускаю|обновляю|применяю|подключаю|включаю|отключаю|передаю|перенаправляю)\b", re.IGNORECASE)

# ----------------------------------------------------------------------------- перевод на оператора
HUMAN = r"(?:оператор\w*|специалист\w*|сотрудник\w*|человек\w*|коллег\w*|менеджер\w*)"
TRANSFER_NOW = re.compile(r"\b(соединяю|перевожу|переключаю|передаю|перенаправляю)\s+(?:вас|ваш\w*)", re.IGNORECASE)
TRANSFER_OFFER = re.compile(
    r"\b(?:перевести|переведу|перевод\w*|соединить|соединю|передать|передам|переключить|переключу|"
    r"перенаправить|перенаправлю)\s+(?:\w+\s+){0,4}" + HUMAN, re.IGNORECASE)

# ----------------------------------------------------------------------------- действия на устройстве клиента
OWN_ACTION_PROMISE = re.compile(
    r"\b(?:я\s+(?:\w+\s+){0,2}(?:проверю|посмотрю|сброшу|перезагружу|запущу|проведу|настрою|исправлю|"
    r"попробую|могу)|сейчас\s+(?:я\s+)?(?:проверю|сброшу|запущу))\b", re.IGNORECASE)
DEVICE_OBJECT = re.compile(r"устройств|телефон|apn|sim|сим-карт|настройк|соединени|подключени|\bсет[иь]\b|"
                           r"удал[её]нн|со своей стороны", re.IGNORECASE)

# ----------------------------------------------------------------------------- повторный запрос данных
# «проверьте правильность написания» после неудачного поиска — законная просьба, поэтому «проверьте» не запрос
REQUEST_VERB = re.compile(r"\b(предоставьте|укажите|назовите|сообщите|уточните|подтвердите|напишите|пришлите|"
                          r"продиктуйте|введите|дайте|выполните|запустите)\b", re.IGNORECASE)
REQUESTED_FIELDS = {
    "email": re.compile(r"e-?mail|электронн\w+ (?:почт|адрес)\w*|адрес\w* почты", re.IGNORECASE),
    "phone": re.compile(r"номер\w* (?:вашего |мобильного )?телефон|телефонн\w+ номер|номер\w* лини|"
                        r"как\w+ (?:именно )?номер", re.IGNORECASE),
    "birth_date": re.compile(r"дат\w+ рождения", re.IGNORECASE),
    "user_id": re.compile(r"user[ _]?id|id пользователя|идентификатор пользователя|id клиента", re.IGNORECASE),
    "order_id": re.compile(r"номер\w* заказа|order[ _]?id|id заказа", re.IGNORECASE),
    "reservation_id": re.compile(r"номер\w* брон\w*|код\w* брон\w*|reservation", re.IGNORECASE),
    "zip": re.compile(r"индекс|zip", re.IGNORECASE),
}
FIELD_VALUES = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
    "birth_date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|"
                             r"\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*"
                             r"\s+\d{4}", re.IGNORECASE),
    "user_id": re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b"),
    "order_id": re.compile(r"#?\bW\d{7}\b"),
    "reservation_id": re.compile(r"\b(?!HAT\d)(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b"),  # не рейс
    "zip": re.compile(r"(?<!\d)\d{5}(?!\d)"),
}
NEED_MARKER = re.compile(r"\b(необходим\w*|нужн\w*|требуется)\b", re.IGNORECASE)
VERIFICATION_LIST_MIN_FIELDS = 3       # «подтвердите любые два из: …» — запрос сразу нескольких полей
DOMAINS_WITHOUT_VERIFICATION = ("airline", "retail")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
CUSTOMER_TOOL_MENTION = re.compile(r"`?\b([a-z]+(?:_[a-z]+)+)\b`?")


# ----------------------------------------------------------------------------- вспомогательные
def _effective_tool_name(call: ToolCall) -> str:
    if call.name == CALL_DISCOVERABLE_TOOL:
        return str((call.arguments or {}).get("agent_tool_name", call.name))
    return call.name


def _successful_agent_writes(turns: Iterable[Turn]) -> list:
    """Успешные изменяющие вызовы агента; разблокировка инструмента действием не считается."""
    return [call for turn in turns for call in turn.tool_calls
            if call.caller == AGENT and not call.failed and not is_read_only_call(call) and call.name != UNLOCK_TOOL]


def _turns_since_last_customer_message(transcript: Transcript) -> list:
    last_customer = max((i for i, turn in enumerate(transcript.turns) if turn.role == CUSTOMER and turn.text),
                        default=-1)
    return transcript.turns[last_customer + 1:]


def _tool_fragments_for(verb: str) -> tuple:
    for root, fragments in ACTION_FAMILIES:
        if re.match(root, verb, re.IGNORECASE):
            return fragments
    return ()


def _is_backed(verb: str, transcript: Transcript) -> bool:
    """Слова агента подтверждает вызов: любой изменяющий после последней реплики клиента или подходящий по смыслу."""
    if _successful_agent_writes(_turns_since_last_customer_message(transcript)):
        return True
    fragments = _tool_fragments_for(verb)
    return any(fragment in _effective_tool_name(call).lower()
               for call in _successful_agent_writes(transcript.turns) for fragment in fragments)


def _sentences(text: str) -> list:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]


def _request_units(text: str) -> list:
    """Предложения текста; строка, которая кончается двоеточием, склеивается со следующим за ней списком."""
    units: list = []
    lines = [line for line in text.split("\n") if line.strip()]
    position = 0
    while position < len(lines):
        line = lines[position].strip()
        position += 1
        if line.endswith(":"):
            while position < len(lines) and LIST_ITEM.match(lines[position]):
                line += " " + LIST_ITEM.sub("", lines[position]).strip()
                position += 1
            units.append(line)
        else:
            units.extend(_sentences(line))
    return units


def _customer_text(transcript: Transcript) -> str:
    return "\n".join(turn.text for turn in transcript.turns if turn.role == CUSTOMER and turn.text)


def _tool_results_text(transcript: Transcript) -> str:
    return "\n".join(call.result or "" for call in transcript.all_calls() if call.caller == AGENT)


def _snippet(text: str, match: re.Match, width: int = 90) -> str:
    start, end = max(0, match.start() - width // 3), min(len(text), match.end() + width)
    return text[start:end].replace("\n", " ").strip()


# ----------------------------------------------------------------------------- сигналы
def claims_unbacked_action(transcript: Transcript, turn: Turn) -> Optional[TextSignal]:
    for match in FIRST_PERSON_PAST_ACTION.finditer(turn.text):
        if not _is_backed(match.group(1), transcript):
            return TextSignal("claims_unbacked_action", _snippet(turn.text, match))
    return None


def narrates_action(transcript: Transcript, turn: Turn) -> Optional[TextSignal]:
    match = PRESENT_ACTION.search(turn.text)
    return TextSignal("narrates_action", _snippet(turn.text, match)) if match else None


def transfer_without_call(transcript: Transcript, turn: Turn) -> Optional[TextSignal]:
    """«Соединяю вас с оператором» в ходе без вызовов: перевод делается только вызовом инструмента.

    Прошлый перевод в истории не оправдывает: в данных диалог после него продолжается, и организаторы
    размечают повторное «соединяю» без вызова как ошибку.
    """
    match = TRANSFER_NOW.search(turn.text)
    return TextSignal("transfer_without_call", _snippet(turn.text, match)) if match else None


def promises_device_action(transcript: Transcript, turn: Turn) -> Optional[TextSignal]:
    """Агент берётся сам проверить или изменить устройство клиента (telecom), хотя это делает только клиент."""
    if transcript.domain != "telecom":
        return None
    for sentence in _sentences(turn.text):
        match = OWN_ACTION_PROMISE.search(sentence)
        if match and DEVICE_OBJECT.search(sentence):
            return TextSignal("promises_device_action", sentence[:160])
    return None


def offers_transfer(transcript: Transcript, turn: Turn) -> Optional[TextSignal]:
    match = TRANSFER_OFFER.search(turn.text)
    return TextSignal("offers_transfer", _snippet(turn.text, match)) if match else None


def _requested_fields(sentence: str) -> set:
    return {field for field, pattern in REQUESTED_FIELDS.items() if pattern.search(sentence)}


def _known_fields(transcript: Transcript) -> set:
    """Поля, значения которых клиент уже назвал; в доменах без верификации — ещё и найденные инструментами."""
    sources = [_customer_text(transcript)]
    if transcript.domain in DOMAINS_WITHOUT_VERIFICATION:
        sources.append(_tool_results_text(transcript))
    text = "\n".join(sources)
    return {field for field, pattern in FIELD_VALUES.items() if pattern.search(text)}


def _repeated_customer_action(transcript: Transcript, sentence: str) -> Optional[str]:
    """Агент просит выполнить на устройстве то, что клиент уже выполнил в последнем ответе, и с тех пор
    никто ничего не менял."""
    last_agent_message = max((i for i, turn in enumerate(transcript.turns) if turn.role == AGENT and turn.text),
                             default=-1)
    mentioned = set(CUSTOMER_TOOL_MENTION.findall(sentence))
    for position in range(last_agent_message + 1, len(transcript.turns)):
        for call in transcript.turns[position].tool_calls:
            if call.caller == CUSTOMER and call.name in mentioned:
                if not _successful_agent_writes(transcript.turns[position + 1:]):
                    return call.name
    return None


def reasks_known_value(transcript: Transcript, turn: Turn) -> Optional[TextSignal]:
    known = _known_fields(transcript)
    for sentence in _request_units(turn.text):
        if not (REQUEST_VERB.search(sentence) or NEED_MARKER.search(sentence) or sentence.endswith("?")):
            continue
        repeated_action = _repeated_customer_action(transcript, sentence)
        if repeated_action:
            return TextSignal("reasks_known_value", f"повторно просит {repeated_action}: {sentence[:120]}")
        requested = _requested_fields(sentence)
        if not requested:
            continue
        already_known = requested & known
        if len(requested) >= VERIFICATION_LIST_MIN_FIELDS:
            if len(already_known) >= 2:
                return TextSignal("reasks_known_value", sentence[:160])
        elif already_known:
            return TextSignal("reasks_known_value", sentence[:160])
    return None


SIGNAL_CHECKS = (claims_unbacked_action, narrates_action, transfer_without_call, promises_device_action,
                 offers_transfer, reasks_known_value)


PROCESS_SIGNALS = frozenset(SIGNAL_NAMES) - {"offers_transfer"}   # предложение перевода — не ошибка само по себе


def find_text_signals(transcript: Transcript, turn: Turn) -> list:
    """Сигналы для хода без вызовов; для хода с вызовами — пустой список."""
    if turn.has_tool_calls or not turn.text:
        return []
    signals = []
    for check in SIGNAL_CHECKS:
        signal = check(transcript, turn)
        if signal is not None:
            signals.append(signal)
    return signals


def has_process_signal(transcript: Transcript, turn: Turn) -> bool:
    return any(signal.name in PROCESS_SIGNALS for signal in find_text_signals(transcript, turn))


# ----------------------------------------------------------------------------- ошибка раньше в диалоге
def _prefix(transcript: Transcript, position: int) -> Transcript:
    return Transcript(system_text=transcript.system_text, policy_text=transcript.policy_text,
                      tools_text=transcript.tools_text, turns=transcript.turns[:position],
                      domain=transcript.domain, current_time=transcript.current_time)


def earlier_error_in_history(transcript: Transcript, tools: dict) -> bool:
    """В истории уже есть заметная ошибка агента: упавший вызов, срабатывание детерминированной проверки на
    прошлом ходе с вызовами или процессный сигнал на прошлом текстовом ходе.

    Нужен, чтобы проверить на лидерборде, считают ли организаторы ошибкой продолжение прежней ошибки.
    """
    for position, past in enumerate(transcript.turns):
        if past.role != AGENT or past.number is None:
            continue
        if any(call.failed for call in past.tool_calls):
            return True
        prefix = _prefix(transcript, position)
        if past.tool_calls:
            if find_violations(prefix, past, tools, ConversationState(prefix)):
                return True
        elif past.text and has_process_signal(prefix, past):
            return True
    return False
