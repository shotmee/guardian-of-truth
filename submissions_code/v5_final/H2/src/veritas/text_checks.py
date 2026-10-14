# Deterministic signals for a text-only turn.
#
# In this mode the weak agent typically fails on process rather than on
# facts:
#   * narrates a past action without a backing tool call;
#   * narrates an action "right now" without any tool call this turn;
#   * says it is routing to a human without issuing the transfer call;
#   * promises to do what only happens on the customer's device;
#   * offers a human hand-off (often prematurely);
#   * re-asks something the customer already stated.
#
# The baseline prompt is poor at catching these -- it hunts invented
# facts -- so we feed the signals into the ranker as separate features.
# They are hints, not verdicts.

import re as _re

from .checks import (
    CALL_DISCOVERABLE_TOOL,
    UNLOCK_TOOL,
    ConversationState,
    find_violations,
    is_read_only_call,
)
from .transcript import AGENT, CUSTOMER, Transcript


def _ic(pattern):
    return _re.compile(pattern, _re.IGNORECASE)


SIGNAL_NAMES = (
    "claims_unbacked_action", "narrates_action", "transfer_without_call",
    "promises_device_action", "offers_transfer", "reasks_known_value",
)


class TextSignal:
    __slots__ = ("name", "evidence")

    def __init__(self, name, evidence):
        self.name, self.evidence = name, evidence

    def __repr__(self):
        return "TextSignal(%r, %r)" % (self.name, self.evidence)


# --- claimed vs. backed actions ---------------------------------------------
# verb-root pattern -> tool-name fragments that back the claim.
_ACTION_FAMILIES_SOURCE = {
    "отмен|аннулир": ("cancel", "close", "void"),
    "оформ|заказ|заброниров|созда|откры|выпуст|пода|зарегистр": (
        "book", "order", "create", "open", "issue", "file", "submit",
        "apply", "register", "dispute", "replace", "exchange", "return",
    ),
    "измен|обнов|перен[её]с|замен|исправ|настро": (
        "update", "modify", "change", "replace", "set", "edit", "exchange",
    ),
    "отправ": ("send", "request", "submit"),
    "вернул|возврат": ("return", "refund"),
    "сброс": ("reset",),
    "перев[её]л|соедин|переда": ("transfer",),
    "заблок|замороз|приостанов|закры": (
        "block", "freeze", "suspend", "lock", "close",
    ),
    "разблок|восстанов|возобнов|активир": (
        "unblock", "unfreeze", "resume", "unlock", "activate", "enable",
        "restore",
    ),
    "включ|подключ": ("enable", "activate", "add", "toggle"),
    "выключ|отключ": ("disable", "deactivate", "remove", "toggle"),
    "оплат|списа|начисл": ("pay", "charge", "refuel", "credit"),
    "добав": ("add", "update"),
    "удал": ("remove", "delete"),
}
ACTION_FAMILIES = tuple(_ACTION_FAMILIES_SOURCE.items())


_PAST_VERB_FORMS = (
    r"(?:отменил|аннулировал|оформил|заказал|забронировал|создал|открыл|"
    r"выпустил|подал|зарегистрировал|изменил|обновил|перен[её]с|заменил|"
    r"исправил|настроил|отправил|вернул|сбросил|перев[её]л|соединил|"
    r"передал|заблокировал|заморозил|приостановил|разблокировал|"
    r"восстановил|возобновил|активировал|включил|подключил|выключил|"
    r"отключил|оплатил|списал|начислил|добавил|удалил|закрыл|выполнил)\w*"
)
FIRST_PERSON_PAST_ACTION = _ic(r"\bя\s+(?:\w+\s+){0,3}(" + _PAST_VERB_FORMS + r")")

PRESENT_ACTION = _ic(
    r"\b(сбрасываю|отправляю|соединяю|перевожу|переключаю|закрываю|"
    r"блокирую|оформляю|отменяю|заказываю|выполняю|запускаю|обновляю|"
    r"применяю|подключаю|включаю|отключаю|передаю|перенаправляю)\b"
)

# --- transfer-to-human expressions ------------------------------------------
_HUMAN_NOUN = r"(?:оператор\w*|специалист\w*|сотрудник\w*|человек\w*|коллег\w*|менеджер\w*)"
TRANSFER_NOW = _ic(
    r"\b(соединяю|перевожу|переключаю|передаю|перенаправляю)\s+(?:вас|ваш\w*)"
)
TRANSFER_OFFER = _ic(
    r"\b(?:перевести|переведу|перевод\w*|соединить|соединю|передать|передам|"
    r"переключить|переключу|перенаправить|перенаправлю)"
    r"\s+(?:\w+\s+){0,4}" + _HUMAN_NOUN
)

# --- device-side promises (telecom) -----------------------------------------
OWN_ACTION_PROMISE = _ic(
    r"\b(?:я\s+(?:\w+\s+){0,2}(?:проверю|посмотрю|сброшу|перезагружу|"
    r"запущу|проведу|настрою|исправлю|попробую|могу)"
    r"|сейчас\s+(?:я\s+)?(?:проверю|сброшу|запущу))\b"
)
DEVICE_OBJECT = _ic(
    r"устройств|телефон|apn|sim|сим-карт|настройк|соединени|подключени|"
    r"\bсет[иь]\b|удал[её]нн|со своей стороны"
)

# --- re-asking a known value ------------------------------------------------
# A note on "проверьте правильность написания" after a failed search: it's a
# legitimate request, so the verb "проверьте" (check) is intentionally NOT in
# the request list.
REQUEST_VERB = _ic(
    r"\b(предоставьте|укажите|назовите|сообщите|уточните|подтвердите|"
    r"напишите|пришлите|продиктуйте|введите|дайте|выполните|запустите)\b"
)

REQUESTED_FIELDS = {
    "email": _ic(r"e-?mail|электронн\w+ (?:почт|адрес)\w*|адрес\w* почты"),
    "phone": _ic(
        r"номер\w* (?:вашего |мобильного )?телефон|"
        r"телефонн\w+ номер|номер\w* лини|"
        r"как\w+ (?:именно )?номер"
    ),
    "birth_date": _ic(r"дат\w+ рождения"),
    "user_id": _ic(
        r"user[ _]?id|id пользователя|идентификатор пользователя|id клиента"
    ),
    "order_id": _ic(r"номер\w* заказа|order[ _]?id|id заказа"),
    "reservation_id": _ic(
        r"номер\w* брон\w*|код\w* брон\w*|reservation"
    ),
    "zip": _ic(r"индекс|zip"),
}

FIELD_VALUES = {
    "email":          _re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone":          _re.compile(r"(?<!\d)(?:\+?1[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
    "birth_date":     _ic(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|"
        r"\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|"
        r"август|сентябр|октябр|ноябр|декабр)\w*\s+\d{4}"
    ),
    "user_id":        _re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b"),
    "order_id":       _re.compile(r"#?\bW\d{7}\b"),
    "reservation_id": _re.compile(
        r"\b(?!HAT\d)(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b"
    ),
    "zip":            _re.compile(r"(?<!\d)\d{5}(?!\d)"),
}

NEED_MARKER = _ic(r"\b(необходим\w*|нужн\w*|требуется)\b")

VERIFICATION_LIST_MIN_FIELDS = 3
DOMAINS_WITHOUT_VERIFICATION = ("airline", "retail")

SENTENCE_BOUNDARY = _re.compile(r"(?<=[.!?])\s+|\n+")
LIST_ITEM = _re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
CUSTOMER_TOOL_MENTION = _re.compile(r"`?\b([a-z]+(?:_[a-z]+)+)\b`?")


# =========================================================== internal helpers
def _effective_tool_name(call):
    if call.name == CALL_DISCOVERABLE_TOOL:
        return str((call.arguments or {}).get("agent_tool_name", call.name))
    return call.name


def _successful_agent_writes(turns):
    bag = []
    for turn in turns:
        for item in turn.tool_calls:
            if item.caller != AGENT or item.failed:
                continue
            if is_read_only_call(item) or item.name == UNLOCK_TOOL:
                continue
            bag.append(item)
    return bag


def _turns_since_last_customer_message(transcript):
    last = -1
    for pos, turn in enumerate(transcript.turns):
        if turn.role == CUSTOMER and turn.text:
            last = pos
    return transcript.turns[last + 1:]


def _tool_fragments_for(verb):
    for root, fragments in ACTION_FAMILIES:
        if _re.match(root, verb, _re.IGNORECASE):
            return fragments
    return ()


def _is_backed(verb, transcript):
    recent = _turns_since_last_customer_message(transcript)
    if _successful_agent_writes(recent):
        return True
    fragments = _tool_fragments_for(verb)
    if not fragments:
        return False
    for call in _successful_agent_writes(transcript.turns):
        tool_lc = _effective_tool_name(call).lower()
        for piece in fragments:
            if piece in tool_lc:
                return True
    return False


def _sentences(text):
    return [p.strip() for p in SENTENCE_BOUNDARY.split(text) if p.strip()]


def _request_units(text):
    """Yield sentence-sized units. A line that ends with ":" is glued to the
    bullet list beneath it, so "please provide: a, b, c" counts as one
    request for a, b and c combined."""
    rows = [ln for ln in text.split("\n") if ln.strip()]
    out, cursor = [], 0
    while cursor < len(rows):
        line = rows[cursor].strip()
        cursor += 1
        if line.endswith(":"):
            while cursor < len(rows) and LIST_ITEM.match(rows[cursor]):
                line = line + " " + LIST_ITEM.sub("", rows[cursor]).strip()
                cursor += 1
            out.append(line)
        else:
            out.extend(_sentences(line))
    return out


def _customer_text(transcript):
    return "\n".join(t.text for t in transcript.turns if t.role == CUSTOMER and t.text)


def _tool_results_text(transcript):
    return "\n".join((c.result or "") for c in transcript.all_calls() if c.caller == AGENT)


def _snippet(text, matched, width=90):
    lo = max(0, matched.start() - width // 3)
    hi = min(len(text), matched.end() + width)
    return text[lo:hi].replace("\n", " ").strip()


# =========================================================== per-signal probes
def claims_unbacked_action(transcript, turn):
    for hit in FIRST_PERSON_PAST_ACTION.finditer(turn.text):
        if not _is_backed(hit.group(1), transcript):
            return TextSignal("claims_unbacked_action", _snippet(turn.text, hit))
    return None


def narrates_action(transcript, turn):
    hit = PRESENT_ACTION.search(turn.text)
    return TextSignal("narrates_action", _snippet(turn.text, hit)) if hit else None


def transfer_without_call(transcript, turn):
    # Even a prior transfer in history does not justify a repeat "connecting"
    # without a tool call -- the organisers mark it.
    hit = TRANSFER_NOW.search(turn.text)
    return TextSignal("transfer_without_call", _snippet(turn.text, hit)) if hit else None


def promises_device_action(transcript, turn):
    if transcript.domain != "telecom":
        return None
    for sentence in _sentences(turn.text):
        hit = OWN_ACTION_PROMISE.search(sentence)
        if hit is not None and DEVICE_OBJECT.search(sentence):
            return TextSignal("promises_device_action", sentence[:160])
    return None


def offers_transfer(transcript, turn):
    hit = TRANSFER_OFFER.search(turn.text)
    return TextSignal("offers_transfer", _snippet(turn.text, hit)) if hit else None


def _requested_fields(sentence):
    return {name for name, pattern in REQUESTED_FIELDS.items() if pattern.search(sentence)}


def _known_fields(transcript):
    sources = [_customer_text(transcript)]
    if transcript.domain in DOMAINS_WITHOUT_VERIFICATION:
        sources.append(_tool_results_text(transcript))
    pool = "\n".join(sources)
    return {name for name, pattern in FIELD_VALUES.items() if pattern.search(pool)}


def _repeated_customer_action(transcript, sentence):
    last_agent = -1
    for pos, turn in enumerate(transcript.turns):
        if turn.role == AGENT and turn.text:
            last_agent = pos
    seen = set(CUSTOMER_TOOL_MENTION.findall(sentence))
    for pos in range(last_agent + 1, len(transcript.turns)):
        for call in transcript.turns[pos].tool_calls:
            if call.caller != CUSTOMER or call.name not in seen:
                continue
            if not _successful_agent_writes(transcript.turns[pos + 1:]):
                return call.name
    return None


def reasks_known_value(transcript, turn):
    already = _known_fields(transcript)
    for sentence in _request_units(turn.text):
        asks = (REQUEST_VERB.search(sentence) or NEED_MARKER.search(sentence)
                or sentence.endswith("?"))
        if not asks:
            continue
        repeat = _repeated_customer_action(transcript, sentence)
        if repeat:
            return TextSignal("reasks_known_value",
                              "re-asks %s: %s" % (repeat, sentence[:120]))
        requested = _requested_fields(sentence)
        if not requested:
            continue
        overlap = requested & already
        if len(requested) >= VERIFICATION_LIST_MIN_FIELDS:
            if len(overlap) >= 2:
                return TextSignal("reasks_known_value", sentence[:160])
        elif overlap:
            return TextSignal("reasks_known_value", sentence[:160])
    return None


SIGNAL_CHECKS = (
    claims_unbacked_action, narrates_action, transfer_without_call,
    promises_device_action, offers_transfer, reasks_known_value,
)

# "offers_transfer" alone is not an error -- just a hint.
PROCESS_SIGNALS = frozenset(SIGNAL_NAMES) - {"offers_transfer"}


def find_text_signals(transcript, turn):
    if turn.has_tool_calls or not turn.text:
        return []
    bag = []
    for check in SIGNAL_CHECKS:
        hit = check(transcript, turn)
        if hit is not None:
            bag.append(hit)
    return bag


def has_process_signal(transcript, turn):
    return any(s.name in PROCESS_SIGNALS for s in find_text_signals(transcript, turn))


# =========================================================== prior-error check
def _prefix(transcript, position):
    return Transcript(
        system_text=transcript.system_text,
        policy_text=transcript.policy_text,
        tools_text=transcript.tools_text,
        turns=transcript.turns[:position],
        domain=transcript.domain,
        current_time=transcript.current_time,
    )


def earlier_error_in_history(transcript, tools):
    """A visible mistake sits earlier: a failed call, a rule firing on an
    earlier call turn, or a process signal on an earlier text-only turn.

    Used to probe whether the organisers still count the continuation of an
    earlier mistake as a mistake."""
    for pos, past in enumerate(transcript.turns):
        if past.role != AGENT or past.number is None:
            continue
        if any(c.failed for c in past.tool_calls):
            return True
        prefix = _prefix(transcript, pos)
        if past.tool_calls:
            if find_violations(prefix, past, tools, ConversationState(prefix)):
                return True
            continue
        if past.text and has_process_signal(prefix, past):
            return True
    return False
