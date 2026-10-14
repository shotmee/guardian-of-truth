"""Детерминированные проверки хода агента (без LLM).

Каждое правило смотрит только на видимый контекст: схему [AVAILABLE TOOLS], историю вызовов и их
результаты, текущее время из policy. Если данных для проверки нет, правило молчит. Срабатывание
любого правила означает ошибку в ходе.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from .tool_schema import ToolParameter, ToolSpec, parse_available_tools
from .transcript import ToolCall, Transcript, Turn

READ_ONLY_PREFIXES = ("get_", "find_", "search_", "list_", "check_", "kb_search")
READ_ONLY_TOOLS = {"grep", "get_current_time", "KB_search_bm25", "KB_search_dense", "KB_search"}
FREE_TEXT_ARGUMENTS = {"summary", "query", "pattern", "reason", "note", "message", "description",
                       "content", "comment", "expression", "arguments"}
PLACEHOLDER = re.compile(
    r"placeholder|^<[^<>]{1,40}>$|^\{[^{}]{1,40}\}$|\bx{3,}\b|^(tbd|todo|unknown|n/?a|none|null|string|example|"
    r"user_id|order_id|item_id|account_id|card_id)$|^your[_ ]|_here$|^\.\.\.$|\[insert",
    re.IGNORECASE,
)
ENTITY_ID_KEYS = ("reservation_id", "order_id", "line_id", "bill_id", "customer_id", "device_id", "plan_id", "id")

UNLOCK_TOOL = "unlock_discoverable_agent_tool"
CALL_DISCOVERABLE_TOOL = "call_discoverable_agent_tool"

# airline policy
FREE_BAGS_PER_PASSENGER = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}
MAX_PAYMENTS_BY_TYPE = {"certificate": 1, "credit_card": 1, "gift_card": 3}
MAX_PASSENGERS = 5
FREE_CANCELLATION_WINDOW = timedelta(hours=24)

# retail policy
ALLOWED_CANCEL_REASONS = ("no longer needed", "ordered by mistake")
PENDING_ORDER_ACTIONS = ("cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
                         "modify_pending_order_payment")
DELIVERED_ORDER_ACTIONS = ("return_delivered_order_items", "exchange_delivered_order_items")
ONCE_PER_ORDER_ACTIONS = ("return_delivered_order_items", "exchange_delivered_order_items",
                          "modify_pending_order_items")
ITEM_SWAP_ACTIONS = ("modify_pending_order_items", "exchange_delivered_order_items")

# telecom policy
MAX_REFUEL_GB = 2


@dataclass(frozen=True)
class Violation:
    rule: str
    call_index: int                    # номер вызова в проверяемом ходе
    message: str


# ----------------------------------------------------------------------------- типы вызовов
def is_read_only_call(call: ToolCall) -> bool:
    if call.name == CALL_DISCOVERABLE_TOOL:
        target = str((call.arguments or {}).get("agent_tool_name", ""))
        return target.lower().startswith(READ_ONLY_PREFIXES)
    return call.name in READ_ONLY_TOOLS or call.name.lower().startswith(READ_ONLY_PREFIXES)


def has_side_effects(turn: Turn) -> bool:
    """Хотя бы один вызов что-то меняет или решает: запись, перевод, разблокировка, расчёт."""
    return any(not is_read_only_call(call) for call in turn.tool_calls)


# ----------------------------------------------------------------------------- вспомогательные
def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in sorted(value.items())}
    return value


def call_signature(call: ToolCall) -> str:
    """Имя + нормализованные аргументы: одинаковая подпись = тот же самый вызов."""
    normalized = normalize_value(call.arguments or {})
    return call.name + "|" + json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def iter_strings(value: Any, key: str = "") -> Iterator:
    """Все строковые значения вместе с именем ближайшего ключа: (key, text)."""
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item, key)
    elif isinstance(value, dict):
        for child_key, item in value.items():
            yield from iter_strings(item, child_key)


def parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    has_time = "T" in text or ":" in text
    for pattern in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:19] if has_time else text[:10], pattern)
        except ValueError:
            continue
    return None


def _flight_keys(flights: Any) -> set:
    return {(str(f.get("flight_number")), str(f.get("date"))) for f in flights or [] if isinstance(f, dict)}


class ConversationState:
    """Что агент уже знает из истории: последние версии сущностей, профили, статусы рейсов, разблокировки."""

    def __init__(self, transcript: Transcript):
        self.entities_by_id: dict = {}           # id -> последняя версия записи (бронь, заказ, линия, счёт …)
        self.user_profiles: dict = {}            # user_id -> профиль из get_user_details
        self.flight_statuses: dict = {}          # (номер рейса, дата) -> статус
        self.unlocked_tools: set = set()
        self.agent_calls: list = []
        for call in transcript.agent_calls():
            self.agent_calls.append(call)
            self._remember(call)

    def entity(self, entity_id: str) -> Optional[dict]:
        return self.entities_by_id.get(entity_id) if entity_id else None

    def _remember(self, call: ToolCall):
        if call.failed:
            return
        arguments = call.arguments or {}
        if call.name == UNLOCK_TOOL and arguments.get("agent_tool_name"):
            self.unlocked_tools.add(str(arguments["agent_tool_name"]))
            return
        if call.name == "get_flight_status":
            status = (call.result or "").strip().strip('"').lower()
            key = (str(arguments.get("flight_number", "")), str(arguments.get("date", "")))
            self.flight_statuses[key] = status
            return
        result = call.result_as_json()
        if call.name == "get_user_details" and isinstance(result, dict):
            self.user_profiles[str(result.get("user_id") or arguments.get("user_id"))] = result
        records = result if isinstance(result, list) else [result]
        for record in records:
            if isinstance(record, dict):
                self._index_entity(record)

    def _index_entity(self, record: dict):
        for key in ENTITY_ID_KEYS:
            entity_id = record.get(key)
            if isinstance(entity_id, str) and entity_id:
                self.entities_by_id[entity_id] = record
                return


# ----------------------------------------------------------------------------- общие правила
def _schema_violations(index: int, call: ToolCall, spec: ToolSpec) -> list:
    violations = []
    arguments = call.arguments or {}
    for parameter in spec.parameters.values():
        if parameter.required and parameter.name not in arguments:
            violations.append(Violation("missing_param", index, f"{call.name}: нет обязательного '{parameter.name}'"))
    for name, value in arguments.items():
        parameter = spec.parameters.get(name)
        if parameter is None:
            violations.append(Violation("unknown_param", index, f"{call.name}: лишний параметр '{name}'"))
            continue
        violations += _value_violations(index, call.name, parameter, value)
    return violations


def _value_violations(index: int, tool_name: str, parameter: ToolParameter, value: Any) -> list:
    if value is None:
        if parameter.required:
            return [Violation("null_param", index, f"{tool_name}: '{parameter.name}' = null")]
        return []
    if parameter.type == "array" and not isinstance(value, list):
        return [Violation("type", index, f"{tool_name}: '{parameter.name}' должен быть массивом")]
    if parameter.type == "object" and not isinstance(value, dict):
        return [Violation("type", index, f"{tool_name}: '{parameter.name}' должен быть объектом")]
    violations = []
    if parameter.allowed_values and isinstance(value, str) and value not in parameter.allowed_values:
        violations.append(Violation("enum", index,
                                    f"{tool_name}: '{parameter.name}'={value!r} вне {parameter.allowed_values}"))
    if parameter.fields:
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, dict):
                violations.append(Violation("type", index, f"{tool_name}: элемент '{parameter.name}' не объект"))
                continue
            for nested in parameter.fields.values():
                if nested.required and nested.name not in item:
                    violations.append(Violation("missing_nested", index,
                                                f"{tool_name}: в '{parameter.name}' нет '{nested.name}'"))
                elif (nested.name in item and nested.allowed_values and isinstance(item[nested.name], str)
                      and item[nested.name] not in nested.allowed_values):
                    violations.append(Violation("enum", index,
                                                f"{tool_name}: '{parameter.name}.{nested.name}' вне enum"))
    return violations


def _placeholder_violations(index: int, call: ToolCall) -> list:
    for key, text in iter_strings(call.arguments or {}):
        if key in FREE_TEXT_ARGUMENTS and "placeholder" not in text.lower():
            continue
        if PLACEHOLDER.search(text.strip()):
            return [Violation("placeholder", index, f"{call.name}: заглушка {key}={text[:60]!r}")]
    return []


def _discoverable_call_violations(index: int, call: ToolCall, unlocked_tools: set) -> list:
    violations = []
    arguments = call.arguments or {}
    target = str(arguments.get("agent_tool_name", ""))
    if target and target not in unlocked_tools:
        violations.append(Violation("not_unlocked", index, f"{target} не разблокирован"))
    raw_inner = arguments.get("arguments")
    if isinstance(raw_inner, str) and raw_inner.strip():
        try:
            inner = json.loads(raw_inner)
        except ValueError:
            return violations + [Violation("bad_json", index, "arguments — невалидный JSON")]
        for _, text in iter_strings(inner):
            if "placeholder" in text.lower() or PLACEHOLDER.search(text.strip()):
                violations.append(Violation("placeholder", index, f"заглушка в arguments: {text[:60]!r}"))
                break
    return violations


def _repeated_failure_violations(index: int, call: ToolCall, state: ConversationState,
                                 history_by_signature: dict) -> list:
    """Тот же вызов уже вернул ошибку, и с тех пор не было успешных изменяющих вызовов."""
    previous = history_by_signature.get(call_signature(call)) or []
    if not previous or not previous[-1].failed:
        return []
    position = state.agent_calls.index(previous[-1])
    changed_since = any(not later.failed and not is_read_only_call(later)
                        for later in state.agent_calls[position + 1:])
    if changed_since:
        return []
    return [Violation("repeat_failed", index, f"{call.name}: повтор вызова, который уже вернул ошибку")]


# ----------------------------------------------------------------------------- airline
# Все доменные правила имеют одну сигнатуру: (index, call, state, now, turn_calls) -> [Violation].
def _airline_cancellation(index, call, state, now, turn_calls) -> list:
    if call.name != "cancel_reservation" or now is None:
        return []
    reservation_id = str((call.arguments or {}).get("reservation_id", ""))
    reservation = state.entity(reservation_id)
    if not isinstance(reservation, dict):
        return []
    flights = [f for f in reservation.get("flights") or [] if isinstance(f, dict)]
    start_of_today = now.replace(hour=0, minute=0, second=0)
    already_flown = any((parse_datetime(f.get("date")) or now) < start_of_today for f in flights)
    if already_flown:
        return [Violation("cancel_flown", index, f"отмена {reservation_id}: часть перелёта уже в прошлом")]
    created = parse_datetime(reservation.get("created_at"))
    if created is None:
        return []
    booked_recently = now - created <= FREE_CANCELLATION_WINDOW
    is_business = str(reservation.get("cabin", "")).lower() == "business"
    is_insured = str(reservation.get("insurance", "")).lower() == "yes"
    cancelled_by_airline = any(
        "cancel" in state.flight_statuses.get((str(f.get("flight_number")), str(f.get("date"))), "")
        for f in flights)
    if booked_recently or is_business or is_insured or cancelled_by_airline:
        return []
    return [Violation("cancel_rules", index, f"отмена {reservation_id}: ни одно условие отмены не выполнено")]


def _airline_basic_economy_change(index, call, state, now, turn_calls) -> list:
    if call.name != "update_reservation_flights":
        return []
    arguments = call.arguments or {}
    reservation_id = str(arguments.get("reservation_id", ""))
    reservation = state.entity(reservation_id)
    if not isinstance(reservation, dict) or str(reservation.get("cabin", "")).lower() != "basic_economy":
        return []
    old_flights, new_flights = _flight_keys(reservation.get("flights")), _flight_keys(arguments.get("flights"))
    if old_flights and new_flights and old_flights != new_flights:
        return [Violation("basic_economy_modify", index, f"{reservation_id}: basic economy нельзя менять рейсы")]
    return []


def _airline_baggage(index, call, state, now, turn_calls) -> list:
    if call.name not in ("update_reservation_baggages", "book_reservation"):
        return []
    arguments = call.arguments or {}
    if call.name == "book_reservation":
        user_id, cabin = str(arguments.get("user_id", "")), str(arguments.get("cabin", "")).lower()
        passengers = len(arguments.get("passengers") or [])
        reservation = None
    else:
        reservation = state.entity(str(arguments.get("reservation_id", "")))
        if not isinstance(reservation, dict):
            reservation = None
        user_id = str(reservation.get("user_id", "")) if reservation else ""
        cabin = str(reservation.get("cabin", "")).lower() if reservation else ""
        passengers = len(reservation.get("passengers") or []) if reservation else 0
    violations = []
    profile = state.user_profiles.get(user_id)
    membership = str((profile or {}).get("membership", "")).lower()
    total, nonfree = arguments.get("total_baggages"), arguments.get("nonfree_baggages")
    allowance = FREE_BAGS_PER_PASSENGER.get(membership, {}).get(cabin)
    if profile and allowance is not None and passengers and isinstance(total, int) and isinstance(nonfree, int):
        expected_nonfree = max(0, total - allowance * passengers)
        if nonfree != expected_nonfree:
            violations.append(Violation("baggage_calc", index, f"nonfree={nonfree}, по политике {expected_nonfree}"))
    if reservation is not None and isinstance(total, int):
        current_total = reservation.get("total_baggages")
        if isinstance(current_total, int) and total < current_total:
            violations.append(Violation("baggage_remove", index, "убирать багаж нельзя"))
    return violations


def _airline_payment_sources(index, call, state, now, turn_calls) -> list:
    """Оплата только способами из профиля пользователя."""
    arguments = call.arguments or {}
    payment_ids = [arguments["payment_id"]] if isinstance(arguments.get("payment_id"), str) else []
    payment_ids += [pm["payment_id"] for pm in arguments.get("payment_methods") or []
                    if isinstance(pm, dict) and isinstance(pm.get("payment_id"), str)]
    reservation = state.entity(str(arguments.get("reservation_id", ""))) or {}
    profile = state.user_profiles.get(str(arguments.get("user_id") or reservation.get("user_id") or ""))
    if not profile or not isinstance(profile.get("payment_methods"), dict):
        return []
    return [Violation("payment_not_in_profile", index, f"{payment_id} нет в профиле")
            for payment_id in payment_ids if payment_id not in profile["payment_methods"]]


def _airline_booking_limits(index, call, state, now, turn_calls) -> list:
    if call.name != "book_reservation":
        return []
    arguments = call.arguments or {}
    payment_ids = [str(pm.get("payment_id", "")) for pm in arguments.get("payment_methods") or []
                   if isinstance(pm, dict)]
    violations = []
    if any(sum(p.startswith(kind) for p in payment_ids) > limit for kind, limit in MAX_PAYMENTS_BY_TYPE.items()):
        violations.append(Violation("payment_limits", index, "превышены лимиты способов оплаты"))
    if len(arguments.get("passengers") or []) > MAX_PASSENGERS:
        violations.append(Violation("too_many_pax", index, f"больше {MAX_PASSENGERS} пассажиров"))
    return violations


# ----------------------------------------------------------------------------- retail
def _order_of(call: ToolCall, state: ConversationState) -> tuple:
    order_id = str((call.arguments or {}).get("order_id", ""))
    order = state.entity(order_id)
    return order_id, (order if isinstance(order, dict) else None)


def _retail_item_pairs(index, call, state, now, turn_calls) -> list:
    if call.name not in ITEM_SWAP_ACTIONS:
        return []
    arguments = call.arguments or {}
    old_items, new_items = arguments.get("item_ids") or [], arguments.get("new_item_ids") or []
    if not isinstance(old_items, list) or not isinstance(new_items, list):
        return []
    if len(old_items) != len(new_items):
        return [Violation("items_len", index, "длины item_ids и new_item_ids различаются")]
    if any(str(old) == str(new) for old, new in zip(old_items, new_items)):
        return [Violation("same_item", index, "new_item_id совпадает со старым")]
    return []


def _retail_order_status(index, call, state, now, turn_calls) -> list:
    order_id, order = _order_of(call, state)
    status = str(order.get("status", "")).lower() if order else ""
    if not status:
        return []
    if call.name in PENDING_ORDER_ACTIONS and status != "pending":
        return [Violation("order_status", index, f"{order_id}: статус '{status}', а нужен pending")]
    if call.name in DELIVERED_ORDER_ACTIONS and status != "delivered":
        return [Violation("order_status", index, f"{order_id}: статус '{status}', а нужен delivered")]
    return []


def _retail_cancel_reason(index, call, state, now, turn_calls) -> list:
    if call.name != "cancel_pending_order":
        return []
    reason = str((call.arguments or {}).get("reason", "")).strip().lower()
    if reason and reason not in ALLOWED_CANCEL_REASONS:
        return [Violation("cancel_reason", index, f"причина '{reason}' не из политики")]
    return []


def _retail_single_change(index, call, state, now, turn_calls) -> list:
    """return / exchange / modify items — один раз на заказ (учитываем историю и этот же ход)."""
    order_id = str((call.arguments or {}).get("order_id", ""))
    if call.name not in ONCE_PER_ORDER_ACTIONS or not order_id:
        return []

    def changes_this_order(other: ToolCall) -> bool:
        return other.name in ONCE_PER_ORDER_ACTIONS and str((other.arguments or {}).get("order_id", "")) == order_id

    earlier_in_history = any(changes_this_order(h) and not h.failed for h in state.agent_calls)
    earlier_in_turn = any(changes_this_order(other) for other in turn_calls[:index])
    if earlier_in_history or earlier_in_turn:
        return [Violation("once_per_order", index, f"{order_id}: второе изменение заказа")]
    return []


def _retail_payment(index, call, state, now, turn_calls) -> list:
    payment_method = (call.arguments or {}).get("payment_method_id")
    if not isinstance(payment_method, str):
        return []
    _, order = _order_of(call, state)
    user_id = str(order.get("user_id", "")) if order else ""
    if user_id:
        profile = state.user_profiles.get(user_id)
    else:                               # заказ не открывали — годится единственный известный профиль
        profile = next(iter(state.user_profiles.values())) if len(state.user_profiles) == 1 else None
    violations = []
    profile_methods = (profile or {}).get("payment_methods")
    if isinstance(profile_methods, dict) and profile_methods and payment_method not in profile_methods:
        violations.append(Violation("payment_not_in_profile", index, f"{payment_method} нет в профиле"))
    if call.name == "return_delivered_order_items" and order:
        original = {str(p.get("payment_method_id")) for p in order.get("payment_history") or [] if isinstance(p, dict)}
        if original and payment_method not in original and not payment_method.startswith("gift_card"):
            violations.append(Violation("refund_method", index,
                                        f"возврат на {payment_method}: не исходный и не подарочная карта"))
    return violations


# ----------------------------------------------------------------------------- telecom
def _telecom_resume_line(index, call, state, now, turn_calls) -> list:
    if call.name != "resume_line" or now is None:
        return []
    line = state.entity(str((call.arguments or {}).get("line_id", "")))
    contract_end = parse_datetime(line.get("contract_end_date")) if isinstance(line, dict) else None
    if contract_end is not None and contract_end < now:
        return [Violation("resume_expired", index, "контракт линии истёк — снимать приостановку нельзя")]
    return []


def _telecom_refuel_limit(index, call, state, now, turn_calls) -> list:
    if call.name != "refuel_data":
        return []
    try:
        amount = float((call.arguments or {}).get("gb_amount", 0))
    except (TypeError, ValueError):
        return []
    if amount > MAX_REFUEL_GB:
        return [Violation("refuel_max", index, f"больше {MAX_REFUEL_GB} GB за раз")]
    return []


def _telecom_payment_request(index, call, state, now, turn_calls) -> list:
    if call.name != "send_payment_request":
        return []
    bill = state.entity(str((call.arguments or {}).get("bill_id", "")))
    if isinstance(bill, dict) and bill.get("status") and "overdue" not in str(bill["status"]).lower():
        return [Violation("bill_not_overdue", index, f"счёт в статусе {bill['status']}")]
    return []


DOMAIN_RULES = {
    "airline": (_airline_cancellation, _airline_basic_economy_change, _airline_baggage,
                _airline_payment_sources, _airline_booking_limits),
    "retail": (_retail_item_pairs, _retail_order_status, _retail_cancel_reason, _retail_single_change,
               _retail_payment),
    "telecom": (_telecom_resume_line, _telecom_refuel_limit, _telecom_payment_request),
}


# ----------------------------------------------------------------------------- точка входа
def find_violations(transcript: Transcript, turn: Turn, tools: Optional[dict] = None,
                    state: Optional[ConversationState] = None) -> list:
    tools = tools if tools is not None else parse_available_tools(transcript.tools_text)
    state = state if state is not None else ConversationState(transcript)
    history_by_signature: dict = {}
    for call in state.agent_calls:
        history_by_signature.setdefault(call_signature(call), []).append(call)
    unlocked_tools = set(state.unlocked_tools)
    domain_rules = DOMAIN_RULES.get(transcript.domain, ())

    violations: list = []
    for index, call in enumerate(turn.tool_calls):
        if call.name == UNLOCK_TOOL and (call.arguments or {}).get("agent_tool_name"):
            unlocked_tools.add(str(call.arguments["agent_tool_name"]))
        if not call.has_valid_arguments:
            violations.append(Violation("bad_json", index, f"{call.name}: невалидный JSON аргументов"))
            continue
        spec = tools.get(call.name)
        if spec is None:
            if call.name not in unlocked_tools:
                violations.append(Violation("unknown_tool", index, f"инструмента {call.name} нет в [AVAILABLE TOOLS]"))
            continue
        violations += _schema_violations(index, call, spec)
        violations += _placeholder_violations(index, call)
        if call.name == CALL_DISCOVERABLE_TOOL:
            violations += _discoverable_call_violations(index, call, unlocked_tools)
        violations += _repeated_failure_violations(index, call, state, history_by_signature)
        for rule in domain_rules:
            violations += rule(index, call, state, transcript.current_time, turn.tool_calls)
    return violations
