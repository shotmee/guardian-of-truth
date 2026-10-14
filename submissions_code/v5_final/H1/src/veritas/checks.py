# Rule-based turn checks (no LLM involved).
#
# Every rule only consults the visible context: the [AVAILABLE TOOLS] block,
# the recorded call history, their results, and the "current time" that may
# come from the policy. When a rule does not have enough data, it stays
# silent. Any single rule firing marks the turn as mistaken.

import json as _j
import re as _re
from datetime import datetime as _dt, timedelta as _td

from .tool_schema import parse_available_tools


def _ic(pattern):
    return _re.compile(pattern, _re.IGNORECASE)


# -------- shared constants (names kept stable for cross-module imports) -----
READ_ONLY_PREFIXES = ("get_", "find_", "search_", "list_", "check_", "kb_search")

READ_ONLY_TOOLS = {
    "grep", "get_current_time",
    "KB_search_bm25", "KB_search_dense", "KB_search",
}

FREE_TEXT_ARGUMENTS = {
    "summary", "query", "pattern", "reason", "note", "message",
    "description", "content", "comment", "expression", "arguments",
}

PLACEHOLDER = _ic(
    r"placeholder|^<[^<>]{1,40}>$|^\{[^{}]{1,40}\}$|\bx{3,}\b|"
    r"^(tbd|todo|unknown|n/?a|none|null|string|example|user_id|order_id|"
    r"item_id|account_id|card_id)$|^your[_ ]|_here$|^\.\.\.$|\[insert"
)

ENTITY_ID_KEYS = (
    "reservation_id", "order_id", "line_id", "bill_id",
    "customer_id", "device_id", "plan_id", "id",
)

UNLOCK_TOOL = "unlock_discoverable_agent_tool"
CALL_DISCOVERABLE_TOOL = "call_discoverable_agent_tool"

# ---- airline tables --------------------------------------------------------
FREE_BAGS_PER_PASSENGER = {
    "regular": dict(basic_economy=0, economy=1, business=2),
    "silver":  dict(basic_economy=1, economy=2, business=3),
    "gold":    dict(basic_economy=2, economy=3, business=4),
}
MAX_PAYMENTS_BY_TYPE = dict(certificate=1, credit_card=1, gift_card=3)
MAX_PASSENGERS = 5
FREE_CANCELLATION_WINDOW = _td(hours=24)

# ---- retail tables ---------------------------------------------------------
ALLOWED_CANCEL_REASONS = ("no longer needed", "ordered by mistake")
PENDING_ORDER_ACTIONS = (
    "cancel_pending_order", "modify_pending_order_items",
    "modify_pending_order_address", "modify_pending_order_payment",
)
DELIVERED_ORDER_ACTIONS = ("return_delivered_order_items", "exchange_delivered_order_items")
ONCE_PER_ORDER_ACTIONS = (
    "return_delivered_order_items", "exchange_delivered_order_items",
    "modify_pending_order_items",
)
ITEM_SWAP_ACTIONS = ("modify_pending_order_items", "exchange_delivered_order_items")

# ---- telecom tables --------------------------------------------------------
MAX_REFUEL_GB = 2


# =============================================================== Violation
class Violation(tuple):
    __slots__ = ()

    def __new__(cls, rule, call_index, message):
        return tuple.__new__(cls, (rule, call_index, message))

    @property
    def rule(self):
        return self[0]

    @property
    def call_index(self):
        return self[1]

    @property
    def message(self):
        return self[2]

    def __repr__(self):
        return "Violation(%r, %r, %r)" % tuple(self)


def _v(rule, index, msg):
    return Violation(rule, index, msg)


# =============================================================== call shape
def is_read_only_call(call):
    cname = call.name
    if cname == CALL_DISCOVERABLE_TOOL:
        inner = str((call.arguments or {}).get("agent_tool_name", ""))
        return inner.lower().startswith(READ_ONLY_PREFIXES)
    return cname in READ_ONLY_TOOLS or cname.lower().startswith(READ_ONLY_PREFIXES)


def has_side_effects(turn):
    return any(not is_read_only_call(c) for c in turn.tool_calls)


# =============================================================== helpers
def normalize_value(payload):
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, str):
        return payload.strip().lower()
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, list):
        return [normalize_value(x) for x in payload]
    if isinstance(payload, dict):
        return {k: normalize_value(v) for k, v in sorted(payload.items())}
    return payload


def call_signature(call):
    # Same name + canonicalised arguments means the same call.
    body = _j.dumps(normalize_value(call.arguments or {}), ensure_ascii=False, sort_keys=True)
    return call.name + "|" + body


def iter_strings(payload, key=""):
    """Yield (key, text) for every string reachable through nested values."""
    if isinstance(payload, str):
        yield key, payload
        return
    if isinstance(payload, list):
        for item in payload:
            for pair in iter_strings(item, key):
                yield pair
        return
    if isinstance(payload, dict):
        for sub_key, sub in payload.items():
            for pair in iter_strings(sub, sub_key):
                yield pair


def parse_datetime(payload):
    if not isinstance(payload, str):
        return None
    stripped = payload.strip()
    has_time = ("T" in stripped) or (":" in stripped)
    cut = stripped[:19] if has_time else stripped[:10]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return _dt.strptime(cut, fmt)
        except ValueError:
            pass
    return None


def _flight_keys(flights):
    bag = set()
    for row in (flights or []):
        if isinstance(row, dict):
            bag.add((str(row.get("flight_number")), str(row.get("date"))))
    return bag


# =============================================================== state
class ConversationState:
    """Everything the agent has already learned from history:
    entity records, user profiles, flight statuses, unlocks."""

    def __init__(self, transcript):
        self.entities_by_id = {}
        self.user_profiles = {}
        self.flight_statuses = {}
        self.unlocked_tools = set()
        self.agent_calls = []
        for item in transcript.agent_calls():
            self.agent_calls.append(item)
            self._absorb(item)

    def entity(self, ident):
        return self.entities_by_id.get(ident) if ident else None

    def _absorb(self, call):
        if call.failed:
            return
        args = call.arguments or {}
        name = call.name
        if name == UNLOCK_TOOL and args.get("agent_tool_name"):
            self.unlocked_tools.add(str(args["agent_tool_name"]))
            return
        if name == "get_flight_status":
            cleaned = (call.result or "").strip().strip('"').lower()
            tag = (str(args.get("flight_number", "")), str(args.get("date", "")))
            self.flight_statuses[tag] = cleaned
            return
        decoded = call.result_as_json()
        if name == "get_user_details" and isinstance(decoded, dict):
            key = str(decoded.get("user_id") or args.get("user_id"))
            self.user_profiles[key] = decoded
        batch = decoded if isinstance(decoded, list) else [decoded]
        for row in batch:
            if isinstance(row, dict):
                self._absorb_entity(row)

    def _absorb_entity(self, row):
        for id_key in ENTITY_ID_KEYS:
            found = row.get(id_key)
            if isinstance(found, str) and found:
                self.entities_by_id[found] = row
                return


# =============================================================== generic rules
def _schema_violations(idx, call, spec):
    bag, args = [], call.arguments or {}
    cname = call.name
    for param in spec.parameters.values():
        if param.required and param.name not in args:
            bag.append(_v("missing_param", idx,
                          "%s: missing required '%s'" % (cname, param.name)))
    for arg_name, arg_val in args.items():
        param = spec.parameters.get(arg_name)
        if param is None:
            bag.append(_v("unknown_param", idx,
                          "%s: extra parameter '%s'" % (cname, arg_name)))
            continue
        bag.extend(_value_violations(idx, cname, param, arg_val))
    return bag


def _value_violations(idx, tool_name, param, candidate):
    if candidate is None:
        return [_v("null_param", idx, "%s: '%s' = null" % (tool_name, param.name))] if param.required else []
    expected = param.type
    if expected == "array" and not isinstance(candidate, list):
        return [_v("type", idx, "%s: '%s' must be an array" % (tool_name, param.name))]
    if expected == "object" and not isinstance(candidate, dict):
        return [_v("type", idx, "%s: '%s' must be an object" % (tool_name, param.name))]
    bag = []
    if param.allowed_values and isinstance(candidate, str) and candidate not in param.allowed_values:
        bag.append(_v("enum", idx,
                      "%s: '%s'=%r outside %r" % (tool_name, param.name, candidate, param.allowed_values)))
    if param.fields:
        iterable = candidate if isinstance(candidate, list) else [candidate]
        for row in iterable:
            if not isinstance(row, dict):
                bag.append(_v("type", idx, "%s: element '%s' is not an object" % (tool_name, param.name)))
                continue
            for sub in param.fields.values():
                if sub.required and sub.name not in row:
                    bag.append(_v("missing_nested", idx,
                                  "%s: '%s' lacks '%s'" % (tool_name, param.name, sub.name)))
                    continue
                if sub.name not in row:
                    continue
                sub_val = row[sub.name]
                if sub.allowed_values and isinstance(sub_val, str) and sub_val not in sub.allowed_values:
                    bag.append(_v("enum", idx,
                                  "%s: '%s.%s' outside enum" % (tool_name, param.name, sub.name)))
    return bag


def _placeholder_violations(idx, call):
    for key, text in iter_strings(call.arguments or {}):
        if key in FREE_TEXT_ARGUMENTS and "placeholder" not in text.lower():
            continue
        stripped = text.strip()
        if PLACEHOLDER.search(stripped):
            return [_v("placeholder", idx, "%s: placeholder %s=%r" % (call.name, key, text[:60]))]
    return []


def _discoverable_call_violations(idx, call, unlocked):
    bag = []
    args = call.arguments or {}
    name = str(args.get("agent_tool_name", ""))
    if name and name not in unlocked:
        bag.append(_v("not_unlocked", idx, "%s not unlocked" % name))
    raw_inner = args.get("arguments")
    if isinstance(raw_inner, str) and raw_inner.strip():
        try:
            inner = _j.loads(raw_inner)
        except ValueError:
            bag.append(_v("bad_json", idx, "arguments: invalid JSON"))
            return bag
        for _, text in iter_strings(inner):
            needle = text.strip()
            if "placeholder" in text.lower() or PLACEHOLDER.search(needle):
                bag.append(_v("placeholder", idx, "placeholder inside arguments: %r" % text[:60]))
                break
    return bag


def _repeated_failure_violations(idx, call, state, by_sig):
    history = by_sig.get(call_signature(call)) or []
    if not history or not history[-1].failed:
        return []
    pos = state.agent_calls.index(history[-1])
    for later in state.agent_calls[pos + 1:]:
        if (not later.failed) and (not is_read_only_call(later)):
            return []
    return [_v("repeat_failed", idx, "%s: retry after prior failure" % call.name)]


# =============================================================== airline rules
# Rule signature: (idx, call, state, now, turn_calls) -> list[Violation].
def _airline_cancellation(idx, call, state, now, _turn):
    if call.name != "cancel_reservation" or now is None:
        return []
    booking_id = str((call.arguments or {}).get("reservation_id", ""))
    booking = state.entity(booking_id)
    if not isinstance(booking, dict):
        return []
    flights = [f for f in (booking.get("flights") or []) if isinstance(f, dict)]
    today_begin = now.replace(hour=0, minute=0, second=0)
    for leg in flights:
        when = parse_datetime(leg.get("date")) or now
        if when < today_begin:
            return [_v("cancel_flown", idx, "%s: cancellation after flown leg" % booking_id)]
    created = parse_datetime(booking.get("created_at"))
    if created is None:
        return []
    cabin_lc = str(booking.get("cabin", "")).lower()
    insurance_lc = str(booking.get("insurance", "")).lower()
    is_recent = (now - created) <= FREE_CANCELLATION_WINDOW
    cancelled_by_carrier = False
    for leg in flights:
        tag = (str(leg.get("flight_number")), str(leg.get("date")))
        if "cancel" in state.flight_statuses.get(tag, ""):
            cancelled_by_carrier = True
            break
    if is_recent or cabin_lc == "business" or insurance_lc == "yes" or cancelled_by_carrier:
        return []
    return [_v("cancel_rules", idx, "%s: no cancellation condition met" % booking_id)]


def _airline_basic_economy_change(idx, call, state, _now, _turn):
    if call.name != "update_reservation_flights":
        return []
    args = call.arguments or {}
    booking_id = str(args.get("reservation_id", ""))
    booking = state.entity(booking_id)
    if not isinstance(booking, dict) or str(booking.get("cabin", "")).lower() != "basic_economy":
        return []
    before = _flight_keys(booking.get("flights"))
    after = _flight_keys(args.get("flights"))
    if before and after and before != after:
        return [_v("basic_economy_modify", idx, "%s: basic economy forbids flight change" % booking_id)]
    return []


def _airline_baggage(idx, call, state, _now, _turn):
    if call.name not in ("update_reservation_baggages", "book_reservation"):
        return []
    args = call.arguments or {}
    booking = None
    if call.name == "book_reservation":
        user_id = str(args.get("user_id", ""))
        cabin_lc = str(args.get("cabin", "")).lower()
        pax_count = len(args.get("passengers") or [])
    else:
        booking = state.entity(str(args.get("reservation_id", "")))
        if not isinstance(booking, dict):
            booking = None
        if booking is None:
            user_id, cabin_lc, pax_count = "", "", 0
        else:
            user_id = str(booking.get("user_id", ""))
            cabin_lc = str(booking.get("cabin", "")).lower()
            pax_count = len(booking.get("passengers") or [])
    out = []
    profile = state.user_profiles.get(user_id)
    tier_lc = str((profile or {}).get("membership", "")).lower()
    total = args.get("total_baggages")
    nonfree = args.get("nonfree_baggages")
    allowance = FREE_BAGS_PER_PASSENGER.get(tier_lc, {}).get(cabin_lc)
    if (profile is not None and allowance is not None and pax_count
            and isinstance(total, int) and isinstance(nonfree, int)):
        expected = max(0, total - allowance * pax_count)
        if nonfree != expected:
            out.append(_v("baggage_calc", idx, "nonfree=%s, policy says %s" % (nonfree, expected)))
    if booking is not None and isinstance(total, int):
        current_total = booking.get("total_baggages")
        if isinstance(current_total, int) and total < current_total:
            out.append(_v("baggage_remove", idx, "removing baggage is forbidden"))
    return out


def _airline_payment_sources(idx, call, state, _now, _turn):
    """A payment id must come from the user's registered methods."""
    args = call.arguments or {}
    payment_ids = []
    if isinstance(args.get("payment_id"), str):
        payment_ids.append(args["payment_id"])
    for pm in (args.get("payment_methods") or []):
        if isinstance(pm, dict) and isinstance(pm.get("payment_id"), str):
            payment_ids.append(pm["payment_id"])
    booking = state.entity(str(args.get("reservation_id", ""))) or {}
    user_key = str(args.get("user_id") or booking.get("user_id") or "")
    profile = state.user_profiles.get(user_key)
    if not profile or not isinstance(profile.get("payment_methods"), dict):
        return []
    known = profile["payment_methods"]
    return [_v("payment_not_in_profile", idx, "%s absent from profile" % pid)
            for pid in payment_ids if pid not in known]


def _airline_booking_limits(idx, call, _state, _now, _turn):
    if call.name != "book_reservation":
        return []
    args = call.arguments or {}
    payment_ids = []
    for pm in (args.get("payment_methods") or []):
        if isinstance(pm, dict):
            payment_ids.append(str(pm.get("payment_id", "")))
    out = []
    for kind, cap in MAX_PAYMENTS_BY_TYPE.items():
        used = sum(1 for pid in payment_ids if pid.startswith(kind))
        if used > cap:
            out.append(_v("payment_limits", idx, "payment-type cap exceeded"))
            break
    if len(args.get("passengers") or []) > MAX_PASSENGERS:
        out.append(_v("too_many_pax", idx, "more than %d passengers" % MAX_PASSENGERS))
    return out


# =============================================================== retail rules
def _order_of(call, state):
    order_id = str((call.arguments or {}).get("order_id", ""))
    rec = state.entity(order_id)
    return order_id, (rec if isinstance(rec, dict) else None)


def _retail_item_pairs(idx, call, _state, _now, _turn):
    if call.name not in ITEM_SWAP_ACTIONS:
        return []
    args = call.arguments or {}
    before = args.get("item_ids") or []
    after = args.get("new_item_ids") or []
    if not isinstance(before, list) or not isinstance(after, list):
        return []
    if len(before) != len(after):
        return [_v("items_len", idx, "item_ids and new_item_ids have different lengths")]
    for a, b in zip(before, after):
        if str(a) == str(b):
            return [_v("same_item", idx, "new_item_id equals old one")]
    return []


def _retail_order_status(idx, call, state, _now, _turn):
    order_id, order = _order_of(call, state)
    status_lc = str((order or {}).get("status", "")).lower()
    if not status_lc:
        return []
    if call.name in PENDING_ORDER_ACTIONS and status_lc != "pending":
        return [_v("order_status", idx, "%s: status '%s', wanted pending" % (order_id, status_lc))]
    if call.name in DELIVERED_ORDER_ACTIONS and status_lc != "delivered":
        return [_v("order_status", idx, "%s: status '%s', wanted delivered" % (order_id, status_lc))]
    return []


def _retail_cancel_reason(idx, call, _state, _now, _turn):
    if call.name != "cancel_pending_order":
        return []
    reason_lc = str((call.arguments or {}).get("reason", "")).strip().lower()
    if reason_lc and reason_lc not in ALLOWED_CANCEL_REASONS:
        return [_v("cancel_reason", idx, "reason '%s' not in policy" % reason_lc)]
    return []


def _retail_single_change(idx, call, state, _now, turn_calls):
    """return / exchange / modify may happen at most once per order."""
    order_id = str((call.arguments or {}).get("order_id", ""))
    if call.name not in ONCE_PER_ORDER_ACTIONS or not order_id:
        return []

    def _touches(other):
        if other.name not in ONCE_PER_ORDER_ACTIONS:
            return False
        return str((other.arguments or {}).get("order_id", "")) == order_id

    for past in state.agent_calls:
        if _touches(past) and not past.failed:
            return [_v("once_per_order", idx, "%s: second change" % order_id)]
    for earlier in turn_calls[:idx]:
        if _touches(earlier):
            return [_v("once_per_order", idx, "%s: second change" % order_id)]
    return []


def _retail_payment(idx, call, state, _now, _turn):
    pm = (call.arguments or {}).get("payment_method_id")
    if not isinstance(pm, str):
        return []
    _, order = _order_of(call, state)
    user_id = str((order or {}).get("user_id", ""))
    if user_id:
        profile = state.user_profiles.get(user_id)
    elif len(state.user_profiles) == 1:
        profile = next(iter(state.user_profiles.values()))
    else:
        profile = None
    bag = []
    methods = (profile or {}).get("payment_methods")
    if isinstance(methods, dict) and methods and pm not in methods:
        bag.append(_v("payment_not_in_profile", idx, "%s absent from profile" % pm))
    if call.name == "return_delivered_order_items" and order is not None:
        originals = set()
        for p in (order.get("payment_history") or []):
            if isinstance(p, dict):
                originals.add(str(p.get("payment_method_id")))
        if originals and pm not in originals and not pm.startswith("gift_card"):
            bag.append(_v("refund_method", idx, "refund to %s: not original and not gift card" % pm))
    return bag


# =============================================================== telecom rules
def _telecom_resume_line(idx, call, state, now, _turn):
    if call.name != "resume_line" or now is None:
        return []
    row = state.entity(str((call.arguments or {}).get("line_id", "")))
    if not isinstance(row, dict):
        return []
    end = parse_datetime(row.get("contract_end_date"))
    if end is not None and end < now:
        return [_v("resume_expired", idx, "line contract expired; cannot resume")]
    return []


def _telecom_refuel_limit(idx, call, _state, _now, _turn):
    if call.name != "refuel_data":
        return []
    try:
        amount = float((call.arguments or {}).get("gb_amount", 0))
    except (TypeError, ValueError):
        return []
    if amount > MAX_REFUEL_GB:
        return [_v("refuel_max", idx, "more than %d GB in a single request" % MAX_REFUEL_GB)]
    return []


def _telecom_payment_request(idx, call, state, _now, _turn):
    if call.name != "send_payment_request":
        return []
    bill = state.entity(str((call.arguments or {}).get("bill_id", "")))
    if isinstance(bill, dict) and bill.get("status") and "overdue" not in str(bill["status"]).lower():
        return [_v("bill_not_overdue", idx, "bill status %s" % bill["status"])]
    return []


DOMAIN_RULES = dict(
    airline=(
        _airline_cancellation, _airline_basic_economy_change, _airline_baggage,
        _airline_payment_sources, _airline_booking_limits,
    ),
    retail=(
        _retail_item_pairs, _retail_order_status, _retail_cancel_reason,
        _retail_single_change, _retail_payment,
    ),
    telecom=(_telecom_resume_line, _telecom_refuel_limit, _telecom_payment_request),
)


# =============================================================== entry point
def find_violations(transcript, turn, tools=None, state=None):
    if tools is None:
        tools = parse_available_tools(transcript.tools_text)
    if state is None:
        state = ConversationState(transcript)

    by_sig = {}
    for past in state.agent_calls:
        by_sig.setdefault(call_signature(past), []).append(past)

    unlocked = set(state.unlocked_tools)
    rules = DOMAIN_RULES.get(transcript.domain, ())
    out = []

    for idx, call in enumerate(turn.tool_calls):
        if call.name == UNLOCK_TOOL and (call.arguments or {}).get("agent_tool_name"):
            unlocked.add(str(call.arguments["agent_tool_name"]))

        if not call.has_valid_arguments:
            out.append(_v("bad_json", idx, "%s: invalid JSON arguments" % call.name))
            continue

        spec = tools.get(call.name)
        if spec is None:
            if call.name not in unlocked:
                out.append(_v("unknown_tool", idx, "tool %s not in [AVAILABLE TOOLS]" % call.name))
            continue

        out.extend(_schema_violations(idx, call, spec))
        out.extend(_placeholder_violations(idx, call))
        if call.name == CALL_DISCOVERABLE_TOOL:
            out.extend(_discoverable_call_violations(idx, call, unlocked))
        out.extend(_repeated_failure_violations(idx, call, state, by_sig))
        for rule in rules:
            out.extend(rule(idx, call, state, transcript.current_time, turn.tool_calls))

    return out
