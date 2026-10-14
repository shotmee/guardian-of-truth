# Dialogue transcript parser.
#
# What the organisers' wire format looks like:
#   - Lines whose body is a bracketed role tag (left/right-white-square
#     brackets around a word like SYSTEM, USER, ASSISTANT, or
#     "ASSISTANT * turn N") open a new turn.
#   - "\t-> TOOL_CALL name: {json}"              a tool invocation.
#   - "\t<- TOOL_RESPONSE name[ [ERROR]]: text"  the environment's reply.
#   - A line that starts with "\t  " continues whatever came above.
#
# The call that returns everything the agent saw up to its turn is
# ``parse_context``; the audited turn is pulled out by ``parse_agent_turn``.

import json as _json
import re as _re
from datetime import datetime as _dt

# Role tokens -- exported.
AGENT, CUSTOMER, SYSTEM = "assistant", "user", "system"

# Sentinel for the audited (last) turn index.
CHECKED_TURN_INDEX = -1

# Bracket characters that surround role markers in the wire format.
_LB = chr(0x27e6)   # left-white-square bracket
_RB = chr(0x27e7)   # right-white-square bracket

# Regex library.
MARKER_LINE = _re.compile("(?m)^(" + _LB + "[^" + _RB + r"\n]*" + _RB + r")[ \t]*$")
TOOL_CALL_LINE = _re.compile("^\\t" + chr(0x2192) + r" TOOL_CALL ([^:\s]+):\s?(.*)$")
TOOL_RESPONSE_LINE = _re.compile("^\\t" + chr(0x2190) + r" TOOL_RESPONSE ([^:\s\[]+)( \[ERROR\])?:\s?(.*)$")
TURN_NUMBER = _re.compile(chr(0x445) + chr(0x43e) + chr(0x434) + r"\s+(\d+)")
POLICY_TIME = _re.compile(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TOOL_TIME = _re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
ERROR_RESULT = _re.compile(r"^\s*(error\b|exception\b|tool '[^']*' not found)", _re.IGNORECASE)

CONTINUATION_PREFIX = "\t  "
TOOLS_SECTION = "[AVAILABLE TOOLS]"
KNOWN_DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")


class ToolCall:
    __slots__ = ("name", "raw_arguments", "arguments", "caller",
                 "turn_index", "result", "marked_error")

    def __init__(self, name, raw_arguments, arguments, caller, turn_index,
                 result=None, marked_error=False):
        self.name, self.raw_arguments, self.arguments = name, raw_arguments, arguments
        self.caller, self.turn_index = caller, turn_index
        self.result, self.marked_error = result, marked_error

    @property
    def has_valid_arguments(self):
        return self.arguments is not None

    @property
    def failed(self):
        if self.marked_error:
            return True
        return self.result is not None and ERROR_RESULT.match(self.result[:200]) is not None

    def result_as_json(self):
        if self.result is None:
            return None
        try:
            return _json.loads(self.result)
        except ValueError:
            return None

    def describe(self, max_length=600):
        rendered = "%s(%s)" % (self.name, self.raw_arguments)
        return rendered if len(rendered) <= max_length else rendered[:max_length] + chr(0x2026)


class Turn:
    __slots__ = ("role", "number", "text", "tool_calls")

    def __init__(self, role, number, text, tool_calls=None):
        self.role, self.number, self.text = role, number, text
        self.tool_calls = list(tool_calls) if tool_calls else []

    @property
    def has_tool_calls(self):
        return len(self.tool_calls) > 0


class Transcript:
    __slots__ = ("system_text", "policy_text", "tools_text", "turns",
                 "domain", "current_time")

    def __init__(self, system_text, policy_text, tools_text, turns, domain, current_time):
        self.system_text, self.policy_text = system_text, policy_text
        self.tools_text, self.turns = tools_text, turns
        self.domain, self.current_time = domain, current_time

    def agent_calls(self):
        return [c for t in self.turns for c in t.tool_calls if c.caller == AGENT]

    def all_calls(self):
        return [c for t in self.turns for c in t.tool_calls]


# --- internals ---------------------------------------------------------------

def _try_parse_object(raw_text):
    if not raw_text:
        candidate = {}
    else:
        try:
            candidate = _json.loads(raw_text)
        except ValueError:
            return None
    return candidate if isinstance(candidate, dict) else None


def _role_from_marker(marker):
    stripped = marker.strip(_LB + _RB).upper()
    if stripped.startswith("SYSTEM"):
        return SYSTEM
    return CUSTOMER if stripped.startswith("USER") else AGENT


def _turn_number_in(marker):
    found = TURN_NUMBER.search(marker)
    return int(found.group(1)) if found else None


def split_marked_blocks(text):
    # Returns [(marker, body), ...] split on <<...>> markers at the start of a line.
    pieces = MARKER_LINE.split(text)
    pairs = []
    k = 1
    while k < len(pieces):
        pairs.append((pieces[k], pieces[k + 1].strip("\n")))
        k += 2
    return pairs


def detect_domain(system_text, example_id=""):
    head = (example_id or "").split("__")[0]
    if head in KNOWN_DOMAINS:
        return head
    probe = system_text[:3000].lower()
    for needle, label in (("airline", "airline"), ("retail", "retail"),
                          ("telecom", "telecom"), ("bank", "banking_knowledge")):
        if needle in probe:
            return label
    return "other"


def _current_time(system_text, calls):
    # First choice: the clock from policy (used by airline/telecom).
    # Fallback: the newest value seen in a get_current_time result (banking).
    policy_hit = POLICY_TIME.search(system_text)
    if policy_hit is not None:
        try:
            return _dt.strptime(policy_hit.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    newest = None
    for item in calls:
        if item.name != "get_current_time" or not item.result:
            continue
        matched = TOOL_TIME.search(item.result)
        if matched is None:
            continue
        newest = _dt.strptime(matched.group(1) + " " + matched.group(2), "%Y-%m-%d %H:%M:%S")
    return newest


def _parse_turn_body(role, body, turn_index):
    """Build a Turn from a role marker's body: pull out plain text, tool
    invocations and tool results. Results pair with their invocations by
    name in FIFO order."""
    text_rows, call_bag = [], []
    waiting = {}
    cursor = None
    last_was_result = False
    caller_role = AGENT if role == SYSTEM else role

    for line in body.split("\n"):
        hit = TOOL_CALL_LINE.match(line)
        if hit is not None:
            raw_args = hit.group(2).strip()
            cursor = ToolCall(
                name=hit.group(1),
                raw_arguments=raw_args,
                arguments=_try_parse_object(raw_args),
                caller=caller_role,
                turn_index=turn_index,
            )
            call_bag.append(cursor)
            waiting.setdefault(cursor.name, []).append(cursor)
            last_was_result = False
            continue

        hit = TOOL_RESPONSE_LINE.match(line)
        if hit is not None:
            tool_name = hit.group(1)
            queue = waiting.get(tool_name) or []
            target = queue.pop(0) if queue else None
            if target is None:
                target = ToolCall(name=tool_name, raw_arguments="", arguments={},
                                  caller=role, turn_index=turn_index)
                call_bag.append(target)
            target.result = hit.group(3)
            target.marked_error = bool(hit.group(2))
            cursor, last_was_result = target, True
            continue

        if line.startswith(CONTINUATION_PREFIX) and cursor is not None:
            tail = line[len(CONTINUATION_PREFIX):]
            if last_was_result:
                cursor.result = (cursor.result or "") + "\n" + tail
            else:
                cursor.raw_arguments = cursor.raw_arguments + "\n" + tail
                recovered = _try_parse_object(cursor.raw_arguments)
                if recovered is not None:
                    cursor.arguments = recovered
            continue

        text_rows.append(line)

    return Turn(role=role, number=None,
                text="\n".join(text_rows).strip(),
                tool_calls=call_bag)


# --- public parsers ----------------------------------------------------------

def parse_context(prompt, example_id=""):
    system_body = ""
    turns = []
    for marker, body in split_marked_blocks(prompt):
        role = _role_from_marker(marker)
        if role == SYSTEM and not system_body:
            system_body = body
            continue
        this_turn = _parse_turn_body(role, body, turn_index=len(turns))
        this_turn.number = _turn_number_in(marker)
        turns.append(this_turn)

    tools_at = system_body.find(TOOLS_SECTION)
    if tools_at >= 0:
        policy_body = system_body[:tools_at].rstrip()
        tools_block = system_body[tools_at:]
    else:
        policy_body, tools_block = system_body, ""

    every_call = [c for t in turns for c in t.tool_calls]
    return Transcript(
        system_text=system_body,
        policy_text=policy_body,
        tools_text=tools_block,
        turns=turns,
        domain=detect_domain(system_body, example_id),
        current_time=_current_time(system_body, every_call),
    )


def parse_agent_turn(response):
    pairs = split_marked_blocks(response)
    if not pairs:
        return _parse_turn_body(AGENT, response.strip("\n"), CHECKED_TURN_INDEX)
    marker, body = pairs[0]
    audited = _parse_turn_body(AGENT, body, CHECKED_TURN_INDEX)
    audited.number = _turn_number_in(marker)
    for item in audited.tool_calls:
        item.caller = AGENT
    return audited
