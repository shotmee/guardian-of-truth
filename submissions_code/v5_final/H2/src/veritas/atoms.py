# Strategy C ("Atoms"): three narrow single-token questions aimed at each
# individual call in a turn (arguments / policy / step).
#
# Every question gets a single-digit answer, P('1') read from top-logprobs
# with no chain of thought. The context is placed first and the question is
# the last line, so questions for one and the same turn share almost all of
# their tokens via the prefix cache.
# The turn's score is the maximum over its calls of the mean over the
# questions.

from pathlib import Path as _Path

from .chat import describe_turn, judge_prompt, shorten_transcript
from .engine import first_token_alternatives, map_concurrently, yes_probability

_PROMPTS = _Path(__file__).resolve().parent / "prompts"

INSTRUCTIONS_TAIL = (
    "\n\nYou will be asked one narrow question about ONE call of the audited "
    "turn. Answer with a single digit: 1 or 0. No other text."
)

QUESTIONS = {
    "args": (
        "Question about CALL {number} (`{name}`): is every argument value of "
        "this call correct and supported by the context — taken from the "
        "customer's words or earlier tool results, the right entity, correctly "
        "computed, matching the tool schema? Answer 1 if at least one value is "
        "invented, wrong, taken from the wrong place or miscomputed; answer 0 "
        "if all values are supported."
    ),
    "policy": (
        "Question about CALL {number} (`{name}`): given the policy and the "
        "facts in the context, is the agent allowed to perform this call at "
        "this moment — the tool is available to the agent, preconditions "
        "are met, required explicit confirmation was obtained, no rule forbids "
        "it? Answer 1 if performing it now violates the policy; answer 0 if it "
        "is allowed."
    ),
    "step": (
        "Question about CALL {number} (`{name}`): compared with what a careful "
        "agent following the policy would do next, is this call a mistake "
        "— e.g. a premature transfer while options remain, repeating a "
        "failed call, acting on the wrong item/order/reservation, skipping a "
        "required check, or a tool the agent cannot use? Answer 1 if it is a "
        "mistake; answer 0 if it is a sensible step."
    ),
}


def _build_questions(candidates, question_kinds, max_calls_per_turn):
    bag = []
    for item in candidates:
        for n, call in enumerate(item.turn.tool_calls[:max_calls_per_turn], 1):
            for kind in question_kinds:
                bag.append((item, n, call.name, kind))
    # Side-effect turns go first; if we run out of time, we only lose reads.
    bag.sort(key=lambda q: (not q[0].has_side_effects, q[0].index, q[1]))
    return bag


def run(candidates, config, client, template, deadline):
    settings = config.get("atoms", {})
    question_kinds = settings.get("views", list(QUESTIONS))
    max_calls_per_turn = int(settings.get("max_calls", 6))
    max_chars = int(config.get("max_context_chars", 330000))
    workers = int(config.get("workers", 96))

    system_prompt = (_PROMPTS / "standard.md").read_text(encoding="utf-8") + INSTRUCTIONS_TAIL
    questions = _build_questions(candidates, question_kinds, max_calls_per_turn)

    def _answer(question):
        item, number, tool_name, kind = question
        tail = QUESTIONS[kind].format(number=number, name=tool_name)
        for cap in (max_chars, max_chars // 2, max_chars // 4):
            body = (
                "<CONTEXT>\n" + shorten_transcript(item.prompt, cap) + "\n</CONTEXT>\n\n"
                + "<AUDITED_TURN>\n" + describe_turn(item.turn) + "\n</AUDITED_TURN>\n\n"
                + tail
            )
            resp = client.complete(
                judge_prompt(template, system_prompt, body, thinking=False),
                max_tokens=1, temperature=0.0, logprobs=20,
                skip_special_tokens=False,
            )
            if resp is None:
                return None
            if "error" in resp:
                continue
            return yes_probability(first_token_alternatives(resp))
        return None

    raw = map_concurrently(_answer, questions, workers, deadline,
                           "C·атомарные вопросы")

    per_turn = {}
    for position, probability in raw.items():
        item, number, _tool, kind = questions[position]
        per_turn.setdefault(item.index, {}).setdefault(number, {})[kind] = probability

    out = {}
    for turn_index, calls in per_turn.items():
        best = 0.0
        for call_scores in calls.values():
            mean = sum(call_scores.values()) / len(call_scores)
            if mean > best:
                best = mean
        out[turn_index] = {"score": best, "calls": calls}
    return out
