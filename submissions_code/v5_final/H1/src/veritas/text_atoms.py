# Text "atoms": narrow single-token questions aimed at a turn without tool
# calls (the agent only writes to the customer).
#
# Five questions by error type plus one overall:
#   facts        - a specific fact contradicts the context or is unsupported;
#   action_claim - the agent reports an action it did not perform and cannot;
#   reask        - re-asking something the customer already said / did;
#   refusal      - unjustified refusal or human-agent transfer;
#   policy       - promises or confirms what policy forbids;
#   overall      - the same standard, as a single final verdict.
# One digit answer, P('1') via top-logprobs, no reasoning. Because the
# context sits first and the question last, questions for the same turn
# almost entirely share prefix cache.

from pathlib import Path as _Path

from .chat import judge_prompt, shorten_transcript
from .engine import first_token_alternatives, map_concurrently, yes_probability

_PROMPTS = _Path(__file__).resolve().parent / "prompts"

INSTRUCTIONS_TAIL = (
    "\n\nYou will be asked one narrow question about the AUDITED MESSAGE. "
    "Answer with a single digit: 1 or 0. No other text."
)

QUESTIONS = {
    "facts": (
        "Question: does the AUDITED MESSAGE state at least one specific fact "
        "(a number, price, fee, date, limit, status, name, product feature, "
        "condition, reason or computed value) that contradicts the CONTEXT or "
        "is not supported by it? Answer 1 if yes; answer 0 if every such fact "
        "is supported."
    ),
    "action_claim": (
        "Question: does the AUDITED MESSAGE say that an action has been done, "
        "is being done right now, or will be done by the agent itself "
        "(booking, cancelling, changing, sending, ordering, resetting, "
        "checking, transferring), although no successful tool call in the "
        "CONTEXT did it and no available tool can do it? Answer 1 if yes; "
        "answer 0 otherwise."
    ),
    "reask": (
        "Question: does the AUDITED MESSAGE ask the customer for information "
        "or for an action that the customer has already provided or performed "
        "earlier, or that is already available in earlier tool results? "
        "Answer 1 if yes; answer 0 otherwise."
    ),
    "refusal": (
        "Question: does the AUDITED MESSAGE refuse, say that something is "
        "impossible, or offer or push a transfer to a human agent, although "
        "the policy and the available tools let the agent make progress "
        "itself or the policy does not allow the transfer yet? Answer 1 if "
        "yes; answer 0 if there is no such refusal or transfer, or it is "
        "justified."
    ),
    "policy": (
        "Question: does the AUDITED MESSAGE offer, promise or confirm an "
        "option, action or condition that the policy forbids, or state a "
        "policy rule incorrectly? Answer 1 if yes; answer 0 otherwise."
    ),
    "overall": (
        "Question: taking everything into account, does the AUDITED MESSAGE "
        "contain an error as defined above? Answer 1 if yes; answer 0 if no."
    ),
}


class TextTurn:
    __slots__ = ("index", "prompt", "text")

    def __init__(self, index, prompt, text):
        self.index = index
        self.prompt = prompt
        self.text = text


def run(turns, config, client, template, deadline):
    """{row_index: {"score": mean of questions, "answers": {q: P('1')}}}."""
    settings = config.get("text_atoms", {})
    question_kinds = settings.get("questions", list(QUESTIONS))
    system_prompt = (_PROMPTS / "text_standard.md").read_text(encoding="utf-8") + INSTRUCTIONS_TAIL
    max_chars = int(config.get("max_context_chars", 330000))
    workers = int(config.get("workers", 96))

    questions = [(turn, kind) for turn in turns for kind in question_kinds]

    def _answer(entry):
        turn, kind = entry
        for cap in (max_chars, max_chars // 2, max_chars // 4):
            payload = (
                "<CONTEXT>\n" + shorten_transcript(turn.prompt, cap) + "\n</CONTEXT>\n\n"
                + "<AUDITED_MESSAGE>\n" + turn.text + "\n</AUDITED_MESSAGE>\n\n"
                + QUESTIONS[kind]
            )
            resp = client.complete(
                judge_prompt(template, system_prompt, payload, thinking=False),
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
                           "T·текстовые вопросы")

    collected = {}
    for position, probability in raw.items():
        turn, kind = questions[position]
        collected.setdefault(turn.index, {})[kind] = probability

    out = {}
    for row_index, kinds in collected.items():
        mean = sum(kinds.values()) / len(kinds)
        out[row_index] = {"score": mean, "answers": kinds}
    return out
