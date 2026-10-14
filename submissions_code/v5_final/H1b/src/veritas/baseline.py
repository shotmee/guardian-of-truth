# P_base - the organisers' baseline prompt: a single token, no reasoning,
# P('1') read from logprobs.
#
# The request shape mirrors the baseline (and our own measurements): the
# organisers' system prompt in prompts/baseline.md, and the user message is
# <prompt>...</prompt> plus <response>...</response>.
# P('1') is read continuously from top-logprobs; if only one of the two
# digits made the top-k, we use that logprob directly rather than hard
# 0/1 - otherwise ties pile up at the top of the ranking.

from pathlib import Path as _Path

from .chat import judge_prompt, shorten_transcript
from .engine import digit_probability, first_token_alternatives, map_concurrently

_PROMPTS = _Path(__file__).resolve().parent / "prompts"
MAX_PROMPT_CHARS = 300000  # matches v2.1 measurements; data rarely exceeds this.


def user_message(prompt, response, max_chars=MAX_PROMPT_CHARS):
    body = (
        "<prompt>\n%s\n</prompt>\n"
        "<response>\n%s\n</response>"
    ) % (shorten_transcript(prompt, max_chars), response)
    return body


def run(items, config, client, template, deadline):
    """{row_index: P('1')} for every item carrying ``index``, ``prompt`` and
    ``response``."""
    system = (_PROMPTS / "baseline.md").read_text(encoding="utf-8").strip()

    def _score(item):
        for cap in (MAX_PROMPT_CHARS, MAX_PROMPT_CHARS // 2):
            request = judge_prompt(
                template, system,
                user_message(item.prompt, item.response, cap),
                thinking=False,
            )
            response = client.complete(
                request, max_tokens=1, temperature=0.0, logprobs=20,
                seed=1, skip_special_tokens=False,
            )
            if response is None:
                return None
            if "error" in response:
                continue
            top = first_token_alternatives(response)
            value = digit_probability(top)
            return 0.5 if value is None else value
        return None

    workers = int(config.get("workers", 96))
    answers = map_concurrently(_score, items, workers, deadline,
                               "B·промпт бейзлайна")
    out = {}
    for position, probability in answers.items():
        out[items[position].index] = probability
    return out
