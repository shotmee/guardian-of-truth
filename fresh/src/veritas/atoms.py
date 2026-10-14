"""Стратегия C «Атомы»: быстрые однотокенные вопросы к каждому вызову по отдельности.

Каждый вызов хода проверяется тремя узкими вопросами (аргументы / правила политики / уместность шага),
ответ — одна цифра, P('1') из logprobs, без рассуждения. Контекст стоит в начале запроса, вопрос — в
самом конце, поэтому вопросы к одному ходу почти целиком берутся из prefix cache.
score хода = максимум по вызовам от среднего по вопросам.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chat import ChatTemplate, describe_turn, judge_prompt, shorten_transcript
from .engine import CompletionClient, first_token_alternatives, map_concurrently, yes_probability

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
INSTRUCTIONS_TAIL = ("\n\nYou will be asked one narrow question about ONE call of the audited turn. Answer with a "
                     "single digit: 1 or 0. No other text.")
QUESTIONS = {
    "args": ("Question about CALL {number} (`{name}`): is every argument value of this call correct and supported "
             "by the context — taken from the customer's words or earlier tool results, the right entity, correctly "
             "computed, matching the tool schema? Answer 1 if at least one value is invented, wrong, taken from "
             "the wrong place or miscomputed; answer 0 if all values are supported."),
    "policy": ("Question about CALL {number} (`{name}`): given the policy and the facts in the context, is the "
               "agent allowed to perform this call at this moment — the tool is available to the agent, "
               "preconditions are met, required explicit confirmation was obtained, no rule forbids it? Answer 1 "
               "if performing it now violates the policy; answer 0 if it is allowed."),
    "step": ("Question about CALL {number} (`{name}`): compared with what a careful agent following the policy "
             "would do next, is this call a mistake — e.g. a premature transfer while options remain, repeating "
             "a failed call, acting on the wrong item/order/reservation, skipping a required check, or a tool the "
             "agent cannot use? Answer 1 if it is a mistake; answer 0 if it is a sensible step."),
}


def run(candidates: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float) -> dict:
    settings = config.get("atoms", {})
    question_kinds = settings.get("views", list(QUESTIONS))
    system_prompt = (PROMPTS_DIR / "standard.md").read_text(encoding="utf-8") + INSTRUCTIONS_TAIL
    max_chars = int(config.get("max_context_chars", 330000))
    workers = int(config.get("workers", 96))
    max_calls_per_turn = int(settings.get("max_calls", 6))

    # (ход, номер вызова, имя инструмента, вид вопроса); сначала ходы с последствиями —
    # если не хватит времени, потеряем только чтения
    questions = [(item, number, call.name, kind)
                 for item in candidates
                 for number, call in enumerate(item.turn.tool_calls[:max_calls_per_turn], 1)
                 for kind in question_kinds]
    questions.sort(key=lambda q: (not q[0].has_side_effects, q[0].index, q[1]))

    def answer(question) -> Optional[float]:
        item, number, tool_name, kind = question
        text = QUESTIONS[kind].format(number=number, name=tool_name)
        for limit in (max_chars, max_chars // 2, max_chars // 4):    # не влезло в контекст — короче история
            request = ("<CONTEXT>\n" + shorten_transcript(item.prompt, limit) + "\n</CONTEXT>\n\n<AUDITED_TURN>\n"
                       + describe_turn(item.turn) + "\n</AUDITED_TURN>\n\n" + text)
            response = client.complete(judge_prompt(template, system_prompt, request, thinking=False), max_tokens=1,
                                       temperature=0.0, logprobs=20, skip_special_tokens=False)
            if response is None:
                return None
            if "error" not in response:
                return yes_probability(first_token_alternatives(response))
        return None

    answers = map_concurrently(answer, questions, workers, deadline, "C·атомарные вопросы")
    by_turn: dict = {}                  # номер строки -> номер вызова -> вид вопроса -> P
    for position, probability in answers.items():
        item, number, _, kind = questions[position]
        by_turn.setdefault(item.index, {}).setdefault(number, {})[kind] = probability
    return {index: dict(score=max(sum(kinds.values()) / len(kinds) for kinds in calls.values()), calls=calls)
            for index, calls in by_turn.items()}
