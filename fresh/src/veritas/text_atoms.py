"""Текстовые «атомы»: узкие однотокенные вопросы к ходу без вызовов (агент только пишет клиенту).

К каждому ходу — пять вопросов по типам ошибок и один общий:
  facts — факт противоречит контексту или ничем не подкреплён;
  action_claim — агент говорит о действии, которого не было и которое ему недоступно;
  reask — снова просит то, что клиент уже сообщил или сделал;
  refusal — необоснованный отказ или перевод на оператора;
  policy — обещает или подтверждает то, что запрещает политика;
  overall — общий вердикт по тому же стандарту.
Ответ — одна цифра, P('1') из logprobs, без рассуждения. Контекст стоит в начале запроса, вопрос — в конце,
поэтому вопросы к одному ходу почти целиком берутся из prefix cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chat import ChatTemplate, judge_prompt, shorten_transcript
from .engine import CompletionClient, first_token_alternatives, map_concurrently, yes_probability

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
INSTRUCTIONS_TAIL = ("\n\nYou will be asked one narrow question about the AUDITED MESSAGE. Answer with a single digit: "
                     "1 or 0. No other text.")
QUESTIONS = {
    "facts": ("Question: does the AUDITED MESSAGE state at least one specific fact (a number, price, fee, date, limit, "
              "status, name, product feature, condition, reason or computed value) that contradicts the CONTEXT or is "
              "not supported by it? Answer 1 if yes; answer 0 if every such fact is supported."),
    "action_claim": ("Question: does the AUDITED MESSAGE say that an action has been done, is being done right now, "
                     "or will be done by the agent itself (booking, cancelling, changing, sending, ordering, resetting, "
                     "checking, transferring), although no successful tool call in the CONTEXT did it and no available "
                     "tool can do it? Answer 1 if yes; answer 0 otherwise."),
    "reask": ("Question: does the AUDITED MESSAGE ask the customer for information or for an action that the customer "
              "has already provided or performed earlier, or that is already available in earlier tool results? "
              "Answer 1 if yes; answer 0 otherwise."),
    "refusal": ("Question: does the AUDITED MESSAGE refuse, say that something is impossible, or offer or push a "
                "transfer to a human agent, although the policy and the available tools let the agent make progress "
                "itself or the policy does not allow the transfer yet? Answer 1 if yes; answer 0 if there is no such "
                "refusal or transfer, or it is justified."),
    "policy": ("Question: does the AUDITED MESSAGE offer, promise or confirm an option, action or condition that the "
               "policy forbids, or state a policy rule incorrectly? Answer 1 if yes; answer 0 otherwise."),
    "overall": ("Question: taking everything into account, does the AUDITED MESSAGE contain an error as defined above? "
                "Answer 1 if yes; answer 0 if no."),
}


@dataclass(frozen=True)
class TextTurn:
    index: int                         # номер строки во входном файле
    prompt: str                        # контекст, который видел агент
    text: str                          # текст проверяемого хода


def run(turns: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float) -> dict:
    """{номер строки: {"score": среднее по вопросам, "answers": {вопрос: P('1')}}}."""
    settings = config.get("text_atoms", {})
    question_kinds = settings.get("questions", list(QUESTIONS))
    system_prompt = (PROMPTS_DIR / "text_standard.md").read_text(encoding="utf-8") + INSTRUCTIONS_TAIL
    max_chars = int(config.get("max_context_chars", 330000))
    workers = int(config.get("workers", 96))
    questions = [(turn, kind) for turn in turns for kind in question_kinds]

    def answer(question) -> Optional[float]:
        turn, kind = question
        for limit in (max_chars, max_chars // 2, max_chars // 4):    # не влезло в контекст — короче история
            request = ("<CONTEXT>\n" + shorten_transcript(turn.prompt, limit) + "\n</CONTEXT>\n\n<AUDITED_MESSAGE>\n"
                       + turn.text + "\n</AUDITED_MESSAGE>\n\n" + QUESTIONS[kind])
            response = client.complete(judge_prompt(template, system_prompt, request, thinking=False), max_tokens=1,
                                       temperature=0.0, logprobs=20, skip_special_tokens=False)
            if response is None:
                return None
            if "error" not in response:
                return yes_probability(first_token_alternatives(response))
        return None

    answers = map_concurrently(answer, questions, workers, deadline, "T·текстовые вопросы")
    by_turn: dict = {}
    for position, probability in answers.items():
        turn, kind = questions[position]
        by_turn.setdefault(turn.index, {})[kind] = probability
    return {index: dict(score=sum(kinds.values()) / len(kinds), answers=kinds) for index, kinds in by_turn.items()}
