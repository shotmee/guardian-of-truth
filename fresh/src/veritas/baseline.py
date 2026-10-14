"""P_base — промпт бейзлайна организаторов: один токен без рассуждения, P('1') из logprobs.

Запрос собирается так же, как в бейзлайне (и в наших замерах): системный промпт организаторов
(prompts/baseline.md), в сообщении пользователя — <prompt>…</prompt> и <response>…</response>.
P('1') считается по top-logprobs непрерывно: если в топе есть только одна цифра, оценка берётся по ней самой,
а не ставится 0/1 — иначе вверху ранжирования появляются ничьи.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chat import ChatTemplate, judge_prompt, shorten_transcript
from .engine import CompletionClient, digit_probability, first_token_alternatives, map_concurrently

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_PROMPT_CHARS = 300000              # как в замерах (v2.1); длиннее в данных почти не бывает


def user_message(prompt: str, response: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    return (f"<prompt>\n{shorten_transcript(prompt, max_chars)}\n</prompt>\n"
            f"<response>\n{response}\n</response>")


def run(items: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float) -> dict:
    """{номер строки: P('1')} для элементов с полями index, prompt и response."""
    system = (PROMPTS_DIR / "baseline.md").read_text(encoding="utf-8").strip()

    def score(item) -> Optional[float]:
        for limit in (MAX_PROMPT_CHARS, MAX_PROMPT_CHARS // 2):
            request = judge_prompt(template, system, user_message(item.prompt, item.response, limit), thinking=False)
            response = client.complete(request, max_tokens=1, temperature=0.0, logprobs=20, seed=1,
                                       skip_special_tokens=False)
            if response is None:
                return None
            if "error" not in response:
                probability = digit_probability(first_token_alternatives(response))
                return 0.5 if probability is None else probability
        return None

    answers = map_concurrently(score, items, int(config.get("workers", 96)), deadline, "B·промпт бейзлайна")
    return {items[position].index: probability for position, probability in answers.items()}
