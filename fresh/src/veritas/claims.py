"""Проверка по утверждениям: модель выписывает из ответа клиенту конкретные факты, затем каждый факт
проверяется по контексту отдельным однотокенным вопросом.

1. Выписка — короткий запрос только с текстом ответа (контекст не нужен), генерация до ~300 токенов.
2. Проверка — по запросу на факт: контекст в начале (prefix cache на все факты хода), факт в конце,
   ответ одной цифрой, P('1') из logprobs.
Оценка хода — максимум и среднее по фактам; ход без фактов получает 0.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .chat import ChatTemplate, judge_prompt, shorten_transcript
from .engine import CompletionClient, first_token_alternatives, map_concurrently, yes_probability

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$")
MAX_CLAIMS = 6
EXTRACTION_TOKENS = 400


def parse_claims(text: str) -> list:
    """Пронумерованные строки ответа модели; «NONE» или пустой ответ — фактов нет."""
    claims = []
    for line in (text or "").splitlines():
        match = NUMBERED_LINE.match(line)
        if match and match.group(1).strip().upper() != "NONE":
            claims.append(match.group(1).strip())
    return claims[:MAX_CLAIMS]


def extract(turns: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float) -> dict:
    """{номер строки: [факты]} для ходов без вызовов (TextTurn)."""
    system = (PROMPTS_DIR / "claim_extraction.md").read_text(encoding="utf-8")

    def ask(turn) -> Optional[list]:
        prompt = judge_prompt(template, system, "AGENT MESSAGE:\n" + turn.text, thinking=False)
        response = client.complete(prompt, max_tokens=EXTRACTION_TOKENS, temperature=0.0)
        if response is None or "error" in response:
            return None
        return parse_claims(response["choices"][0].get("text", ""))

    answers = map_concurrently(ask, turns, int(config.get("workers", 96)), deadline, "K·выписка фактов")
    return {turns[position].index: claims for position, claims in answers.items()}


def verify(turns: list, claims_by_turn: dict, config: dict, client: CompletionClient, template: ChatTemplate,
           deadline: float) -> dict:
    """{номер строки: {"max": …, "mean": …, "count": …, "claims": [(факт, P)]}}."""
    system = (PROMPTS_DIR / "claim_check.md").read_text(encoding="utf-8")
    max_chars = int(config.get("max_context_chars", 330000))
    by_index = {turn.index: turn for turn in turns}
    questions = [(index, claim) for index, claims in claims_by_turn.items() for claim in claims]

    def ask(question) -> Optional[float]:
        index, claim = question
        turn = by_index[index]
        for limit in (max_chars, max_chars // 2, max_chars // 4):
            request = ("<CONTEXT>\n" + shorten_transcript(turn.prompt, limit) + "\n</CONTEXT>\n\n<AGENT_MESSAGE>\n"
                       + turn.text + "\n</AGENT_MESSAGE>\n\n<CLAIM>\n" + claim + "\n</CLAIM>")
            response = client.complete(judge_prompt(template, system, request, thinking=False), max_tokens=1,
                                       temperature=0.0, logprobs=20, skip_special_tokens=False)
            if response is None:
                return None
            if "error" not in response:
                return yes_probability(first_token_alternatives(response))
        return None

    answers = map_concurrently(ask, questions, int(config.get("workers", 96)), deadline, "K·проверка фактов")
    results: dict = {index: {"max": 0.0, "mean": 0.0, "count": 0, "claims": []} for index in claims_by_turn}
    for position, probability in answers.items():
        index, claim = questions[position]
        results[index]["claims"].append((claim, probability))
    for summary in results.values():
        values = [p for _, p in summary["claims"] if p is not None]
        if values:
            summary.update(max=max(values), mean=sum(values) / len(values), count=len(values))
    return results


def run(turns: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float) -> dict:
    claims_by_turn = extract(turns, config, client, template, deadline)
    return verify(turns, claims_by_turn, config, client, template, deadline)
