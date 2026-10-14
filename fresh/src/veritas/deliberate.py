"""Судья с рассуждением для ходов без вызовов: модель сначала думает, потом отвечает одной цифрой.

Два запроса на ход:
  1. thinking-режим до «</think>» (стоп-строка), не больше max_think токенов;
  2. тот же промпт + рассуждение + «</think>\\n\\n», один токен с logprobs → P('1').
Так вероятность берётся из logprobs ответа после рассуждения, а не из текста (в v1 это было слабым местом).
Системный промпт — стандарт текстовых ходов (prompts/text_standard.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .chat import ChatTemplate, judge_prompt, shorten_transcript
from .engine import CompletionClient, first_token_alternatives, map_concurrently, yes_probability

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
INSTRUCTIONS_TAIL = ("\n\nThink it through step by step: check every specific statement of the AUDITED MESSAGE against the "
                     "CONTEXT and the policy. Then answer with a single digit: 1 if the message contains an error, "
                     "0 if it does not.")
END_OF_THINKING = "</think>"
TRUNCATION_NOTE = "\n\n(Thinking budget is over, deciding now.)"


def run(turns: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float) -> dict:
    """{номер строки: {"p": P('1'), "tokens": длина рассуждения в символах, "truncated": bool}}."""
    settings = config.get("deliberate", {})
    max_think = int(settings.get("max_think", 1536))
    temperature = float(settings.get("temperature", 0.6))
    system = (PROMPTS_DIR / "text_standard.md").read_text(encoding="utf-8") + INSTRUCTIONS_TAIL
    max_chars = int(config.get("max_context_chars", 330000))

    def judge(turn) -> Optional[dict]:
        for limit in (max_chars, max_chars // 2):
            request = ("<CONTEXT>\n" + shorten_transcript(turn.prompt, limit) + "\n</CONTEXT>\n\n<AUDITED_MESSAGE>\n"
                       + turn.text + "\n</AUDITED_MESSAGE>")
            prompt = judge_prompt(template, system, request, thinking=True)
            thought = client.complete(prompt, max_tokens=max_think, temperature=temperature, top_p=0.95, seed=1,
                                      stop=[END_OF_THINKING], skip_special_tokens=False)
            if thought is None:
                return None
            if "error" in thought:
                continue
            choice = thought["choices"][0]
            reasoning = choice.get("text", "")
            truncated = choice.get("finish_reason") == "length"
            closing = reasoning.rstrip() + (TRUNCATION_NOTE if truncated else "") + "\n" + END_OF_THINKING + "\n\n"
            answer = client.complete(prompt + closing, max_tokens=1, temperature=0.0, logprobs=20,
                                     skip_special_tokens=False)
            if answer is None or "error" in answer:
                return None
            probability = yes_probability(first_token_alternatives(answer))
            return None if probability is None else {"p": probability, "tokens": len(reasoning), "truncated": truncated}
        return None

    workers = int(settings.get("workers", 64))
    answers = map_concurrently(judge, turns, workers, deadline, "D·рассуждение")
    return {turns[position].index: result for position, result in answers.items()}
