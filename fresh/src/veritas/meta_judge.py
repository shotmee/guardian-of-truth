"""P_meta — промпт бейзлайна организаторов + метаданные задачи τ² (то, что видел их разметчик).

Запрос как у P_base (baseline.py), но в сообщении пользователя перед <prompt> идёт блок <task_spec> с
метаданными задачи, а к системному промпту добавлено объяснение, как ими пользоваться. Ходы, для которых задача
не нашлась (task_meta.find_spec вернул None), не оцениваются — для них остаётся P_base.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .baseline import MAX_PROMPT_CHARS, user_message
from .chat import ChatTemplate, judge_prompt
from .engine import CompletionClient, digit_probability, first_token_alternatives, map_concurrently
from .task_meta import find_spec, render_spec

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
META_NOTE = """

## Метаданные задачи (агент их не видел)

Перед контекстом дан блок `<task_spec>` — скрытая спецификация задачи, по которой шёл диалог:
- `instructions` — чего на самом деле хочет клиент, что он знает и как себя ведёт;
- `expected_actions` — эталонные действия агента с аргументами (решение, верное по политике);
- `communicate_info` — что агент должен сообщить клиенту; `nl_assertions` — проверки результата.

Используй их, чтобы понять, чего клиент хотел и какое решение правильное по политике. Ошибка — если проверяемый
ответ уводит от правильного итога: действие, которого нет в эталоне, или с другими аргументами; отказ или перевод
на оператора там, где эталон предполагает действие; неверные значения или условия; утверждения, противоречащие
данным. Обычные промежуточные шаги к эталону (уточнить данные, найти запись, запросить подтверждение) — не ошибка.
Сама нехватка у агента скрытых сведений — не ошибка: оценивай ответ по тому, что агент мог знать и сделать."""


def system_prompt() -> str:
    return (PROMPTS_DIR / "baseline.md").read_text(encoding="utf-8").strip() + META_NOTE


def meta_message(spec_text: str, prompt: str, response: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    return f"<task_spec>\n{spec_text}\n</task_spec>\n" + user_message(prompt, response, max_chars)


def run(items: list, config: dict, client: CompletionClient, template: ChatTemplate, deadline: float,
        example_ids: Optional[list] = None) -> tuple:
    """({номер строки: P('1')}, {номер строки: как найдена задача}) для элементов с index, prompt, response."""
    system = system_prompt()
    specs, how = {}, {}
    for item in items:
        example_id = example_ids[item.index] if example_ids is not None else str(item.index)
        spec, found = find_spec(example_id, item.prompt)
        how[item.index] = found
        if spec is not None:
            specs[item.index] = render_spec(spec)
    todo = [item for item in items if item.index in specs]

    def score(item) -> Optional[float]:
        for limit in (MAX_PROMPT_CHARS, MAX_PROMPT_CHARS // 2):
            request = judge_prompt(template, system, meta_message(specs[item.index], item.prompt, item.response, limit),
                                   thinking=False)
            response = client.complete(request, max_tokens=1, temperature=0.0, logprobs=20, seed=1,
                                       skip_special_tokens=False)
            if response is None:
                return None
            if "error" not in response:
                probability = digit_probability(first_token_alternatives(response))
                return 0.5 if probability is None else probability
        return None

    answers = map_concurrently(score, todo, int(config.get("workers", 96)), deadline, "M·метаданные задачи")
    return {todo[position].index: p for position, p in answers.items()}, how
