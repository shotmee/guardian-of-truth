"""Блок примеров разметки организаторов для промпта судьи (вариант C): prompts/organizer_examples.md.

Из valid (46 ходов с метками организаторов): все 23 ошибки — ответ агента (усечённый) и объяснение организаторов;
23 не-ошибки — ответ агента и метка 0. Блок одинаков для всех запросов, поэтому vLLM кэширует его как префикс.
python dev/build_exemplars.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "veritas" / "prompts" / "organizer_examples.md"
RESPONSE_CHARS, EXPLANATION_CHARS, NEGATIVE_CHARS = 700, 900, 400


def cut(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " …"


def main() -> None:
    valid = pd.read_parquet(ROOT.parent / "valid.parquet")
    lines = ["## Примеры разметки (настоящие метки и объяснения разметчиков)",
             "", "Ниже ответы агентов из других диалогов с метками. Контекст опущен; примеры показывают, что считается "
             "ошибкой (1), а что нет (0). Оценивай проверяемый ответ по тем же правилам.", ""]
    for number, row in enumerate(valid[valid.label == 1].itertuples(), 1):
        lines += [f"### Ошибка {number} (метка 1)", f"Ответ: {cut(row.response, RESPONSE_CHARS)}",
                  f"Почему ошибка: {cut(row.explanation, EXPLANATION_CHARS)}", ""]
    for number, row in enumerate(valid[valid.label == 0].itertuples(), 1):
        lines += [f"### Не ошибка {number} (метка 0)", f"Ответ: {cut(row.response, NEGATIVE_CHARS)}", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} КБ")


if __name__ == "__main__":
    main()
