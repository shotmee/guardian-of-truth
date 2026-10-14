"""Прогноз F1 на паблике для решения, собранного из измеренных блоков.

Опора — факты с лидерборда (06.10):
    N = 546; ходы без вызовов: 380, в них 96 ошибок; hard-проверки: 18 флагов, 11 ошибок;
    остальные ходы с вызовами: 148, в них 23 ошибки.
    «Атомы» на ходах с вызовами: топ-83 из 148 поймал 22 из 23 (v4-C).
    P_base (промпт бейзлайна) на тексте: ≥0.8 ≈ 70 ходов, ≥0.5 ≈ 213 ходов; TP — из разложения
    сабмитов бейзлайна (94/254) и v3.0 (42/86) с перебором неизвестных долей (Монте-Карло).
Кривые ранжирования — бинормальные: d' вызовов подбирается по точке 22/23, d' текста — по двум точкам P_base.
Разброс прогноза отражает неопределённость этих допущений, а не независимую проверку.

python dev/lb_simulator.py
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

TEXT_TURNS, TEXT_ERRORS = 380, 96
HARD_FLAGS, HARD_ERRORS = 18, 11
CALL_TURNS, CALL_ERRORS = 148, 23
POSITIVES = 130


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def threshold_for(separation: float, positives: int, negatives: int, flags: float) -> float:
    low, high = -6.0, 6.0
    for _ in range(60):
        middle = (low + high) / 2
        flagged = positives * normal_cdf(separation - middle) + negatives * normal_cdf(-middle)
        low, high = (middle, high) if flagged > flags else (low, middle)
    return (low + high) / 2


def true_positives(separation: float, positives: int, negatives: int, flags: float) -> float:
    return positives * normal_cdf(separation - threshold_for(separation, positives, negatives, flags))


def call_separation_posterior(grid: np.ndarray) -> np.ndarray:
    """Апостериорное распределение d' «Атомов» по факту «22 из 23 в топ-83»."""
    weights = []
    for separation in grid:
        hit = true_positives(separation, CALL_ERRORS, CALL_TURNS - CALL_ERRORS, 83) / CALL_ERRORS
        hit = min(max(hit, 1e-9), 1 - 1e-9)
        weights.append(hit ** 22 * (1 - hit))
    weights = np.array(weights)
    return weights / weights.sum()


def text_separation(points: list) -> float:
    """d' текстового ранжировщика по точкам (флагов, TP)."""
    grid = np.linspace(0.1, 3.0, 146)
    errors = [sum((true_positives(d, TEXT_ERRORS, TEXT_TURNS - TEXT_ERRORS, f) - tp) ** 2 for f, tp in points)
              for d in grid]
    return float(grid[int(np.argmin(errors))])


def sample_scenarios(count: int = 600, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.3, 3.5, 321)
    posterior = call_separation_posterior(grid)
    scenarios = []
    for _ in range(count):
        hard_flags = rng.uniform(7, 14)                     # бейзлайн флагует часть hard-ходов
        hard_hits = hard_flags * rng.uniform(0.5, 0.75)
        call_flags = rng.uniform(15, 45)                    # и часть остальных ходов с вызовами
        call_hits = call_flags * rng.uniform(0.15, 0.35)
        text_at_05 = (254 - hard_flags - call_flags, 94 - hard_hits - call_hits)
        old_hard_flags = rng.uniform(13, 18)                # старые hard-правила в v3.0
        text_at_08 = (86 - old_hard_flags, 42 - old_hard_flags * rng.uniform(0.5, 0.65))
        scenarios.append((rng.choice(grid, p=posterior), text_separation([text_at_05, text_at_08])))
    return scenarios


def predict(scenarios: list, call_flags: int, text_flags: int, text_precision_gain: float = 1.0) -> tuple:
    """(10-й перцентиль, медиана, 90-й перцентиль) F1 для hard + топ вызовов + топ текста."""
    scores = []
    for call_separation, text_sep in scenarios:
        hits = HARD_ERRORS
        if call_flags:
            hits += true_positives(call_separation, CALL_ERRORS, CALL_TURNS - CALL_ERRORS, call_flags)
        if text_flags:
            text_hits = true_positives(text_sep, TEXT_ERRORS, TEXT_TURNS - TEXT_ERRORS, text_flags)
            hits += min(TEXT_ERRORS, text_hits * text_precision_gain)
        flags = HARD_FLAGS + call_flags + text_flags
        scores.append(2 * hits / (POSITIVES + flags))
    return tuple(np.percentile(scores, [10, 50, 90]))


def main() -> None:
    scenarios = sample_scenarios()
    variants = [
        ("hard + вызовы топ-83 (= v4-C, сверка)", 83, 0, 1.0),
        ("hard + все текстовые ходы (= зонд №7, сверка)", 0, TEXT_TURNS, 1.0),
        ("hard + вызовы топ-40 + текст P_base топ-70", 40, 70, 1.0),
        ("hard + вызовы топ-40 + текст P_base топ-130", 40, 130, 1.0),
        ("hard + вызовы топ-40 + текст P_base топ-210", 40, 210, 1.0),
        ("hard + вызовы топ-40 + текст топ-130, судья ×1.15", 40, 130, 1.15),
        ("hard + вызовы топ-40 + текст топ-130, судья ×1.3", 40, 130, 1.3),
    ]
    print(f"{'вариант':52s} флагов  F1: 10% / медиана / 90%")
    for name, call_flags, text_flags, gain in variants:
        low, median, high = predict(scenarios, call_flags, text_flags, gain)
        print(f"{name:52s} {HARD_FLAGS + call_flags + text_flags:5d}  {low:.3f} / {median:.3f} / {high:.3f}")


if __name__ == "__main__":
    main()
