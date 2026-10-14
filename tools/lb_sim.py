"""Симулятор публичного лидерборда.

Идея: пул (384 естественных хода, метки Claude, 95% согласия с эталоном на valid) — ближайший к паблику набор,
но паблик «труднее»: у v1.1 recall на пуле 0.70, а на паблике 0.485. Калибруем переход пул -> паблик по
реальному сабмиту v1.1 (и бейзлайну, когда будет его прогон на пуле):
    TPR_pub = TPR_pool * k_tpr,   FPR_pub = FPR_pool * k_fpr,
    k_tpr = 0.485 / TPR_pool(v1.1),  k_fpr = (68 / N_neg) / FPR_pool(v1.1).
Неизвестный размер паблика N перебираем (650..1000; позитивов ровно 130).
Ограничение: один калибровочный сабмит, коэффициенты могут зависеть от типа системы — оценка грубая,
но лучше ранжирует варианты, чем valid (на valid v1.1 давал 0.86)."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

PUBLIC = {"v1.1": {"tp": 63, "fp": 68}, "baseline": {"tp": 94, "fp": 160}}
POS = 130


def rates(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    fpr = ((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
    return float(tpr), float(fpr)


class Sim:
    def __init__(self, calib_tpr: float, calib_fpr: float, ref: str = "v1.1"):
        self.ct, self.cf, self.ref = calib_tpr, calib_fpr, ref

    def public(self, tpr: float, fpr: float, N: int) -> Dict[str, float]:
        nneg = N - POS
        k_tpr = (PUBLIC[self.ref]["tp"] / POS) / max(1e-9, self.ct)
        k_fpr = (PUBLIC[self.ref]["fp"] / nneg) / max(1e-9, self.cf)
        tp = POS * min(1.0, tpr * k_tpr)
        fp = nneg * min(1.0, fpr * k_fpr)
        p = tp / max(1e-9, tp + fp)
        r = tp / POS
        return {"f1": 2 * p * r / max(1e-9, p + r), "p": p, "r": r, "tp": tp, "fp": fp}

    def table(self, y, pred, Ns: Iterable[int] = (650, 800, 1000)) -> Dict[str, float]:
        tpr, fpr = rates(y, pred)
        out = {"tpr_pool": tpr, "fpr_pool": fpr}
        for N in Ns:
            out[f"f1@N{N}"] = self.public(tpr, fpr, N)["f1"]
        return out
