"""Симулятор публичного лидерборда: качество на пуле (378 ходов, labeling/labels_final.jsonl) -> ожидаемый F1 паблика.

Две калибровочные точки (TPR/FPR на пуле -> TP/FP на паблике, 130 позитивов):
  v1.1 (сабмит №1)        пул 0.736 / 0.098  -> паблик 63 TP / 68 FP
  бейзлайн организаторов  пул 0.944 / 0.241  -> паблик 94 TP / 160 FP   (прогон промпта бейзлайна на пуле, ветка A 05.10)

Модель "mix" (по умолчанию): доля ALPHA позитивов паблика похожа на позитивы пула и ловится с TPR пула,
остальные (1 - ALPHA) для наших судей неотличимы от негативов и ловятся с FPR пула (вероятно, ошибки,
видимые только с метаданными задачи, которые были у разметчика):
    TPR_pub = ALPHA * TPR_pool + (1 - ALPHA) * FPR_pool,   FP_pub = FPR_pool * N_NEG.
МНК по двум точкам: ALPHA 0.649, N_NEG 668 (паблик ≈ 800 примеров). Ошибка на точках ±0.025 F1.
Модель "ratio" (старая, только по v1.1): TPR_pub = TPR_pool * 0.658 — на бейзлайне ошибалась на −0.06.
Метки пула и константы должны быть из одного набора: при смене меток пересчитать (tools/label_agreement.py)."""
from __future__ import annotations

import numpy as np

POS = 130
CAL = {"tpr_pool": 0.7361, "fpr_pool": 0.0980, "tp_pub": 63, "fp_pub": 68}   # точка v1.1 (модель ratio)
ALPHA, N_NEG = 0.649, 668                                                   # модель mix (две точки)


def public_from_rates(tpr: float, fpr: float, model: str = "mix") -> dict:
    if model == "ratio":
        tp = POS * min(1.0, tpr * (CAL["tp_pub"] / POS) / CAL["tpr_pool"])
        fp = fpr * CAL["fp_pub"] / CAL["fpr_pool"]
    else:
        tp = POS * min(1.0, ALPHA * tpr + (1 - ALPHA) * fpr)
        fp = fpr * N_NEG
    p = tp / max(1e-9, tp + fp)
    r = tp / POS
    return {"f1": 2 * p * r / max(1e-9, p + r), "p": p, "r": r, "tp": tp, "fp": fp}


def public_from_preds(y, pred, model: str = "mix") -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    fpr = ((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
    out = public_from_rates(float(tpr), float(fpr), model)
    out.update({"tpr_pool": float(tpr), "fpr_pool": float(fpr)})
    return out
