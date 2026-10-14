"""Единые признаки для стекинга: одинаково считаются при обучении (по дампам Colab) и на платформе."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

FEATURES = ["la", "la_base", "la_v1", "b_vfrac", "b_mean", "b_missing", "s_max", "s_mean", "s_missing",
            "hard", "ro_only", "has_text", "has_write", "transfer", "claimed_action_no_call", "repeat_call",
            "missing_numbers", "ungrounded_values",
            "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]


def _logit(p: Optional[float]) -> float:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        p = 0.5
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def make(analysis: Dict[str, Any], pa: Optional[float], pb: List[float], ps: List[float],
         extra: Optional[Dict[str, Optional[float]]] = None) -> Dict[str, float]:
    fl = analysis.get("flags", {}) or {}
    dom = analysis.get("domain", "unknown")
    extra = extra or {}
    f = {
        "la": _logit(pa),
        "la_base": _logit(extra.get("base")),
        "la_v1": _logit(extra.get("v1")),
        "b_vfrac": float(np.mean([x >= 0.5 for x in pb])) if pb else -1.0,
        "b_mean": float(np.mean(pb)) if pb else -1.0,
        "b_missing": 0.0 if pb else 1.0,
        "s_max": float(max(ps)) if ps else -1.0,
        "s_mean": float(np.mean(ps)) if ps else -1.0,
        "s_missing": 0.0 if ps else 1.0,
        "hard": float(bool(analysis.get("hard"))),
        "ro_only": float(fl.get("ro_only", 0)),
        "has_text": float(fl.get("has_text", 0)),
        "has_write": float(fl.get("n_writes", 0) > 0),
        "transfer": float(fl.get("transfer", 0)),
        "claimed_action_no_call": float(fl.get("claimed_action_no_call", 0)),
        "repeat_call": float(fl.get("repeat_call", 0)),
        "missing_numbers": float(fl.get("missing_numbers", 0) > 0),
        "ungrounded_values": float(fl.get("ungrounded_values", 0) > 0),
    }
    for d in ("airline", "banking_knowledge", "retail", "telecom"):
        f[f"dom_{d}"] = float(dom == d)
    return f


def score(f: Dict[str, float], stack: Dict[str, Any]) -> float:
    """Логистическая регрессия на стандартизованных признаках; параметры — из config['stack']."""
    names = stack["features"]
    x = np.array([f.get(n, 0.0) for n in names], dtype=float)
    z = (x - np.array(stack["mu"])) / np.array(stack["sd"])
    s = float(np.dot(z, np.array(stack["coef"])) + stack["intercept"])
    return 1.0 / (1.0 + math.exp(-s))
