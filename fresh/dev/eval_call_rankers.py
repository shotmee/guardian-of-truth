"""Ходы с вызовами на размеченном новом наборе τ² (labeling_v6): точность hard-проверок и качество ранжировщиков
мягких ходов (атомы C, P_base, судьи v1/v2) — AUC и точность на топ-доле.

python dev/eval_call_rankers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "dev"))
from eval_text_rankers import SETS, load_labels, weighted_auc  # noqa: E402


def main() -> None:
    paths = SETS["tau2"]
    units = pd.DataFrame(json.loads(paths["units"].read_text(encoding="utf-8")))
    labels = load_labels(paths["labels"])
    frame = units[units.kind != "text"].merge(labels, on="id").merge(pd.read_parquet(paths["signals"]), on="id",
                                                                       how="left", suffixes=("", "_signal"))
    frame["label"] = frame.label.astype(int)
    hard = frame[frame.kind == "hard"]
    print(f"hard-срабатываний размечено {len(hard)}, ошибок {int(hard.label.sum())} "
          f"(точность {hard.label.mean():.2f}; на паблике 0.61)")
    soft = frame[frame.kind == "call"].reset_index(drop=True)
    print(f"мягких ходов с вызовами {len(soft)}, ошибок {int(soft.label.sum())} (база {soft.label.mean():.2f}; "
          f"на паблике 0.155)")
    flat = np.ones(len(soft))
    for column in ("calls_score", "p_base", "p_v1", "p_v2"):
        scores = soft[column].fillna(soft[column].median()).values + 1e-9 * np.arange(len(soft)) / len(soft)
        order = np.argsort(-scores)
        top = {share: soft.label.values[order[: max(1, int(round(share * len(soft))))]].mean()
               for share in (0.2, 0.3, 0.5)}
        print(f"  {column:12s} AUC {weighted_auc(scores, soft.label.values, flat):.3f}   точность топ-20% {top[0.2]:.2f}, "
              f"топ-30% {top[0.3]:.2f}, топ-50% {top[0.5]:.2f}")


if __name__ == "__main__":
    main()
