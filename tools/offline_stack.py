"""Офлайн-оценка стекинга на признаках v1 (labeling/dump_v1feats.parquet, без GPU) с итоговыми метками
(labeling/labels_final.jsonl + эталон valid) и симулятором паблика (lbsim.CAL).
Запуск: python tools/offline_stack.py"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution" / "src"))
from guardian.devtools import attach_labels, build_pool, evaluate_sets  # noqa: E402

valid = pd.read_parquet(ROOT / "valid.parquet")
pool = build_pool(valid)
lab = {json.loads(x)["id"]: int(json.loads(x)["label"])
       for x in open(ROOT / "labeling/labels_final.jsonl", encoding="utf-8") if x.strip()}
lab.update(dict(zip(valid["id"], valid["label"].astype(int))))
dump = attach_labels(pd.read_parquet(ROOT / "labeling/dump_v1feats.parquet").drop(columns=["label"]), lab)
print(f"меток: {len(dump)} (пул {dump.id.isin(set(pool.id)).sum()}), hard {int(dump.hard.sum())}")

STRUCT = ["ro_only", "has_text", "has_write", "transfer", "claimed_action_no_call", "repeat_call",
          "missing_numbers", "ungrounded_values"]
DOM = ["dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
SETS = {
    "la + структура": ["la"] + STRUCT,
    "la + структура + домен": ["la"] + STRUCT + DOM,
    "la + B + структура": ["la", "b_vfrac", "b_mean", "b_missing"] + STRUCT,
    "la + B + структура + домен": ["la", "b_vfrac", "b_mean", "b_missing"] + STRUCT + DOM,
}
res = evaluate_sets(dump, SETS, pool.id, seeds=(0, 1, 2, 3, 4))
pd.set_option("display.width", 200)
print(res.round(3).to_string())
