"""Оценка сигналов и правил решения на размеченном пуле (метки агентов Claude) + valid (эталон организаторов).
Запуск: python tools/eval_pool.py [--dump colab_results/v1/results/pool_judge.parquet]"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution" / "src"))
from guardian.devtools import build_pool  # noqa: E402
from guardian.rules import analyze  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default=str(ROOT / "colab_results/v1/results/pool_judge.parquet"))
ap.add_argument("--vdump", default=str(ROOT / "colab_results/v1/results/valid_judge.parquet"))
a = ap.parse_args()

# ---- метки
lab = []
_final = ROOT / "labeling/labels_final.jsonl"
for f in [str(_final)] if _final.exists() else sorted(glob.glob(str(ROOT / "labeling/out/*.jsonl"))):
    for ln in open(f, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                lab.append(json.loads(ln))
            except json.JSONDecodeError:
                print("битая строка в", f, ln[:80])
L = pd.DataFrame(lab).drop_duplicates("id", keep="last")
print(f"меток от агентов: {len(L)} | доля 1: {L.label.mean():.3f}")
print("категории позитивов:", L[L.label == 1].cat.value_counts().to_dict())

valid = pd.read_parquet(ROOT / "valid.parquet")
pool = build_pool(valid)
P = pd.read_parquet(a.dump).merge(L[["id", "label", "cat", "why"]], on="id", how="inner")
V = pd.read_parquet(a.vdump)
V["cat"], V["why"] = "gold", ""
print(f"пул с метками: {len(P)} из {len(pool)}")


def feats(d: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    m = src.set_index("id")
    fl = []
    for i in d.id:
        r = m.loc[i]
        an = analyze(i, r.prompt, r.response)
        fl.append({**an["flags"], "hard2": int(bool(an["hard"]))})
    F = pd.DataFrame(fl, index=d.index)
    d = pd.concat([d, F], axis=1)
    d["pa"] = d.p_nothink.fillna(0.5).clip(1e-6, 1 - 1e-6)
    d["la"] = np.log(d.pa / (1 - d.pa))
    s = d.p_think_samples.apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    d["nv"] = s.apply(len)
    d["votes"] = s.apply(lambda v: sum(x >= 0.5 for x in v))
    d["vfrac"] = np.where(d.nv > 0, d.votes / d.nv.clip(lower=1), np.nan)
    d["Bany"] = d.votes >= 1
    d["dom"] = d.id.str.split("__").str[0]
    return d


P = feats(P, pool)
V = feats(V, valid)


def prf(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc, tp, fp


for name, d in (("ПУЛ (Claude)", P), ("VALID (эталон)", V)):
    y = d.label.values
    print(f"\n===== {name}: n={len(d)}, позитивов {int(y.sum())} ({y.mean():.2f})")
    hp = d.hard2 == 1
    print(f"hard-правила: срабатываний {int(hp.sum())}, из них позитивов {int(d.label[hp].sum())}")
    for c in ("la", "vfrac"):
        mm = d[c].notna()
        if mm.sum() > 5 and d.label[mm].nunique() == 2:
            print(f"AUC {c}: {roc_auc_score(d.label[mm], d[c][mm]):.3f} (n={int(mm.sum())})")
    rules = {
        "hard only": d.hard2 == 1,
        "pA>=0.5 | hard (бейзлайн-подобно)": (d.pa >= 0.5) | (d.hard2 == 1),
        "B any | hard": d.Bany | (d.hard2 == 1),
        "gate .997 (v1.1)": (d.hard2 == 1) | (d.Bany & (d.pa >= 0.997)),
        "gate .99": (d.hard2 == 1) | (d.Bany & (d.pa >= 0.99)),
        "gate .999": (d.hard2 == 1) | (d.Bany & (d.pa >= 0.999)),
        "B all | hard": ((d.nv > 0) & (d.votes == d.nv)) | (d.hard2 == 1),
    }
    for t in (0.99, 0.997, 0.999, 0.9995):
        rules[f"pA>={t} | hard"] = (d.pa >= t) | (d.hard2 == 1)
    print(f"{'правило':36s}  F1    P     R    TP  FP  доля")
    for k, r in rules.items():
        f, pr, rc, tp, fp = prf(y, r.values)
        print(f"{k:36s} {f:.3f} {pr:.3f} {rc:.3f} {tp:3d} {fp:3d}  {r.mean():.2f}")
    print("позитивы по доменам:", d.groupby("dom").label.agg(["size", "sum"]).T.to_dict())

# ---- стекинг: CV по траекториям на пуле, проверка на valid
X_cols = ["la", "vfrac", "hard2", "transfer", "claimed_action_no_call", "missing_numbers", "ungrounded_values",
          "repeat_call", "n_calls", "has_text", "n_writes"]


def X(d):
    x = d[X_cols].copy().astype(float)
    x["vfrac"] = x["vfrac"].fillna(-1)
    x["missing_numbers"] = np.log1p(x["missing_numbers"])
    x["ungrounded_values"] = np.log1p(x["ungrounded_values"])
    return x.values


P["traj"] = P.id.str.split("::").str[0]
trajs = P.traj.unique()
rng = np.random.RandomState(0)
folds = {t: k % 5 for k, t in enumerate(rng.permutation(trajs))}
oof = np.zeros(len(P))
for k in range(5):
    tr = P.traj.map(folds) != k
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight=None).fit(X(P[tr]), P.label[tr])
    oof[~tr.values] = clf.predict_proba(X(P[~tr]))[:, 1]
print(f"\n===== СТЕКИНГ (логрег по {len(X_cols)} признакам), CV по траекториям на пуле: AUC {roc_auc_score(P.label, oof):.3f}")
for q in (0.10, 0.15, 0.20, 0.25):
    thr = np.quantile(oof, 1 - q)
    f, pr, rc, tp, fp = prf(P.label, oof >= thr)
    print(f"  флаг верхних {q:.0%}: F1 {f:.3f} P {pr:.3f} R {rc:.3f}")
clf = LogisticRegression(C=0.5, max_iter=2000).fit(X(P), P.label)
pv = clf.predict_proba(X(V))[:, 1]
print(f"  обучен на всём пуле -> valid AUC {roc_auc_score(V.label, pv):.3f}")
print("  веса:", dict(zip(X_cols, np.round(clf.coef_[0], 2))), "bias", round(clf.intercept_[0], 2))
P.assign(oof=oof).to_parquet(ROOT / "labeling/pool_eval.parquet", index=False)
