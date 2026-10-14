"""Сводка dev-прогона v4: метрики на valid (эталон организаторов), согласие стратегий, время.

python dev/analyze.py --dev <dev.parquet> --dumps <dir с dump_*.json> [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(np.array(a)), rankdata(np.array(b)))[0, 1])


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = pd.read_parquet(a.dev)
    lab = dict(zip(dev["id"], dev["label"]))
    valid_ids = set(dev.loc[dev["src"] == "valid", "id"])
    summary = {}
    dumps = {}
    for p in sorted(Path(a.dumps).glob("dump_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d["strategy"]
        dumps[s] = d
        flags = set(d["flags"])
        hard = set(d["hard"])
        scores = {k: v["score"] for k, v in d["scores"].items()}
        conseq = d["conseq"]
        tp = sum(1 for i in valid_ids if i in flags and lab[i] == 1)
        fp = sum(1 for i in valid_ids if i in flags and lab[i] == 0)
        fn = sum(1 for i in valid_ids if i not in flags and lab[i] == 1)
        # valid-позитивы, которые не поймали проверки: попали ли они в ранжированные флаги
        soft_pos = [i for i in valid_ids if lab[i] == 1 and i in conseq and i not in hard]
        soft_hit = sum(1 for i in soft_pos if i in flags)
        thr = min((scores[i] for i in flags if i in scores), default=None)
        pool_flags = [i for i in flags if i not in valid_ids]
        summary[s] = dict(
            seconds=round(d.get("seconds", 0)), n=d["n"], flags=len(flags), hard=len(hard),
            scored=len(scores), valid_tp=tp, valid_fp=fp, valid_fn=fn,
            valid_soft_pos=len(soft_pos), valid_soft_hit=soft_hit,
            pool_flags=len(pool_flags),
            pool_flags_conseq=sum(1 for i in pool_flags if conseq.get(i)),
            score_mean_conseq=float(np.mean([v for k, v in scores.items() if conseq.get(k)] or [np.nan])),
            score_mean_read=float(np.mean([v for k, v in scores.items() if not conseq.get(k)] or [np.nan])),
            min_flag_score=thr,
        )
    names = sorted(dumps)
    pair = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            fx = set(dumps[x]["flags"]) - set(dumps[x]["hard"])
            fy = set(dumps[y]["flags"]) - set(dumps[y]["hard"])
            sx = {k: v["score"] for k, v in dumps[x]["scores"].items()}
            sy = {k: v["score"] for k, v in dumps[y]["scores"].items()}
            common = sorted(set(sx) & set(sy))
            pair[f"{x}~{y}"] = dict(
                jaccard_soft_flags=round(len(fx & fy) / max(1, len(fx | fy)), 3),
                spearman=round(spearman([sx[k] for k in common], [sy[k] for k in common]), 3),
            )
    out = dict(strategies=summary, pairs=pair)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
