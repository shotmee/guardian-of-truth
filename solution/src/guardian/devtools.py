"""Инструменты разработки (используются только в ноутбуке, не на платформе):
пул промежуточных ходов, метрики, подбор порога с поправкой на долю позитивов в паблике."""
from __future__ import annotations

import itertools
import json
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .transcript import HEADER_RE, RESP_PREFIX


def _strip_results(block: str) -> str:
    """Убирает строки ← TOOL_RESPONSE (и их продолжения) — в response результатов нет."""
    out, in_res = [], False
    for ln in block.split("\n"):
        s = ln.lstrip("\t ")
        if s.startswith(RESP_PREFIX):
            in_res = True
            continue
        if in_res and ln.startswith("\t") and not s.startswith("→"):
            continue
        in_res = False
        out.append(ln)
    return "\n".join(out).rstrip("\n")


def build_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Из каждого prompt вырезает все предыдущие ходы агента как отдельные примеры (без меток).
    Тот же агент, те же домены и длины — хороший прокси теста для замеров скорости и доли позитивов."""
    seen, rows = set(), []
    for r in df.itertuples():
        traj = str(r.id).split("::")[0]
        p = r.prompt
        heads = list(HEADER_RE.finditer(p))
        for k, m in enumerate(heads):
            if m.group(1) != "ASSISTANT" or not m.group(2):
                continue
            turn = int(m.group(2))
            if (traj, turn) in seen:
                continue
            seen.add((traj, turn))
            end = heads[k + 1].start() if k + 1 < len(heads) else len(p)
            resp = _strip_results(p[m.start():end].rstrip("\n"))
            rows.append({"id": f"{traj}::t{turn}", "prompt": p[:m.start()].rstrip("\n"), "response": resp,
                         "src_id": r.id, "turn": turn})
    return pd.DataFrame(rows)


def prf(y: np.ndarray, p: np.ndarray, w_neg: float = 1.0) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tp = float(((p == 1) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum()) * w_neg
    fn = float(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def neg_weight(y: np.ndarray, pi: float) -> float:
    """Вес негатива, чтобы доля позитивов стала pi (valid сбалансирован 50/50, паблик — нет)."""
    y = np.asarray(y)
    n_pos, n_neg = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
    return (1 - pi) / pi * n_pos / n_neg


def fuse(pa: np.ndarray, pb: np.ndarray, hard: np.ndarray, w: float) -> np.ndarray:
    pb = np.where(np.isnan(pb), pa, pb)
    s = w * pb + (1 - w) * pa
    s = np.where(np.isnan(s), 0.0, s)
    return np.where(hard, 1.0, s)


def tune(dump: pd.DataFrame, pis: Iterable[float] = (0.2, 0.25, 0.3, 0.35),
         ws: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
         ts: Optional[List[float]] = None) -> pd.DataFrame:
    """Сетка (w, T): средний F1 по сценариям доли позитивов pi + сглаживание по соседним порогам."""
    ts = ts or [round(x, 2) for x in np.arange(0.15, 0.86, 0.05)]
    y = dump["label"].values
    pa = dump["p_nothink"].astype(float).values
    pb = dump["p_think"].astype(float).values
    hard = dump["hard"].astype(bool).values
    rows = []
    for w, t in itertools.product(ws, ts):
        s = fuse(pa, pb, hard, w)
        pred = (s >= t).astype(int)
        f_pis = [prf(y, pred, neg_weight(y, pi))["f1"] for pi in pis]
        raw = prf(y, pred)
        rows.append({"w": w, "T": t, "f1_shifted": float(np.mean(f_pis)), "f1_valid": raw["f1"],
                     "precision_valid": raw["precision"], "recall_valid": raw["recall"]})
    g = pd.DataFrame(rows)
    # сглаживание: среднее по соседним T (±0.05) — защищает от «острого» оптимума на 46 примерах
    g["f1_smooth"] = g.groupby("w")["f1_shifted"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    return g.sort_values(["f1_smooth", "f1_shifted"], ascending=False).reset_index(drop=True)


def replay_labels(dump: pd.DataFrame, cfg: Dict, cap: bool = True) -> np.ndarray:
    """Повторяет решение State.labels() по дампу (без GPU). В gate-режиме рассуждение учитывается
    только у примеров с P_A >= b_min_pa — как на платформе."""
    import json as _json
    from .pipeline import State
    c = dict(cfg) if cap else dict(cfg, max_pos_frac=None)
    st = State(dump["id"].astype(str).tolist(), c)
    b_min = c.get("b_min_pa")
    for i, r in enumerate(dump.itertuples()):
        st.hard[i] = bool(r.hard)
        st.pa[i] = None if pd.isna(r.p_nothink) else float(r.p_nothink)
        s = _json.loads(r.p_think_samples) if isinstance(r.p_think_samples, str) else []
        if b_min is None or st.hard[i] or (st.pa[i] is not None and st.pa[i] >= b_min):
            st.pb[i] = s
    return np.array(st.labels())


def tune_gate(valid_dump: pd.DataFrame, pool_dump: Optional[pd.DataFrame], cfg: Dict,
              pis: Iterable[float] = (0.12, 0.15, 0.2), max_pool_rate: float = 0.25,
              grid: Iterable[float] = (0.98, 0.99, 0.995, 0.997, 0.998, 0.999)) -> pd.DataFrame:
    """Подбор gate_pa: F1 при сдвиге доли позитивов (valid) + доля флагов на пуле (тестоподобные ходы)."""
    y = valid_dump["label"].values
    rows = []
    for g in grid:
        c = dict(cfg, decision="gate", gate_pa=g, b_min_pa=g)
        pred = replay_labels(valid_dump, c, cap=False)
        f = [prf(y, pred, neg_weight(y, pi))["f1"] for pi in pis]
        raw = prf(y, pred)
        rate = float(replay_labels(pool_dump, c, cap=False).mean()) if pool_dump is not None else float("nan")
        rows.append({"gate_pa": g, "f1_shifted": float(np.mean(f)), "tp": raw["tp"], "fp": raw["fp"],
                     "f1_valid": raw["f1"], "pool_flag_rate": rate})
    g = pd.DataFrame(rows)
    g["f1_smooth"] = g["f1_shifted"].rolling(3, center=True, min_periods=1).mean()
    g["ok"] = g["pool_flag_rate"].fillna(0) <= max_pool_rate
    return g.sort_values(["ok", "f1_smooth"], ascending=False).reset_index(drop=True)


def fit_stack(dump: pd.DataFrame, features_list: Optional[List[str]] = None, C: float = 0.5, folds: int = 5,
              pis: Iterable[float] = (0.15, 0.2), seed: int = 0) -> Dict:
    """Стекинг (логрег) на признаках из дампа (колонка feats=json) для не-hard примеров с метками.
    CV по траекториям -> OOF-метрики и порог; финальная модель — на всех данных."""
    import json as _json
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from .features import FEATURES
    names = list(features_list or FEATURES)
    d = dump[(~dump["hard"].astype(bool)) & dump["label"].notna()].reset_index(drop=True)
    F = pd.DataFrame([_json.loads(x) for x in d["feats"]])
    X = F[names].values.astype(float)
    y = d["label"].astype(int).values
    traj = d["id"].astype(str).str.split("::").str[0].values
    rng = np.random.RandomState(seed)
    ut = rng.permutation(np.unique(traj))
    fold = {t: k % folds for k, t in enumerate(ut)}
    oof = np.zeros(len(d))
    for k in range(folds):
        te = np.array([fold[t] == k for t in traj])
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=5000).fit((X[~te] - mu) / sd, y[~te])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    # hard-примеры = 1; метрики считаем на всём наборе (hard + не-hard)
    hard_all = dump[dump["label"].notna()]
    y_h = hard_all[hard_all["hard"].astype(bool)]["label"].astype(int).values
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        pred = (oof >= t).astype(int)
        yy = np.r_[y, y_h]
        pp = np.r_[pred, np.ones(len(y_h), int)]
        r = {"t": t, "f1": prf(yy, pp)["f1"]}
        for pi in pis:
            r[f"f1_pi{pi}"] = prf(yy, pp, neg_weight(yy, pi))["f1"]
        r["flag_rate"] = pp.mean()
        rows.append(r)
    g = pd.DataFrame(rows)
    g["f1_mix"] = g[[f"f1_pi{pi}" for pi in pis]].mean(1)
    g["f1_mix_s"] = g["f1_mix"].rolling(3, center=True, min_periods=1).mean()
    best = g.sort_values("f1_mix_s", ascending=False).iloc[0]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
    stack = {"features": names, "mu": mu.round(6).tolist(), "sd": sd.round(6).tolist(),
             "coef": clf.coef_[0].round(6).tolist(), "intercept": float(round(clf.intercept_[0], 6)),
             "threshold": float(best["t"])}
    return {"stack": stack, "oof_auc": float(roc_auc_score(y, oof)), "grid": g, "best": best.to_dict(),
            "oof": oof, "ids": d["id"].tolist()}


def attach_labels(dump: pd.DataFrame, labels: Dict[str, int]) -> pd.DataFrame:
    d = dump.copy()
    d["label"] = d["id"].map(labels)
    return d[d["label"].notna()].reset_index(drop=True)


def evaluate_sets(dump: pd.DataFrame, sets: Dict[str, List[str]], pool_ids: Iterable[str],
                  seeds: Iterable[int] = (0, 1, 2), C: float = 0.5) -> pd.DataFrame:
    """Для каждого набора признаков: OOF AUC (не-hard), F1 на пуле и ожидаемый F1 паблика (lbsim)
    при пороге, выбранном fit_stack, и при лучшем по симулятору пороге (оптимистично)."""
    from .lbsim import public_from_preds
    pool_ids = set(pool_ids)
    rows = []
    for name, feats in sets.items():
        aucs, sims, sims_best, f1p, thr = [], [], [], [], []
        for sd in seeds:
            res = fit_stack(dump, features_list=feats, C=C, seed=sd)
            oof = dict(zip(res["ids"], res["oof"]))
            P = dump[dump["id"].isin(pool_ids)]
            s = np.array([1.0 if h else oof.get(i, 0.0) for i, h in zip(P["id"], P["hard"].astype(bool))])
            y = P["label"].astype(int).values
            t = res["stack"]["threshold"]
            sims.append(public_from_preds(y, s >= t)["f1"])
            sims_best.append(max(public_from_preds(y, s >= tt)["f1"] for tt in np.arange(0.1, 0.91, 0.05)))
            f1p.append(prf(y, (s >= t).astype(int))["f1"])
            aucs.append(res["oof_auc"])
            thr.append(t)
        rows.append({"набор": name, "n_feats": len(feats), "oof_auc": np.mean(aucs), "f1_pool": np.mean(f1p),
                     "паблик_оценка": np.mean(sims), "паблик_лучший_порог": np.mean(sims_best), "порог": np.mean(thr)})
    return pd.DataFrame(rows).sort_values("паблик_оценка", ascending=False).reset_index(drop=True)


def estimate_public_prevalence(baseline_fpr_valid: float, public_fp: int = 160, public_pos: int = 130) -> Dict[str, float]:
    """По бейзлайну: на паблике у него FP=160 (TP=94 из 130). FPR на valid-негативах -> N_neg паблика."""
    fpr = max(1e-3, baseline_fpr_valid)
    n_neg = public_fp / fpr
    return {"n_neg_public": n_neg, "n_public": n_neg + public_pos, "pi_public": public_pos / (n_neg + public_pos)}


BASELINE_PROMPT_NOTE = "Системный промпт бейзлайна лежит рядом: baseline_prompt.md (для реплики бейзлайна в ноутбуке)."
