"""Лаборатория конфигураций v2 по дампам ветки A (colab_results/v2_main): какие фазы оставить, чтобы уложиться в 30 мин,
и сколько это стоит в ожидаемом F1 паблика. Без GPU.
  - признаки: из feats дампа + B по сэмплам (p_think_samples) + длины (resp/prompt chars, n_calls, turn);
  - B на части кандидатов (как при дедлайне: порядок pa_desc) — b_* = -1, b_missing = 1 для остальных;
  - стекинг: логрег, CV по траекториям (5 фолдов × 5 сидов), hard = 1; порог — лучший по симулятору (mix) на OOF;
  - время на N=800 по фазам e2e-прогона (385 примеров, A100 80GB).
Запуск: python tools/v2_lab.py"""
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
from guardian.lbsim import public_from_preds  # noqa: E402
from guardian.rules import analyze  # noqa: E402

R = ROOT / "colab_results/v2_main/results"
# секунды на 385 примеров пула в e2e-прогоне (лог e2e.log) и старт vLLM
PHASE_S = {"A": 94, "A_base": 58, "A_v1": 71, "S": 70, "B1": 482, "B2": 365}
LOAD_S, N_TEST = 258, 800


def load():
    valid = pd.read_parquet(ROOT / "valid.parquet")
    pool = build_pool(valid)
    lab = {json.loads(x)["id"]: int(json.loads(x)["label"])
           for x in open(ROOT / "labeling/labels_final.jsonl", encoding="utf-8") if x.strip()}
    lab.update(dict(zip(valid["id"], valid["label"].astype(int))))
    D = pd.concat([pd.read_parquet(R / "v2_pool.parquet"), pd.read_parquet(R / "v2_valid.parquet").drop(columns=["label"])])
    D = D.drop_duplicates("id").reset_index(drop=True)
    D["label"] = D["id"].map(lab)
    D = D[D["label"].notna()].reset_index(drop=True)
    D["label"] = D["label"].astype(int)
    src = pd.concat([pool[["id", "prompt", "response"]], valid[["id", "prompt", "response"]]]).drop_duplicates("id").set_index("id")
    F = pd.DataFrame([json.loads(x) for x in D["feats"]])
    fl = [analyze(i, src.at[i, "prompt"], src.at[i, "response"])["flags"] for i in D["id"]]
    F["log_resp"] = np.log1p([f["resp_chars"] for f in fl])
    F["log_prompt"] = np.log1p([f["prompt_chars"] for f in fl])
    F["n_calls"] = [f["n_calls"] for f in fl]
    F["turn"] = np.log1p([f["turn"] for f in fl])
    smp = [json.loads(s) if isinstance(s, str) else list(s) for s in D["p_think_samples"]]
    F["b1_p"] = [s[0] if len(s) else -1.0 for s in smp]
    F["b1_vote"] = [float(s[0] >= 0.5) if len(s) else -1.0 for s in smp]
    D["in_pool"] = D["id"].isin(set(pool["id"]))
    return D, F


def b_subset(F, frac, key="la"):
    """Оставить B только у доли frac кандидатов с наибольшим key (как pa_desc при нехватке времени)."""
    G = F.copy()
    cand = np.where(F["b_missing"].values == 0)[0]
    if frac < 1 and len(cand):
        order = cand[np.argsort(-F[key].values[cand])]
        drop = order[int(round(frac * len(cand))):]
        for c in ("b_vfrac", "b_mean", "b1_p", "b1_vote"):
            G.loc[drop, c] = -1.0
        G.loc[drop, "b_missing"] = 1.0
    return G


def oof(D, F, cols, seed, C=0.5):
    hard = D["hard"].astype(bool).values
    y = D["label"].values
    traj = D["id"].str.split("::").str[0].values
    rng = np.random.RandomState(seed)
    ut = rng.permutation(np.unique(traj))
    fold = np.array([{t: k % 5 for k, t in enumerate(ut)}[t] for t in traj])
    X = F[cols].values.astype(float)
    o = np.zeros(len(y))
    for k in range(5):
        te, tr = fold == k, (fold != k) & ~hard
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        o[te] = LogisticRegression(C=C, max_iter=5000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return np.where(hard, 1.0, o)


def evaluate(D, F, cols, seeds=(0, 1, 2, 3, 4)):
    pm = D["in_pool"].values
    y = D["label"].values
    hard = D["hard"].astype(bool).values
    S = [oof(D, F, cols, s) for s in seeds]
    grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
    out = {"auc": np.mean([roc_auc_score(y[~hard], s[~hard]) for s in S])}
    for model in ("mix", "ratio"):
        f1 = np.array([[public_from_preds(y[pm], s[pm] >= t, model)["f1"] for t in grid] for s in S]).mean(0)
        f1s = pd.Series(f1).rolling(3, center=True, min_periods=1).mean().values   # сглаживание по порогу
        k = int(np.argmax(f1s))
        out[f"{model}"] = f1[k]
        out[f"t_{model}"] = grid[k]
        if model == "mix":
            r = [public_from_preds(y[pm], s[pm] >= grid[k], model) for s in S]
            out.update({"tpr_pool": np.mean([x["tpr_pool"] for x in r]), "fpr_pool": np.mean([x["fpr_pool"] for x in r]),
                        "TP": np.mean([x["tp"] for x in r]), "FP": np.mean([x["fp"] for x in r])})
    return out


def minutes(phases, b_frac=1.0):
    s = sum(PHASE_S[p] * (b_frac if p.startswith("B") else 1.0) for p in phases)
    return (LOAD_S + s * N_TEST / 385) / 60


if __name__ == "__main__":
    D, F = load()
    print(f"примеров {len(D)} (пул {D.in_pool.sum()}), hard {int(D.hard.sum())}, позитивов {D.label.sum()}")
    STRUCT = ["ro_only", "has_text", "has_write", "transfer", "claimed_action_no_call", "repeat_call", "missing_numbers",
              "ungrounded_values", "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
    LEN = ["log_resp", "log_prompt", "n_calls", "turn"]
    B2 = ["b_vfrac", "b_mean", "b_missing"]
    B1 = ["b1_p", "b1_vote", "b_missing"]
    S = ["s_max", "s_mean", "s_missing"]
    rows = []

    def run(name, cols, phases, Fx=F, b_frac=1.0):
        r = evaluate(D, Fx, cols)
        r.update({"конфигурация": name, "мин_N800": minutes(phases, b_frac)})
        rows.append(r)
        print(f"{name:52s} AUC {r['auc']:.3f} | mix {r['mix']:.3f} (t={r['t_mix']}) | ratio {r['ratio']:.3f} | {r['мин_N800']:.0f} мин")

    run("полный (как в ноутбуке)", ["la", "la_base", "la_v1"] + B2 + S + STRUCT, ["A", "A_base", "A_v1", "S", "B1", "B2"])
    run("полный + длины", ["la", "la_base", "la_v1"] + B2 + S + STRUCT + LEN, ["A", "A_base", "A_v1", "S", "B1", "B2"])
    run("A + A_base + B(2) + длины", ["la", "la_base"] + B2 + STRUCT + LEN, ["A", "A_base", "B1", "B2"])
    run("A + A_base + B1 + длины", ["la", "la_base"] + B1 + STRUCT + LEN, ["A", "A_base", "B1"])
    for fr in (0.85, 0.7, 0.5, 0.35):
        Fx = b_subset(F, fr)
        run(f"A + A_base + B1 (доля {fr}) + длины", ["la", "la_base"] + B1 + STRUCT + LEN, ["A", "A_base", "B1"], Fx, fr)
    for fr in (0.5, 0.35):
        Fx = b_subset(F, fr)
        run(f"A + A_base + B(2) (доля {fr}) + длины", ["la", "la_base"] + B2 + STRUCT + LEN, ["A", "A_base", "B1", "B2"], Fx, fr)
    run("A + A_base + длины (без B)", ["la", "la_base"] + STRUCT + LEN, ["A", "A_base"])
    run("A + A_base (без B, без длин)", ["la", "la_base"] + STRUCT, ["A", "A_base"])
    run("A + A_base + A_v1 + S + длины (без B)", ["la", "la_base", "la_v1"] + S + STRUCT + LEN, ["A", "A_base", "A_v1", "S"])
    run("только A + длины", ["la"] + STRUCT + LEN, ["A"])
    run("только A_base + длины", ["la_base"] + STRUCT + LEN, ["A_base"])
    T = pd.DataFrame(rows)
    T.round(4).to_csv(ROOT / "colab_results/v2_main/lab.csv", index=False)
    print("\n-> colab_results/v2_main/lab.csv")
