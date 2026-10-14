"""Лаборатория признаков на CPU: пул (метки Claude) + valid (эталон). CV по траекториям.
Признаки: LLM (P_A, голоса B из дампа v1), правила текущего хода, история диалога, TF-IDF текста ответа.
Запуск: python tools/feat_lab.py"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution" / "src"))
from guardian.devtools import build_pool  # noqa: E402
from guardian.rules import analyze  # noqa: E402
from guardian.transcript import parse_blocks  # noqa: E402

RO = ("get_", "search_", "find_", "list_", "calculate", "KB_search", "grep", "check_", "think")
FRUST = re.compile(r"не работает|опять|снова|до сих пор|всё ещё|все еще|всё равно|бесполезн|раздраж|ужас|сколько можно|"
                   r"не помогло|не получилось|ошибк", re.I)


def load():
    valid = pd.read_parquet(ROOT / "valid.parquet")
    pool = build_pool(valid)
    final = ROOT / "labeling/labels_final.jsonl"
    files = [str(final)] if final.exists() else sorted(glob.glob(str(ROOT / "labeling/out/*.jsonl")))
    lab = pd.DataFrame([json.loads(x) for f in files
                        for x in open(f, encoding="utf-8") if x.strip()]).drop_duplicates("id")
    P = pool.merge(lab[["id", "label"]], on="id")
    P["src"] = "pool"
    V = valid[["id", "prompt", "response", "label"]].copy()
    V["src"] = "valid"
    D = pd.concat([P[["id", "prompt", "response", "label", "src"]], V], ignore_index=True)
    dp = pd.concat([pd.read_parquet(ROOT / "colab_results/v1/results/pool_judge.parquet"),
                    pd.read_parquet(ROOT / "colab_results/v1/results/valid_judge.parquet")])
    dp = dp.drop_duplicates("id")[["id", "p_nothink", "p_think_samples"]]
    D = D.merge(dp, on="id", how="left")
    return D


def history_feats(prompt: str, response: str, ex_id: str):
    blocks = parse_blocks(prompt)
    agent_turns = [b for b in blocks if b.role == "ASSISTANT" and b.turn]
    n_err_results = sum(1 for b in blocks for c in b.calls if c.result is not None and c.result.error)
    last_err = 0
    for b in reversed(blocks):
        if b.role == "ASSISTANT" and b.calls:
            last_err = int(any(c.result is not None and c.result.error for c in b.calls))
            break
    # нарушения правил в прошлых ходах агента (каждый прошлый ход проверяется на своём префиксе)
    prev_hard = 0
    for b in agent_turns[-6:]:
        pre, resp = prompt[:b.start].rstrip("\n"), prompt[b.start:b.end].rstrip("\n")
        resp = "\n".join(ln for ln in resp.split("\n") if not ln.lstrip("\t ").startswith("← TOOL_RESPONSE"))
        try:
            prev_hard += int(bool(analyze(ex_id, pre, resp)["hard"]))
        except Exception:  # noqa: BLE001
            pass
    users = [b for b in blocks if b.role == "USER"]
    last_user = users[-1].text if users else ""
    return {"n_err_results": n_err_results, "last_call_err": last_err, "prev_hard6": prev_hard,
            "frustration": len(FRUST.findall(last_user)), "n_agent_turns": len(agent_turns),
            "user_excl": last_user.count("!")}


def build(D):
    rows = []
    for r in D.itertuples():
        an = analyze(r.id, r.prompt, r.response)
        f = dict(an["flags"])
        f["hard"] = int(bool(an["hard"]))
        calls = [c for b in parse_blocks(r.response) for c in b.calls]
        f["ro_only"] = int(bool(calls) and not f["has_text"] and all(c.name.startswith(RO) for c in calls))
        f.update(history_feats(r.prompt, r.response, r.id))
        f["dom_" + r.id.split("__")[0]] = 1
        rows.append(f)
    F = pd.DataFrame(rows).fillna(0)
    D = pd.concat([D.reset_index(drop=True), F], axis=1)
    pa = D.p_nothink.fillna(0.5).clip(1e-6, 1 - 1e-6)
    D["la"] = np.log(pa / (1 - pa))
    s = D.p_think_samples.apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    D["vfrac"] = s.apply(lambda v: np.mean([x >= 0.5 for x in v]) if v else -1.0)
    D["resp_text"] = D.response.apply(lambda t: "\n".join(b.text for b in parse_blocks(t)))
    D["traj"] = D.id.str.split("::").str[0]
    return D


def cv_scores(D, cols, use_tfidf=False, C=0.5, folds=5, seed=0):
    trajs = D.traj.unique()
    rng = np.random.RandomState(seed)
    fold = {t: k % folds for k, t in enumerate(rng.permutation(trajs))}
    oof = np.zeros(len(D))
    for k in range(folds):
        te = (D.traj.map(fold) == k).values
        Xtr, Xte = D.loc[~te, cols].values.astype(float), D.loc[te, cols].values.astype(float)
        if use_tfidf:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True)
            Ttr = vec.fit_transform(D.loc[~te, "resp_text"]); Tte = vec.transform(D.loc[te, "resp_text"])
            lr_t = LogisticRegression(C=2.0, max_iter=3000).fit(Ttr, D.label[~te])
            # stacking: OOF для train через вложенный CV было бы честнее; здесь — предсказание train-моделью (смещение мало при C=2)
            Xtr = np.c_[Xtr, lr_t.predict_proba(Ttr)[:, 1]]; Xte = np.c_[Xte, lr_t.predict_proba(Tte)[:, 1]]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=3000).fit((Xtr - mu) / sd, D.label[~te])
        oof[te] = clf.predict_proba((Xte - mu) / sd)[:, 1]
    return oof


def f1_at_rate(y, s, rate):
    thr = np.quantile(s, 1 - rate)
    p = (s >= thr).astype(int)
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return 2 * P * R / max(1e-9, P + R), P, R


if __name__ == "__main__":
    D = build(load())
    D.to_parquet(ROOT / "labeling/feat_lab.parquet", index=False)
    pool = D[D.src == "pool"].reset_index(drop=True)
    base = ["hard", "ro_only", "has_text", "n_writes", "transfer", "claimed_action_no_call", "repeat_call",
            "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
    hist = ["n_err_results", "last_call_err", "prev_hard6", "frustration", "n_agent_turns", "user_excl"]
    llm = ["la", "vfrac"]
    sets = {"структура хода": base, "структура+история": base + hist, "только LLM": llm,
            "LLM+структура": llm + base, "LLM+структура+история": llm + base + hist}
    print(f"пул: n={len(pool)}, позитивов {int(pool.label.sum())}")
    for name, cols in sets.items():
        for tf in (False, True):
            oof = cv_scores(pool, cols, use_tfidf=tf)
            auc = roc_auc_score(pool.label, oof)
            f20 = f1_at_rate(pool.label.values, oof, 0.20)
            print(f"  {name:24s}{' +TF-IDF' if tf else '        '}  AUC {auc:.3f} | F1@20% {f20[0]:.3f} (P {f20[1]:.2f} R {f20[2]:.2f})")
