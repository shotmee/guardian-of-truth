"""Ветка D: контекстная абляция (PMI) — генератор ноутбука (импортируется из make_branch_notebooks.py)."""


def branch_pmi(NB, settings, common, summary):
    nb = NB()
    nb.md(r"""
# Guardian of Truth — ветка D: контекстная абляция (PMI) как признак

**Идея:** модель-наблюдатель (наша Qwen3.6-35B-A3B) оценивает вероятность ответа агента в трёх вариантах контекста:
- полный контекст;
- без результатов инструментов (`← TOOL_RESPONSE`);
- без policy.

Подкреплённое содержание (суммы, ID, условия из базы знаний) резко теряет вероятность без результатов инструментов. Выдуманное — почти не теряет (PMI, Nandwani et al. 2023; GASP 2026).

Признаки — средняя logprob ответа и разности между вариантами. Проверяем, добавляют ли они что-то к признакам v1 (P_A судьи + структура хода), через стекинг с CV по траекториям и симулятор паблика. Генерации нет, только prefill: 3 прохода на пример.

**Ресурсы:** ≈1–1.5 ч ≈ 8–12 CU. Нужны дампы v1 на Диске (`guardian_of_truth/v1/results`).
""")
    settings(nb, "pmi", True)
    common(nb)
    nb.code(r'''
# ===== 7. Сервер (генеративный, нужны prompt_logprobs) =====
DEV_CFG = f"{SOL}/src/guardian/config_dev_v2.json"
proc, BASE, LOAD_S = serve(f"{SOL}/model", DEV_CFG, "vllm_pmi")
''')
    nb.code(r'''
# ===== 8. Три варианта контекста и logprob ответа =====
import re, requests
from concurrent.futures import ThreadPoolExecutor
ALL = pd.concat([pool[["id", "prompt", "response"]], dv[["id", "prompt", "response"]]]).drop_duplicates("id")
ALL = ALL[ALL.id.isin(LAB)].reset_index(drop=True)
SERVED = json.load(open(DEV_CFG))["served_name"]
def no_tools(p):
    out, skip = [], False
    for ln in p.split("\n"):
        s = ln.lstrip("\t ")
        if s.startswith("← TOOL_RESPONSE"):
            skip = True; out.append("\t← TOOL_RESPONSE (скрыто)"); continue
        if skip and ln.startswith("\t") and not s.startswith("→"):
            continue
        skip = False; out.append(ln)
    return "\n".join(out)
def no_policy(p):
    return re.sub(r"<policy>.*?</policy>", "<policy>(скрыто)</policy>", p, flags=re.S)
def resp_lp(prefix, resp):
    txt = prefix + "\n\n" + resp
    n_pre = requests.post(BASE.replace("/v1", "") + "/tokenize", json={"model": SERVED, "prompt": prefix + "\n\n"}, timeout=300).json()["count"]
    r = requests.post(BASE + "/completions", json={"model": SERVED, "prompt": txt, "max_tokens": 1, "temperature": 0,
                                                    "prompt_logprobs": 0}, timeout=900).json()
    pl = r["choices"][0]["prompt_logprobs"]
    vals = [list(d.values())[0]["logprob"] for d in pl[n_pre:] if d]
    return np.array(vals, float)
def feats_pmi(row):
    try:
        full = resp_lp(row.prompt[-280000:], row.response)
        nt = resp_lp(no_tools(row.prompt)[-280000:], row.response)
        npl = resp_lp(no_policy(row.prompt)[-280000:], row.response)
        k = min(len(full), len(nt), len(npl))
        if k == 0:
            return None
        f, t, p_ = full[-k:], nt[-k:], npl[-k:]
        d = f - t
        return {"lp_mean": f.mean(), "lp_p10": np.percentile(f, 10), "pmi_tools": d.mean(),
                "pmi_tools_p10": np.percentile(d, 10), "pmi_tools_frac_neg": float((d < -1).mean()),
                "pmi_policy": (f - p_).mean(), "n_tok": k}
    except Exception as e:
        print("ошибка", row.id, str(e)[:200]); return None
with ThreadPoolExecutor(12) as ex:
    PM = list(tqdm(ex.map(feats_pmi, ALL.itertuples()), total=len(ALL), desc="PMI"))
PMI = pd.DataFrame([dict(id=i, **(p or {})) for i, p in zip(ALL.id, PM)])
PMI.to_parquet(f"{RES}/pmi_feats.parquet", index=False); stop_vllm(proc)
display(PMI.describe().round(3))
''')
    nb.code(r'''
# ===== 9. Прибавка к признакам v1: стекинг с CV по траекториям + симулятор паблика =====
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from guardian.rules import analyze
V1 = pd.concat([pd.read_parquet(f"{DRIVE_ROOT}/v1/results/pool_judge.parquet"),
                pd.read_parquet(f"{DRIVE_ROOT}/v1/results/valid_judge.parquet")]).drop_duplicates("id")
M = ALL.merge(V1[["id", "p_nothink", "p_think_samples"]], on="id", how="left").merge(PMI, on="id", how="left")
AN = [analyze(r.id, r.prompt, r.response) for r in M.itertuples()]
M["hard"] = [bool(a["hard"]) for a in AN]; M["label"] = M.id.map(LAB)
for k in ("ro_only", "has_text", "n_writes", "transfer", "claimed_action_no_call"):
    M[k] = [a["flags"].get(k, 0) for a in AN]
pa = M.p_nothink.fillna(0.5).clip(1e-6, 1 - 1e-6); M["la"] = np.log(pa / (1 - pa))
M["vfrac"] = [np.mean([x >= .5 for x in json.loads(s)]) if isinstance(s, str) and json.loads(s) else -1 for s in M.p_think_samples]
for d_ in ("airline", "banking_knowledge", "retail", "telecom"):
    M["dom_" + d_] = (M.id.str.split("__").str[0] == d_).astype(float)
BASEF = ["la", "vfrac", "ro_only", "has_text", "n_writes", "transfer", "claimed_action_no_call",
         "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
PMIF = ["lp_mean", "lp_p10", "pmi_tools", "pmi_tools_p10", "pmi_tools_frac_neg", "pmi_policy"]
M[PMIF] = M[PMIF].fillna(M[PMIF].median())
traj = M.id.str.split("::").str[0].values; nh = ~M.hard.values; y = M.label.astype(int).values
pool_mask = M.id.isin(set(pool.id)).values
def cv(cols, seed):
    rng = np.random.RandomState(seed); ut = rng.permutation(np.unique(traj)); fold = {t: k % 5 for k, t in enumerate(ut)}
    oof = np.zeros(len(M)); X = M[cols].values.astype(float)
    for k in range(5):
        te = np.array([fold[t] == k for t in traj]); tr = ~te & nh
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        oof[te] = LogisticRegression(C=0.5, max_iter=5000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return oof
rows = []
for name, cols in (("v1 (la+B+структура)", BASEF), ("v1 + PMI", BASEF + PMIF), ("только PMI + структура", PMIF + BASEF[2:])):
    aucs, best = [], []
    for sd in range(5):
        o = cv(cols, sd); s = np.where(M.hard.values, 1.0, o)
        aucs.append(roc_auc_score(y[nh], o[nh]))
        best.append(max(public_from_preds(y[pool_mask], s[pool_mask] >= t)["f1"] for t in np.arange(0.1, 0.9, 0.05)))
    rows.append({"признаки": name, "oof_auc": np.mean(aucs), "паблик_лучший_порог": np.mean(best)})
PMR = pd.DataFrame(rows); display(PMR.round(3))
for c in PMIF:
    print(f"AUC отдельного признака {c}: {roc_auc_score(y[nh], -M.loc[nh, c]):.3f} (чем меньше значение, тем подозрительнее)")
''')
    summary(nb, r'''summary.update({"pmi": PMR.round(4).to_dict("records")})''')
    nb.save("guardian_pmi.ipynb")
