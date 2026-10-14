"""Лёгкая ветка C+D для L4 (обычная ОЗУ, без vLLM и весов 35B) — генератор ноутбука (импортируется из make_branch_notebooks.py)."""


def branch_lite(NB, settings, common, summary):
    nb = NB(gpu="L4", high_ram=False)
    nb.md(r"""
# Guardian of Truth — лёгкая ветка (L4): контекстная абляция + зонд на маленькой модели

Запускается **параллельно** с основным ноутбуком: нужны L4 (24 ГБ) и обычная ОЗУ. vLLM и веса 35B не нужны.

**Что делает.** Модель-наблюдатель Qwen3.5-4B (transformers, bf16) читает контекст агента и его ответ. Для каждого хода — 3 прохода (только prefill, без генерации):
1. Полный контекст. Logprob каждого токена ответа + скрытые состояния ответа на слоях 1/3, 1/2, 2/3 (для зонда).
2. Контекст без `← TOOL_RESPONSE`. Подкреплённое данными (суммы, ID, условия) резко теряет вероятность, выдуманное — нет (PMI).
3. Контекст без policy.

Дальше — стекинг с CV по траекториям и симулятор паблика: насколько PMI и зонд добавляют к признакам v1. Если прирост заметный, то же самое делаем на 35B (ветки C/D на A100) или встраиваем 4B в сабмит (≈ +9 ГБ к архиву).

**Ресурсы:** L4, обычная ОЗУ (нужно ~10–15 ГБ), диск ~15 ГБ. ≈1.5–2 ч. CU — ориентировочно 3–5 (точный расход в час видно в панели «Ресурсы» Colab). Чекпоинты на Диск: при обрыве перезапусти, продолжит с места остановки.

**Нужно на Диске:** `guardian_of_truth/v1/results/{pool,valid}_judge.parquet` (дампы v1). Без них ноутбук тоже отработает, но без сравнения с v1.
""")
    settings(nb, "lite_l4", False)
    common(nb, lite=True)
    nb.code(r'''
# ===== 7. Модель-наблюдатель (transformers, bf16) + проверка, что веса загрузились правильно =====
import gc, re, torch, transformers
from transformers import AutoTokenizer
from tqdm.auto import tqdm
assert torch.cuda.is_bf16_supported(), "GPU без bf16 (T4) — выбери L4"
MODEL_NAME = "Qwen/Qwen3.5-4B"        # на A100 можно "Qwen/Qwen3.5-9B" и MAX_TOK = 65536
MAX_CHARS, MAX_TOK, MAX_RESP_TOK, ABLATE_POLICY = 90000, 32768, 3072, True
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def find_decoder(m):
    for path in ("model.language_model", "language_model", "model.model", "model"):
        o = m
        try:
            for a in path.split("."): o = getattr(o, a)
        except AttributeError:
            continue
        if hasattr(o, "layers") and hasattr(o, "norm"): return o
    raise RuntimeError("не нашёл текстовый декодер")

CAP = {"on": False, "n_pre": 0}
@torch.no_grad()
def resp_lp(prefix, resp_ids, max_tok):
    """logprob каждого токена ответа при данном контексте (хвост контекста, если не влезает)."""
    pre = tok(prefix + "\n\n", add_special_tokens=False)["input_ids"]
    ids = pre[-(max_tok - len(resp_ids)):] + resp_ids
    n_pre = len(ids) - len(resp_ids)
    x = torch.tensor([ids], device="cuda")
    CAP["n_pre"] = n_pre
    h = DEC(input_ids=x, use_cache=False).last_hidden_state[0]
    hs, tgt = h[n_pre - 1:-1], x[0, n_pre:]
    out = []
    for i in range(0, len(tgt), 512):
        lg = HEAD(hs[i:i + 512]).float()
        out.append(torch.log_softmax(lg, -1).gather(1, tgt[i:i + 512, None])[:, 0])
    return torch.cat(out).cpu().numpy()

r0 = dv.iloc[0]
def smoke():
    ids = tok(r0.response, add_special_tokens=False)["input_ids"][:256]
    return float(resp_lp(r0.prompt[-6000:], ids, 8192).mean())

mdl = None
for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
    try:
        m = getattr(transformers, cls).from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda").eval()
        DEC, HEAD = find_decoder(m), m.get_output_embeddings()
        lp0 = smoke()
        print(f"{cls}: {type(m).__name__} | средний logprob токена ответа на примере: {lp0:.2f}")
        if lp0 > -6:
            mdl = m; break
        print("подозрительно низкий logprob — похоже, веса загрузились не полностью; пробую другой класс")
    except Exception as e:
        print(cls, "не подошёл:", str(e)[:300])
    m = None; gc.collect(); torch.cuda.empty_cache()
assert mdl is not None, "модель не загрузилась — пришли Claude вывод этой ячейки"
NL = len(DEC.layers); LAYERS = [NL // 3, NL // 2, (2 * NL) // 3]
def mk_hook(L):
    def f(mod, inp, out):
        if CAP["on"]:
            h = out[0] if isinstance(out, (tuple, list)) else out
            s = CAP["n_pre"]
            CAP[L] = torch.cat([h[0, s:].float().mean(0), h[0, -1].float()]).cpu().numpy()
    return f
for hd in globals().get("HOOKS", []): hd.remove()
HOOKS = [DEC.layers[L].register_forward_hook(mk_hook(L)) for L in LAYERS]
print(f"слоёв {NL}, зонд на слоях {LAYERS} | GPU занято {torch.cuda.memory_allocated() / 2**30:.1f} ГБ")
''')
    nb.code(r'''
# ===== 8. Прогон: logprob ответа в 3 контекстах + скрытые состояния ответа (чекпоинт каждые 20 примеров) =====
from guardian.prompts import truncate_prompt
TAG = MODEL_NAME.split("/")[-1]
CACHE_F, HID_F = f"{CACHE}/lite_{TAG}.jsonl", f"{RES}/hid_{TAG}.npz"
ALL = pd.concat([pool[["id", "prompt", "response"]], dv[["id", "prompt", "response"]]]).drop_duplicates("id")
ALL = ALL[ALL.id.isin(LAB)].reset_index(drop=True)
HID = {}
if os.path.exists(HID_F):
    z = np.load(HID_F); HID = {str(i): x for i, x in zip(z["ids"], z["X"])}
DONE = {}
if os.path.exists(CACHE_F):
    for l in open(CACHE_F, encoding="utf-8"):
        try:
            r = json.loads(l); DONE[r["id"]] = r
        except Exception:
            pass
DONE = {k: v for k, v in DONE.items() if k in HID}   # без скрытых состояний — пересчитать
print(f"примеров {len(ALL)} | уже готово {len(DONE)}")

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

def feats(r, max_tok):
    rid = tok(r.response, add_special_tokens=False)["input_ids"][:MAX_RESP_TOK]
    CAP["on"] = True
    f = resp_lp(truncate_prompt(r.prompt, MAX_CHARS), rid, max_tok)
    CAP["on"] = False
    hid = np.concatenate([CAP[L] for L in LAYERS]).astype(np.float32)
    t = resp_lp(truncate_prompt(no_tools(r.prompt), MAX_CHARS), rid, max_tok)
    p_ = resp_lp(truncate_prompt(no_policy(r.prompt), MAX_CHARS), rid, max_tok) if ABLATE_POLICY else f
    d = f - t
    dig = np.array([any(ch.isdigit() for ch in s) for s in tok.convert_ids_to_tokens(rid)])
    row = {"id": r.id, "n_tok": len(rid), "max_tok": max_tok,
           "lp_mean": f.mean(), "lp_p10": np.percentile(f, 10), "lp_min": f.min(),
           "pmi_tools": d.mean(), "pmi_tools_p10": np.percentile(d, 10), "pmi_tools_min": d.min(),
           "pmi_tools_frac_neg": (d < -1).mean(), "pmi_policy": (f - p_).mean(),
           "lp_digit": f[dig].mean() if dig.any() else 0.0, "pmi_tools_digit": d[dig].mean() if dig.any() else 0.0,
           "n_digit": int(dig.sum())}
    return {k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}, hid

def save_hid():
    np.savez(HID_F, ids=np.array(list(HID)), X=np.stack(list(HID.values())))

todo = [r for r in ALL.itertuples() if r.id not in DONE]
t0, n_new = time.time(), 0
with open(CACHE_F, "a", encoding="utf-8") as fo:
    for r in tqdm(todo, desc=f"3 прохода на пример ({TAG})"):
        row = None
        for lim in (MAX_TOK, MAX_TOK // 2, MAX_TOK // 4):
            try:
                row, hid = feats(r, lim); break
            except torch.cuda.OutOfMemoryError:
                CAP["on"] = False; gc.collect(); torch.cuda.empty_cache(); print(f"\nOOM на {r.id} при {lim} токенах — уменьшаю")
        if row is None:
            continue
        DONE[r.id], HID[r.id] = row, hid; n_new += 1
        fo.write(json.dumps(row, ensure_ascii=False) + "\n"); fo.flush()
        if n_new % 20 == 0:
            save_hid()
if HID:
    save_hid()
SEC_PER_EX = (time.time() - t0) / max(1, n_new)
print(f"готово {len(DONE)} из {len(ALL)} | {SEC_PER_EX:.1f} с на пример | файлы: {CACHE_F}, {HID_F}")
''')
    nb.code(r'''
# ===== 9. Что это добавляет: стекинг с CV по траекториям + симулятор паблика =====
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from guardian.rules import analyze
M = pd.DataFrame([DONE[i] for i in ALL.id if i in DONE and i in HID])
v1p = f"{DRIVE_ROOT}/v1/results"
HAVE_V1 = os.path.exists(f"{v1p}/pool_judge.parquet") and os.path.exists(f"{v1p}/valid_judge.parquet")
if HAVE_V1:
    V1 = pd.concat([pd.read_parquet(f"{v1p}/pool_judge.parquet"), pd.read_parquet(f"{v1p}/valid_judge.parquet")]).drop_duplicates("id")
    M = M.merge(V1[["id", "p_nothink", "p_think_samples"]], on="id", how="left")
else:
    print("⚠️ дампов v1 на Диске нет — сравнение без признаков судьи v1"); M["p_nothink"] = np.nan; M["p_think_samples"] = None
SRC = ALL.set_index("id")
AN = [analyze(i, SRC.at[i, "prompt"], SRC.at[i, "response"]) for i in M.id]
M["hard"] = [bool(a["hard"]) for a in AN]; M["label"] = M.id.map(LAB).astype(int)
for k in ("ro_only", "has_text", "n_writes", "transfer", "claimed_action_no_call"):
    M[k] = [float(a["flags"].get(k, 0)) for a in AN]
pa = M.p_nothink.astype(float).fillna(0.5).clip(1e-6, 1 - 1e-6); M["la"] = np.log(pa / (1 - pa))
def _vf(s):
    try:
        v = json.loads(s) if isinstance(s, str) else list(s)
    except Exception:
        v = []
    return float(np.mean([x >= .5 for x in v])) if len(v) else -1.0
M["vfrac"] = [_vf(s) for s in M.p_think_samples]
for d_ in ("airline", "banking_knowledge", "retail", "telecom"):
    M["dom_" + d_] = (M.id.str.split("__").str[0] == d_).astype(float)
M["log_n_tok"] = np.log1p(M.n_tok.astype(float))   # длина ответа: lp_min/lp_p10 сильно от неё зависят — держим в базе
STRUCT = ["ro_only", "has_text", "n_writes", "transfer", "claimed_action_no_call", "log_n_tok",
          "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
JUDGE = ["la", "vfrac"] if HAVE_V1 else []
PMIF = ["lp_mean", "lp_p10", "lp_min", "pmi_tools", "pmi_tools_p10", "pmi_tools_min", "pmi_tools_frac_neg",
        "pmi_policy", "lp_digit", "pmi_tools_digit"]
M[PMIF] = M[PMIF].astype(float).fillna(M[PMIF].median())
H = np.stack([HID[i] for i in M.id]).astype(np.float32); D_ = H.shape[1] // (2 * len(LAYERS))
traj = M.id.str.split("::").str[0].values; hard = M.hard.values; y = M.label.values
pool_mask = M.id.isin(set(pool.id)).values
print(f"примеров {len(M)} | hard {hard.sum()} | позитивов {y.sum()} | в пуле {pool_mask.sum()}")

def folds_of(seed):
    rng = np.random.RandomState(seed); ut = rng.permutation(np.unique(traj)); f = {t: k % 5 for k, t in enumerate(ut)}
    return np.array([f[t] for t in traj])
def cv(X, seed, C=0.5, pca=None):
    fo = folds_of(seed); oof = np.zeros(len(y))
    for k in range(5):
        te = fo == k; tr = ~te & ~hard
        Xtr, Xte = X[tr], X[te]
        if pca:
            pc = PCA(n_components=min(pca, tr.sum() - 1, Xtr.shape[1]), random_state=0).fit(Xtr); Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        oof[te] = LogisticRegression(C=C, max_iter=5000).fit((Xtr - mu) / sd, y[tr]).predict_proba((Xte - mu) / sd)[:, 1]
    return oof
def score(oof):
    s = np.where(hard, 1.0, oof)
    pub = max(public_from_preds(y[pool_mask], s[pool_mask] >= t)["f1"] for t in np.arange(0.05, 0.95, 0.025))
    return roc_auc_score(y[~hard], oof[~hard]), pub

SEEDS = (0, 1, 2)
rows, PROBE_OOF = [], {}
for j, L in enumerate(LAYERS):
    blk = H[:, j * 2 * D_:(j + 1) * 2 * D_]
    for part, X in (("mean", blk[:, :D_]), ("last", blk[:, D_:])):
        for C in (0.02, 0.1):
            o = np.mean([cv(X, s, C, pca=64) for s in SEEDS], 0)
            auc, pub = score(o)
            rows.append({"признаки": f"зонд: слой {L}, {part}, C={C}", "oof_auc": auc, "паблик_лучший_порог": pub})
            PROBE_OOF[(L, part, C)] = o
best_key = max(PROBE_OOF, key=lambda k: roc_auc_score(y[~hard], PROBE_OOF[k][~hard]))
po = np.clip(PROBE_OOF[best_key], 1e-4, 1 - 1e-4); M["probe"] = np.log(po / (1 - po))
print("лучший зонд:", best_key)
SETS = {"структура": STRUCT, "PMI + структура": PMIF + STRUCT, "зонд + структура": ["probe"] + STRUCT,
        "PMI + зонд + структура": PMIF + ["probe"] + STRUCT}
if HAVE_V1:
    SETS.update({"v1 (la+B+структура)": JUDGE + STRUCT, "v1 + PMI": JUDGE + PMIF + STRUCT,
                 "v1 + зонд": JUDGE + ["probe"] + STRUCT, "v1 + PMI + зонд": JUDGE + PMIF + ["probe"] + STRUCT})
for name, cols in tqdm(SETS.items(), desc="стекинг"):
    X = M[cols].values.astype(float)
    res = [score(cv(X, s, 0.5)) for s in SEEDS]
    rows.append({"признаки": name, "oof_auc": np.mean([r[0] for r in res]), "паблик_лучший_порог": np.mean([r[1] for r in res])})
R = pd.DataFrame(rows).sort_values("паблик_лучший_порог", ascending=False).reset_index(drop=True)
display(R.round(3))
print("паблик_лучший_порог — симулятор при лучшем пороге (оптимистично); сравнивать строки между собой.")
SINGLE = {c: roc_auc_score(y[~hard], -M.loc[~hard, c]) for c in PMIF}
for c, a in SINGLE.items():
    print(f"AUC отдельного признака {c}: {a:.3f} (меньше значение — подозрительнее; < 0.5 — наоборот)")
''')
    summary(nb, r'''summary.update({"model": MODEL_NAME, "max_tok": MAX_TOK, "sec_per_example": SEC_PER_EX, "have_v1": HAVE_V1,
                "n": int(len(M)), "best_probe": str(best_key), "lite": R.round(4).to_dict("records"),
                "single_auc": {k: round(v, 4) for k, v in SINGLE.items()}})''')
    nb.save("guardian_lite_l4.ipynb")
