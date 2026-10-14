"""Генерирует ноутбуки веток решения в notebooks_extended/ из файлов solution/ и разметки labeling/out/.
Запуск: python tools/make_branch_notebooks.py"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solution"
OUT = ROOT / "notebooks_extended"


class NB:
    def __init__(self, gpu="A100", high_ram=True):
        self.cells, self.gpu, self.high_ram = [], gpu, high_ram

    def md(self, s):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})

    def code(self, s):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": s.strip("\n").splitlines(keepends=True)})

    def save(self, name):
        nb = {"cells": self.cells,
              "metadata": {"accelerator": "GPU", "colab": {"gpuType": self.gpu, **({"machine_shape": "hm"} if self.high_ram else {})},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 0}
        OUT.mkdir(exist_ok=True)
        (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("->", OUT / name, f"({len(self.cells)} ячеек)")


def settings(nb, version, need_weights):
    nb.code(rf'''
# ===== НАСТРОЙКИ =====
VERSION = "{version}"
KAGGLE_DATASET = "shotme/aij-minecraft-kulbit"
DRIVE_ROOT = "/content/drive/MyDrive/guardian_of_truth"
WORK = "/content/work"
SOL = f"{{WORK}}/solution"
MODEL_DIR = f"{{WORK}}/model"            # веса бейзлайна Qwen3.6-35B-A3B-FP8 (Kaggle)
NEED_BASELINE_WEIGHTS = {need_weights}
VLLM_VERSION = "0.19.1"                # как в среде проверки
DELETE_PARTS_AFTER_EXTRACT = True
import time, os, sys, json, shutil, subprocess; T_NOTEBOOK = time.time()
''')


def common(nb, lite=False):
    """lite=True: без vLLM и весов 35B, GPU от 20 ГБ с bf16 (L4) — для параллельного запуска на обычной ОЗУ."""
    nb.code(r'''
# ===== 1. Железо =====
q = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).decode().strip()
GPU_NAME, GPU_MEM = q.split(", ")[0], int(q.split(", ")[1])
print(f"GPU: {GPU_NAME}, {GPU_MEM/1024:.0f} ГБ | RAM: {os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**30:.0f} ГБ | диск: {shutil.disk_usage('/content').free/2**30:.0f} ГБ")
''' + ('assert GPU_MEM >= 20000, "Нужна GPU от 22 ГБ с bf16 (L4 / A100 / H100); T4 не подойдёт"'
       if lite else 'assert GPU_MEM >= 70000, "Нужна GPU 80 ГБ (A100 80GB / H100)"') + r'''
CU_PER_HOUR = 15.0 if "H100" in GPU_NAME else 7.5 if "A100" in GPU_NAME else 2.0   # L4 — ориентировочно, точное значение в панели «Ресурсы» Colab
''')
    nb.code(r'''
# ===== 2. Google Drive =====
from google.colab import drive
drive.mount("/content/drive")
DV = f"{DRIVE_ROOT}/{VERSION}"
RES, LOGS, CACHE = f"{DV}/results", f"{DV}/logs", f"{DV}/cache"
for d in (RES, LOGS, CACHE, WORK):
    os.makedirs(d, exist_ok=True)
print("Результаты:", DV)
''')
    nb.code(r'''
# ===== 3. Установка (≈3-6 мин) =====
def pip(*a): return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a]).returncode
''' + (r'''pip("-U", "transformers", "accelerate")   # лёгкий режим: vLLM не нужен, нужна свежая версия transformers с qwen3_5
pip("kagglehub", "pyarrow", "pandas", "tqdm", "scikit-learn", "huggingface_hub")
print(subprocess.check_output([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"]).decode())
''' if lite else r'''if pip(f"vllm=={VLLM_VERSION}", "mistral_common>=1.11.5") != 0:
    pip("vllm", "mistral_common>=1.11.5")
pip("kagglehub", "pyarrow", "pandas", "openai", "tqdm", "scikit-learn", "huggingface_hub")
print(subprocess.check_output([sys.executable, "-c", "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"]).decode())
'''))
    nb.code(r'''
# ===== 4. Данные с Kaggle (valid.parquet + веса бейзлайна при необходимости) =====
import glob, kagglehub
try:
    from google.colab import userdata
    os.environ.setdefault("KAGGLE_USERNAME", userdata.get("KAGGLE_USERNAME")); os.environ.setdefault("KAGGLE_KEY", userdata.get("KAGGLE_KEY"))
except Exception:
    print("секреты Kaggle не заданы — пробую без них")
FILES = ["valid.parquet"] + ([f"baseline.tar.gz.0{i}" for i in range(7)] if NEED_BASELINE_WEIGHTS and not os.path.exists(f"{MODEL_DIR}/.extract_ok") else [])
PATHS = {}
for f in FILES:
    PATHS[f] = kagglehub.dataset_download(KAGGLE_DATASET, path=f); print("✓", f)
shutil.copy(PATHS["valid.parquet"], f"{WORK}/valid.parquet")
''')
    nb.code(r'''
# ===== 5. Распаковка весов бейзлайна (чекпоинт .extract_ok) =====
import io, tarfile
from tqdm.auto import tqdm
if NEED_BASELINE_WEIGHTS and not os.path.exists(f"{MODEL_DIR}/.extract_ok"):
    parts = [PATHS[f"baseline.tar.gz.0{i}"] for i in range(7)]
    bar = tqdm(total=sum(os.path.getsize(p) for p in parts), unit="B", unit_scale=True, desc="распаковка")
    class Concat(io.RawIOBase):
        def __init__(self, paths): self.paths, self.k, self.f = list(paths), 0, None
        def readable(self): return True
        def readinto(self, b):
            while True:
                if self.f is None:
                    if self.k >= len(self.paths): return 0
                    self.f = open(self.paths[self.k], "rb")
                n = self.f.readinto(b)
                if n: bar.update(n); return n
                self.f.close()
                if DELETE_PARTS_AFTER_EXTRACT: os.remove(self.paths[self.k])
                self.f, self.k = None, self.k + 1
    os.makedirs(MODEL_DIR, exist_ok=True)
    with tarfile.open(fileobj=io.BufferedReader(Concat(parts), 1 << 24), mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith("baseline/model/"):
                dst = os.path.join(MODEL_DIR, m.name[len("baseline/model/"):]); os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tf.extractfile(m) as src, open(dst, "wb") as out: shutil.copyfileobj(src, out, 1 << 24)
    bar.close(); open(f"{MODEL_DIR}/.extract_ok", "w").write("ok")
print("веса бейзлайна:", "есть" if os.path.exists(f"{MODEL_DIR}/.extract_ok") else "не нужны в этой ветке")
''')
    nb.md("## Код решения (из `solution/`) и разметка пула (метки Claude, 95% согласия с эталоном на valid)")
    nb.code("!mkdir -p /content/work/solution/scripts /content/work/solution/src/guardian")
    files = [SOL / "pyproject.toml", SOL / "scripts" / "predict.py"] + sorted(
        p for p in (SOL / "src" / "guardian").iterdir() if p.is_file() and p.suffix in {".py", ".md", ".json"})
    for p in files:
        nb.code(f"%%writefile /content/work/solution/{p.relative_to(SOL).as_posix()}\n" + p.read_text(encoding="utf-8"))
    final = ROOT / "labeling/labels_final.jsonl"  # две независимые разметки, спорные исключены (tools/label_agreement.py)
    labels = final.read_text(encoding="utf-8") if final.exists() else "".join(
        open(f, encoding="utf-8").read() for f in sorted(glob.glob(str(ROOT / "labeling/out/*.jsonl"))))
    nb.code("%%writefile /content/work/labels_pool.jsonl\n" + labels)
    nb.code(r'''
# ===== 6. Общие помощники: пул, метки, сервер, запуск predict.py =====
import pandas as pd, numpy as np, urllib.request
if os.path.exists(f"{MODEL_DIR}/.extract_ok") and not os.path.islink(f"{SOL}/model"):
    os.symlink(MODEL_DIR, f"{SOL}/model")
sys.path.insert(0, f"{SOL}/src")
for m in [k for k in list(sys.modules) if k.startswith("guardian")]: del sys.modules[m]
from guardian.devtools import build_pool, attach_labels, evaluate_sets, fit_stack, prf
from guardian.lbsim import public_from_preds
from guardian.server import start_vllm, stop_vllm, tail
from guardian.features import FEATURES
dv = pd.read_parquet(f"{WORK}/valid.parquet")
pool = build_pool(dv); pool.to_parquet(f"{WORK}/pool.parquet", index=False)
LAB = {json.loads(l)["id"]: int(json.loads(l)["label"]) for l in open(f"{WORK}/labels_pool.jsonl", encoding="utf-8") if l.strip()}
LAB.update(dict(zip(dv.id, dv.label)))
print(f"пул {len(pool)} | меток {len(LAB)} | доля позитивов в пуле {np.mean([LAB[i] for i in pool.id if i in LAB]):.2f}")

def serve(model_path, cfg_path, log_name, extra=None, lm_only=True, timeout=1800):
    cfg = json.load(open(cfg_path))
    log = f"{WORK}/{log_name}.log"
    proc = start_vllm(model_path, cfg["served_name"], cfg["port"], cfg["max_model_len"], cfg["gpu_memory_utilization"],
                      cfg["max_num_seqs"], log, (cfg.get("vllm_extra_args") or []) + (extra or []), lm_only=lm_only)
    base, t0 = f"http://127.0.0.1:{cfg['port']}/v1", time.time()
    while True:
        if proc.poll() is not None:
            print(tail(log, 80)); raise RuntimeError("vLLM упал — лог выше")
        try:
            if urllib.request.urlopen(f"{base}/models", timeout=5).status == 200: break
        except Exception: pass
        if time.time() - t0 > timeout: raise TimeoutError(tail(log, 40))
        print(f"\r[{time.time()-t0:5.0f}s] загрузка... {tail(log, 1)[-150:]}", end="", flush=True); time.sleep(10)
    print(f"\nvLLM готов за {time.time()-t0:.0f}s"); return proc, base, time.time() - t0

def run_predict(inp, tag, cfg_path, model_path, base=None, deadline=7200, extra=""):
    b = f"--base-url {base}" if base else ""
    t0 = time.time()
    get_ipython().system(f"cd {SOL} && python scripts/predict.py --input {inp} --output {RES}/{tag}_pred.parquet "
                         f"--dump {RES}/{tag}.parquet --config {cfg_path} --model-path {model_path} {b} "
                         f"--cache {CACHE}/{tag}.jsonl --deadline {deadline} {extra} 2>&1 | tee {LOGS}/{tag}.log")
    return time.time() - t0

def load_dumps(*tags):
    d = pd.concat([pd.read_parquet(f"{RES}/{t}.parquet") for t in tags]).drop_duplicates("id")
    return attach_labels(d, LAB)
''')


def summary(nb, extra=""):
    nb.code(r'''
# ===== Сводка ветки на Диск =====
summary = {"version": VERSION, "gpu": GPU_NAME, "hours": (time.time() - T_NOTEBOOK) / 3600}
summary["cu_estimate"] = summary["hours"] * CU_PER_HOUR
''' + extra + r'''
json.dump(summary, open(f"{DV}/summary.json", "w"), indent=2, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str)[:4000])
print(f"\n≈ {summary['hours']:.2f} ч ≈ {summary['cu_estimate']:.0f} CU. Пришли Claude файл {DV}/summary.json и папку results.")
''')


# ============================================================================ ветка A: v2 main
def branch_main():
    nb = NB()
    nb.md(r"""
# Guardian of Truth — ветка A (v2): промпт v2 + доказательства + проверка по предложениям + стекинг признаков

**Зачем:** v1.1 дал на паблике 0.4828. Разметка пула (384 хода, метки Claude) показала: судья слаб в тексте (выдуманные факты и правила) и ложно ругает безобидные поиски и расчёты. Стекинг признаков даёт на пуле AUC ~0.90, а ожидаемый паблик — 0.57 на признаках v1.

**Что делает ноутбук:**
1. Dev-прогон судьи v2 на valid и пуле. Фазы:
   - A — без рассуждений;
   - A_base — промпт бейзлайна;
   - A_v1 — старый промпт;
   - S — проверка по предложениям;
   - B — рассуждение, 2 сэмпла, только для ходов не из чистых чтений.
2. Подбор набора признаков и стекинга (CV по траекториям). Оценка паблика через симулятор, откалиброванный по сабмиту №1.
3. Проверка калибровки симулятора: что он предсказывает для бейзлайна (на паблике 0.4896).
4. Сборка боевого `config.json` с лучшим набором признаков и замер полного прогона «как на платформе».

**Ресурсы:** A100 80GB, High-RAM, ≈2–2.5 ч ≈ 15–20 CU.
""")
    settings(nb, "v2_main", True)
    common(nb)
    nb.code(r'''
# ===== 7. Dev-сервер (Qwen3.6-35B-A3B-FP8, веса бейзлайна) =====
DEV_CFG = f"{SOL}/src/guardian/config_dev_v2.json"
proc, BASE, LOAD_S = serve(f"{SOL}/model", DEV_CFG, "vllm_dev")
''')
    nb.code(r'''
# ===== 8. Dev-прогон v2 на valid (≈5 мин) и пуле (≈25-35 мин); кэш на Диске — при обрыве продолжит =====
T = {}
T["valid"] = run_predict(f"{WORK}/valid.parquet", "v2_valid", DEV_CFG, f"{SOL}/model", BASE)
T["pool"] = run_predict(f"{WORK}/pool.parquet", "v2_pool", DEV_CFG, f"{SOL}/model", BASE)
print(T)
''')
    nb.code(r'''
# ===== 9. Остановка dev-сервера =====
stop_vllm(proc); time.sleep(10)
''')
    nb.code(r'''
# ===== 10. Наборы признаков: стекинг (CV по траекториям) + оценка паблика симулятором =====
D = load_dumps("v2_valid", "v2_pool")
STRUCT = ["ro_only", "has_text", "has_write", "transfer", "claimed_action_no_call", "repeat_call", "missing_numbers",
          "ungrounded_values", "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
B = ["b_vfrac", "b_mean", "b_missing"]; S = ["s_max", "s_mean", "s_missing"]
SETS = {
    "полный": ["la", "la_base", "la_v1"] + B + S + STRUCT,
    "без B (без рассуждений)": ["la", "la_base", "la_v1"] + S + STRUCT,
    "без S": ["la", "la_base", "la_v1"] + B + STRUCT,
    "без доп. промптов": ["la"] + B + S + STRUCT,
    "la + B + структура (как v1)": ["la"] + B + STRUCT,
    "только A-промпты + структура": ["la", "la_base", "la_v1"] + STRUCT,
}
TAB = evaluate_sets(D, SETS, pool.id, seeds=(0, 1, 2, 3, 4))
display(TAB.round(3))
# калибровка симулятора: бейзлайн = промпт бейзлайна, порог 0.5 (на паблике F1 0.4896, P 0.37, R 0.723)
P = D[D.id.isin(set(pool.id))]
pb = np.array([json.loads(x).get("base") or 0.5 for x in P.p_extra])
sim_base = public_from_preds(P.label.values, (pb >= 0.5) | P.hard.values)
print(f"Проверка симулятора на бейзлайне: предсказано F1 {sim_base['f1']:.3f} (P {sim_base['p']:.2f} R {sim_base['r']:.2f}) | на паблике 0.490 (P 0.37 R 0.72)")
''')
    nb.code(r'''
# ===== 11. Боевой config.json: лучший набор признаков -> только нужные фазы =====
best_name = TAB.iloc[0]["набор"]; feats = SETS[best_name]
res = fit_stack(D, features_list=feats, C=0.5, seed=0)
cfg = json.load(open(DEV_CFG))
cfg.update({"version": "2.0", "decision": "stack", "stack": res["stack"], "max_pos_frac": 0.3,
            "sentence_probe": any(f.startswith("s_") for f in feats),
            "phase_b": any(f.startswith("b_") for f in feats)})
cfg["extra_a"] = {k: v for k, v in (cfg.get("extra_a") or {}).items() if f"la_{k}" in feats}
PROD_CFG = f"{SOL}/src/guardian/config.json"
json.dump(cfg, open(PROD_CFG, "w"), indent=2, ensure_ascii=False)
json.dump(cfg, open(f"{DV}/config.json", "w"), indent=2, ensure_ascii=False)
print("Лучший набор:", best_name, "| порог", res["stack"]["threshold"], "| OOF AUC", round(res["oof_auc"], 3))
print("Фазы:", {"S": cfg["sentence_probe"], "B": cfg["phase_b"], "extra_a": list(cfg["extra_a"])})
''')
    nb.code(r'''
# ===== 12. Полный прогон «как на платформе» на пуле (сам поднимает vLLM, дедлайн 25 мин) =====
t0 = time.time()
get_ipython().system(f"cd {SOL} && python scripts/predict.py --input {WORK}/pool.parquet --output {RES}/e2e_pred.parquet --dump {RES}/e2e_dump.parquet 2>&1 | tee {LOGS}/e2e.log")
WALL = time.time() - t0
e = pd.read_parquet(f"{RES}/e2e_dump.parquet")
print(f"E2E: {len(e)} примеров за {WALL:.0f}s; доля флагов {e.pred.mean():.2f}")
print("Прогноз на N=800: ~", round((LOAD_S + (WALL - LOAD_S) / len(e) * 800) / 60, 1), "мин (дедлайн 25)")
''')
    summary(nb, r'''summary.update({"best_set": best_name, "table": TAB.round(4).to_dict("records"), "baseline_sim_check": sim_base,
                "times": T, "e2e_wall_s": WALL, "server_load_s": LOAD_S})''')
    nb.save("guardian_v2_main.ipynb")


# ============================================================================ ветка B: сравнение моделей
def branch_bakeoff():
    nb = NB()
    nb.md(r"""
# Guardian of Truth — ветка B: сравнение моделей-судей на размеченном пуле

Одинаковый пайплайн v2 (фазы A, S, B) и одинаковая оценка (стекинг + симулятор паблика) для каждой модели:

| модель | вес | зачем |
|---|---|---|
| Qwen3.6-35B-A3B-FP8 | 37.5 ГБ | эталон (наша модель) |
| Qwen3.8-27B-FP8 | 30.9 ГБ | плотная, новее (авг 2026), та же архитектура qwen3_5 — vLLM 0.19.1 её знает |
| GLM-4.7-Flash (bf16) | 62.5 ГБ | семейство разметчика организаторов (GLM 5.2); только оценка качества, под архив нужен свой FP8 |
| Qwen3.6-27B-FP8 | 30.9 ГБ | опционально (`RUN_QWEN36_27B`) |

Для каждой модели считаются качество (OOF AUC, F1 пула, ожидаемый паблик) и скорость (с/пример по фазам), а по скорости — прогноз, успеет ли модель обработать N=800 за 25 минут.
**Ресурсы:** ≈3–4 ч ≈ 25–30 CU (скачивание ~130 ГБ с HF, четыре загрузки vLLM).
""")
    settings(nb, "bakeoff", False)
    common(nb)
    nb.code(r'''
# ===== 7. Список моделей =====
RUN_QWEN36_27B = False
MODELS = [
    {"name": "qwen36_35b_a3b", "repo": "Qwen/Qwen3.6-35B-A3B-FP8", "template": "qwen", "lm_only": True, "mml": 131072},
    {"name": "qwen38_27b", "repo": "Qwen/Qwen3.8-27B-FP8", "template": "qwen", "lm_only": True, "mml": 131072},
    {"name": "glm47_flash", "repo": "zai-org/GLM-4.7-Flash", "template": "hf", "lm_only": False, "mml": 65536, "seqs": 32},
] + ([{"name": "qwen36_27b", "repo": "Qwen/Qwen3.6-27B-FP8", "template": "qwen", "lm_only": True, "mml": 131072}] if RUN_QWEN36_27B else [])
base_cfg = json.load(open(f"{SOL}/src/guardian/config_dev_v2.json"))
base_cfg["extra_a"] = {}   # доп. промпты в сравнении моделей не нужны
''')
    nb.code(r'''
# ===== 8. Прогон каждой модели: скачать -> поднять vLLM -> valid + пул -> удалить веса =====
from huggingface_hub import snapshot_download
SPEED = {}
for m in MODELS:
    tag = m["name"]
    if os.path.exists(f"{RES}/{tag}_pool.parquet"):
        print("уже есть:", tag); continue
    print("=" * 30, tag, "=" * 30)
    path = snapshot_download(m["repo"], local_dir=f"/content/hf/{tag}", max_workers=16)
    cfg = dict(base_cfg, template=m["template"], language_model_only=m["lm_only"], max_model_len=m["mml"],
               max_prompt_chars=int(m["mml"] * 2.3), max_num_seqs=m.get("seqs", 128),
               concurrency_b=min(112, m.get("seqs", 128)))
    cfg_path = f"{WORK}/cfg_{tag}.json"; json.dump(cfg, open(cfg_path, "w"))
    try:
        proc, base, load_s = serve(path, cfg_path, f"vllm_{tag}", lm_only=m["lm_only"])
    except Exception as e:
        print("НЕ ЗАПУСТИЛАСЬ:", tag, e); shutil.rmtree(path, ignore_errors=True); continue
    tv = run_predict(f"{WORK}/valid.parquet", f"{tag}_valid", cfg_path, path, base)
    tp = run_predict(f"{WORK}/pool.parquet", f"{tag}_pool", cfg_path, path, base)
    stop_vllm(proc); time.sleep(10)
    d = pd.read_parquet(f"{RES}/{tag}_pool.parquet")
    SPEED[tag] = {"load_s": load_s, "pool_s": tp, "s_per_example": tp / len(d), "think_tokens_mean": float(d.think_tokens.mean())}
    json.dump(SPEED, open(f"{DV}/speed.json", "w"), indent=2)
    shutil.rmtree(path, ignore_errors=True)
print(SPEED)
''')
    nb.code(r'''
# ===== 9. Сравнение качества: один и тот же набор признаков, стекинг + симулятор =====
STRUCT = ["ro_only", "has_text", "has_write", "transfer", "claimed_action_no_call", "repeat_call", "missing_numbers",
          "ungrounded_values", "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
FEATS = {"la+B+S+структура": ["la", "b_vfrac", "b_mean", "b_missing", "s_max", "s_mean", "s_missing"] + STRUCT,
         "la+структура (без рассуждений)": ["la"] + STRUCT}
rows = []
for m in MODELS:
    tag = m["name"]
    if not os.path.exists(f"{RES}/{tag}_pool.parquet"): continue
    D = load_dumps(f"{tag}_valid", f"{tag}_pool")
    T = evaluate_sets(D, FEATS, pool.id, seeds=(0, 1, 2))
    for r in T.to_dict("records"):
        r["модель"] = tag; r.update(SPEED.get(tag, {})); rows.append(r)
CMP = pd.DataFrame(rows)
if len(CMP):
    CMP["прогноз_N800_мин"] = (CMP["load_s"] + CMP["s_per_example"] * 800) / 60
display(CMP.round(3))
''')
    summary(nb, r'''summary.update({"compare": CMP.round(4).to_dict("records"), "speed": SPEED})''')
    nb.save("guardian_bakeoff.ipynb")


# ============================================================================ ветка C: зонд скрытых состояний
def branch_probe():
    nb = NB()
    nb.md(r"""
# Guardian of Truth — ветка C: линейный зонд на скрытых состояниях судьи

**Идея (нетривиальная):** вместо того чтобы спрашивать у модели «есть ли галлюцинация» одним токеном, берём её внутреннее представление в конце промпта судьи (эмбеддинг последнего токена) и учим логистическую регрессию на наших 430 размеченных ходах. В литературе зонды бывают хрупкими при сдвиге распределения, но здесь обучение и паблик — ходы одного агента в одних доменах. На инференсе это один prefill на пример без генерации: быстро, и освобождается время на другие признаки.

1. vLLM в режиме pooling/embed (Qwen3.6-35B-A3B-FP8 из весов бейзлайна): эмбеддинги промптов судьи v2 для valid и пула.
2. Зонд и зонд + структурные признаки, CV по траекториям, оценка паблика симулятором.
3. Если vLLM не умеет pooling для этой модели, тот же эксперимент на Qwen3.5-4B через transformers.

**Ресурсы:** ≈1–1.5 ч ≈ 8–12 CU.
""")
    settings(nb, "probe", True)
    common(nb)
    nb.code(r'''
# ===== 7. Тексты промптов судьи v2 (как в фазе A) для valid + пул =====
from guardian.rules import analyze
from guardian.prompts import user_content, chat, load_text
from guardian import evidence as EV
JS = load_text("judge_system.md")
ALL = pd.concat([pool[["id", "prompt", "response"]], dv[["id", "prompt", "response"]]]).drop_duplicates("id").reset_index(drop=True)
ALL = ALL[ALL.id.isin(LAB)].reset_index(drop=True)
AN = [analyze(r.id, r.prompt, r.response) for r in ALL.itertuples()]
TXT = [chat(JS, user_content(r.prompt, r.response, a, 300000, EV.build(r.prompt, r.response)), thinking=False) for r, a in zip(ALL.itertuples(), AN)]
print(len(TXT), "промптов; средняя длина", int(np.mean([len(t) for t in TXT])), "симв.")
''')
    nb.code(r'''
# ===== 8. Эмбеддинги через vLLM pooling (последний токен) =====
import requests
from concurrent.futures import ThreadPoolExecutor
EMB = None
def try_pooling(flags):
    log = f"{WORK}/vllm_pool.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", f"{SOL}/model", "--served-model-name", "probe",
           "--port", "8013", "--max-model-len", "131072", "--gpu-memory-utilization", "0.9", "--language-model-only"] + flags
    p = subprocess.Popen(cmd, stdout=open(log, "wb"), stderr=subprocess.STDOUT); t0 = time.time()
    while time.time() - t0 < 1500:
        if p.poll() is not None: print(tail(log, 30)); return None
        try:
            if requests.get("http://127.0.0.1:8013/v1/models", timeout=5).ok: return p
        except Exception: pass
        print(f"\r[{time.time()-t0:4.0f}s] {tail(log, 1)[-120:]}", end=""); time.sleep(10)
    p.kill(); return None
proc = None
for flags in (["--runner", "pooling", "--convert", "embed"], ["--task", "embed"]):
    print("\nпробую", flags); proc = try_pooling(flags)
    if proc: break
if proc:
    def emb(t):
        r = requests.post("http://127.0.0.1:8013/v1/embeddings", json={"model": "probe", "input": t}, timeout=600)
        return r.json()["data"][0]["embedding"]
    with ThreadPoolExecutor(16) as ex:
        EMB = np.array(list(tqdm(ex.map(emb, TXT), total=len(TXT), desc="эмбеддинги")))
    np.save(f"{RES}/emb_qwen36.npy", EMB); stop_vllm(proc)
    print("\nэмбеддинги:", EMB.shape)
else:
    print("\nvLLM pooling не поднялся — переходим к запасному варианту (ячейка 9)")
''')
    nb.code(r'''
# ===== 9. Средние слои маленькой модели (Qwen3.5-9B, transformers): литература — зонд на среднем слое сильнее последнего =====
RUN_SMALL_MODEL_PROBE = True
EMB_SMALL = None
if RUN_SMALL_MODEL_PROBE:
    import torch
    import transformers
    from transformers import AutoTokenizer
    name = "Qwen/Qwen3.5-9B"
    tok = AutoTokenizer.from_pretrained(name); tok.truncation_side = "left"   # обрезаем начало, ответ и вопрос в конце
    mdl = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        try:
            mdl = getattr(transformers, cls).from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda"); print("загружено через", cls); break
        except Exception as e:
            print(cls, "не подошёл:", str(e)[:150])
    nL = mdl.config.get_text_config().num_hidden_layers if hasattr(mdl.config, "get_text_config") else mdl.config.num_hidden_layers
    LAYERS = [nL // 2, (2 * nL) // 3, nL]
    out = []
    for t in tqdm(TXT, desc="hidden states 9B"):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=48000).to("cuda")
        with torch.no_grad():
            h = mdl(**ids, output_hidden_states=True).hidden_states
        out.append(np.concatenate([h[L][0, -1].float().cpu().numpy() for L in LAYERS]))
    EMB_SMALL = np.array(out); np.save(f"{RES}/emb_qwen35_9b_mid.npy", EMB_SMALL)
    del mdl; torch.cuda.empty_cache()
    print("эмбеддинги 9B:", EMB_SMALL.shape, "слои", LAYERS)
if EMB is None and EMB_SMALL is None:
    raise RuntimeError("нет ни одного набора эмбеддингов")
''')
    nb.code(r'''
# ===== 10. Зонд: логрег на эмбеддингах (+ структура), CV по траекториям, симулятор паблика =====
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
y = np.array([LAB[i] for i in ALL.id]); hard = np.array([bool(a["hard"]) for a in AN])
traj = ALL.id.str.split("::").str[0].values
STRUCT = ["ro_only", "has_text", "n_writes", "transfer", "claimed_action_no_call", "repeat_call"]
Xs = np.array([[a["flags"].get(k, 0) for k in STRUCT] for a in AN], float)
def cv(Xfull, C, pca=64, seed=0):
    rng = np.random.RandomState(seed); ut = rng.permutation(np.unique(traj)); fold = {t: k % 5 for k, t in enumerate(ut)}
    oof = np.zeros(len(y)); m = ~hard
    for k in range(5):
        te = np.array([fold[t] == k for t in traj]) & m; tr = ~np.array([fold[t] == k for t in traj]) & m
        X = Xfull
        if pca:
            pc = PCA(n_components=min(pca, tr.sum() - 1), random_state=0).fit(X[tr]); X = pc.transform(X)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=5000).fit((X[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return oof
rows = []
pool_mask = ALL.id.isin(set(pool.id)).values
CANDS = []
if EMB is not None: CANDS += [("35B последний слой", EMB), ("35B + структура", np.c_[EMB, Xs * 10])]
if EMB_SMALL is not None:
    d = EMB_SMALL.shape[1] // 3
    CANDS += [("9B средний слой", EMB_SMALL[:, :d]), ("9B 2/3 слоя", EMB_SMALL[:, d:2 * d]), ("9B 3 слоя", EMB_SMALL),
              ("9B 3 слоя + структура", np.c_[EMB_SMALL, Xs * 10])]
for name, X in CANDS:
    for C in (0.01, 0.05, 0.2):
        for pca in (32, 128):
            oofs = [cv(X, C, pca, s) for s in range(3)]
            o = np.mean(oofs, 0); s = np.where(hard, 1.0, o)
            auc = roc_auc_score(y[~hard], o[~hard])
            best = max(public_from_preds(y[pool_mask], s[pool_mask] >= t)["f1"] for t in np.arange(0.1, 0.9, 0.05))
            rows.append({"признаки": name, "C": C, "pca": pca, "oof_auc": auc, "паблик_лучший_порог": best})
PR = pd.DataFrame(rows).sort_values("oof_auc", ascending=False); display(PR.round(3))
''')
    summary(nb, r'''summary.update({"probe": PR.round(4).to_dict("records")})''')
    nb.save("guardian_probe.ipynb")


if __name__ == "__main__":
    import sys as _s; _s.path.insert(0, str(Path(__file__).parent))
    from nb_branch_pmi import branch_pmi
    branch_main()
    branch_bakeoff()
    branch_probe()
    branch_pmi(NB, settings, common, summary)
    from nb_branch_lite import branch_lite
    branch_lite(NB, settings, common, summary)
