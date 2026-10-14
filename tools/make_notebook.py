"""Генерирует Colab-ноутбук notebooks_extended/guardian_<ver>.ipynb из файлов solution/ (единый источник кода).
Запуск: python tools/make_notebook.py v1"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solution"
VER = sys.argv[1] if len(sys.argv) > 1 else "v1"
OUT = ROOT / "notebooks_extended" / f"guardian_{VER}.ipynb"

cells = []


def md(src: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src: str):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(rf"""
# Guardian of Truth — решение {VER}

**Что делает ноутбук** (шаги идут сверху вниз, долгие шаги сохраняют результат и пропускаются при повторном запуске):
1. Проверка железа → Google Drive → установка vLLM (версия как в среде проверки).
2. Загрузка весов бейзлайна (Qwen3.6-35B-A3B-FP8) и `valid.parquet` с Kaggle → распаковка с прогрессом.
3. Код решения (`%%writefile`): правила + LLM-судья с рассуждением + слияние + тайм-гард.
4. Реплика бейзлайна на valid → оценка доли позитивов в паблике.
5. Наше решение на valid (46) и на пуле промежуточных ходов (~385) → метрики, подбор порога → `config.json`.
6. Полный прогон `predict.py` «как на платформе» (с запуском vLLM) → замер времени.
7. Сохранение `config.json`, дампов и сводки на Диск.

**Среда:** Runtime → Change runtime type → **A100 GPU (80 ГБ) или H100**, High-RAM. GPU 40 ГБ не подходит: веса FP8 занимают ~35 ГБ.
**Ресурсы:** ~1.5–2.5 ч рантайма ≈ 12–20 CU на A100-80GB (≈30–40 CU на H100). Обучения весов в {VER} нет.
""")

code(r'''
# ===== НАСТРОЙКИ =====
VERSION = "''' + VER + r'''"
KAGGLE_DATASET = "shotme/aij-minecraft-kulbit"
DRIVE_ROOT = "/content/drive/MyDrive/guardian_of_truth"
WORK = "/content/work"
SOL = f"{WORK}/solution"
MODEL_DIR = f"{WORK}/model"
VLLM_VERSION = "0.19.1"        # по словам участников, стоит в среде проверки
DELETE_PARTS_AFTER_EXTRACT = True   # освобождает ~30 ГБ диска после распаковки
RUN_BASELINE_REPLICA = True    # реплика бейзлайна на valid -> оценка доли позитивов в паблике
RUN_POOL = True                # прогон на ~385 промежуточных ходах (скорость + доля позитивов)
RUN_E2E = True                 # полный прогон predict.py с запуском vLLM (как на платформе)
BUILD_FULL_ZIP_IN_COLAB = False  # True — собрать полный архив (~35 ГБ) прямо на Диск
import time; T_NOTEBOOK = time.time()
''')

code(r'''
# ===== 1. Железо =====
import os, shutil, subprocess
q = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).decode().strip()
GPU_NAME, GPU_MEM = q.split(", ")[0], int(q.split(", ")[1])
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
DISK_GB = shutil.disk_usage("/content").free / 2**30
print(f"GPU: {GPU_NAME}, {GPU_MEM/1024:.0f} ГБ | RAM: {RAM_GB:.0f} ГБ | свободно на диске: {DISK_GB:.0f} ГБ")
assert GPU_MEM >= 70000, "Нужна GPU на 80 ГБ (A100 80GB или H100): модель FP8 ~35 ГБ + длинные контексты до ~80k токенов"
if RAM_GB < 40: print("⚠️ Мало RAM: включи High-RAM (vLLM грузит 35 ГБ весов через RAM)")
if DISK_GB < 80: print("⚠️ Мало диска: нужно ~70 ГБ (30 ГБ архив + 35 ГБ веса)")
IS_H100 = "H100" in GPU_NAME
A100_FACTOR = 2.2 if IS_H100 else 1.0   # грубый пересчёт времени H100 -> A100 (на платформе A100)
CU_PER_HOUR = 15.0 if IS_H100 else (7.5 if GPU_MEM > 60000 else 5.4)   # ориентировочно, точнее см. в Colab
''')

code(r'''
# ===== 2. Google Drive: папки версии =====
from google.colab import drive
drive.mount("/content/drive")
DV = f"{DRIVE_ROOT}/{VERSION}"
RES, LOGS, CACHE = f"{DV}/results", f"{DV}/logs", f"{DV}/cache"
for d in (RES, LOGS, CACHE, WORK):
    os.makedirs(d, exist_ok=True)
print("Результаты версии будут здесь:", DV)
''')

code(r'''
# ===== 3. Установка vLLM (≈3-6 мин) =====
import subprocess, sys
def pip(*a):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a]).returncode
if pip(f"vllm=={VLLM_VERSION}", "mistral_common>=1.11.5") != 0:
    print(f"⚠️ vllm=={VLLM_VERSION} не встал, ставлю последнюю версию")
    pip("vllm", "mistral_common>=1.11.5")
pip("kagglehub", "pyarrow", "pandas", "openai", "tqdm")
print(subprocess.check_output([sys.executable, "-c",
      "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"]).decode())
''')

code(r'''
# ===== 4. Данные с Kaggle (≈30 ГБ; повторный запуск берёт из кэша) =====
import os, glob, kagglehub
try:
    from google.colab import userdata
    os.environ.setdefault("KAGGLE_USERNAME", userdata.get("KAGGLE_USERNAME"))
    os.environ.setdefault("KAGGLE_KEY", userdata.get("KAGGLE_KEY"))
except Exception:
    print("Секреты KAGGLE_USERNAME/KAGGLE_KEY не заданы — пробую без них (датасет публичный)")
FILES = [f"baseline.tar.gz.0{i}" for i in range(7)] + ["valid.parquet"]
PATHS = {}
if os.path.exists(f"{MODEL_DIR}/.extract_ok"):
    FILES = ["valid.parquet"]
try:
    for f in FILES:
        PATHS[f] = kagglehub.dataset_download(KAGGLE_DATASET, path=f)
        print("✓", f, PATHS[f])
except Exception as e:
    print("Пофайловая загрузка не удалась (", e, ") — качаю датасет целиком")
    root = kagglehub.dataset_download(KAGGLE_DATASET)
    for f in FILES:
        PATHS[f] = glob.glob(f"{root}/**/{f}", recursive=True)[0]
shutil.copy(PATHS["valid.parquet"], f"{WORK}/valid.parquet")
print("valid.parquet ->", f"{WORK}/valid.parquet")
''')

code(r'''
# ===== 5. Распаковка весов из baseline.tar.gz.* (чекпоинт: .extract_ok) =====
import io, tarfile, time
from tqdm.auto import tqdm
if os.path.exists(f"{MODEL_DIR}/.extract_ok"):
    print("Веса уже распакованы:", MODEL_DIR)
else:
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
                if n:
                    bar.update(n); return n
                self.f.close()
                if DELETE_PARTS_AFTER_EXTRACT: os.remove(self.paths[self.k])
                self.f, self.k = None, self.k + 1
    os.makedirs(MODEL_DIR, exist_ok=True)
    with tarfile.open(fileobj=io.BufferedReader(Concat(parts), 1 << 24), mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith("baseline/model/"):
                dst = os.path.join(MODEL_DIR, m.name[len("baseline/model/"):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tf.extractfile(m) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 24)
    bar.close()
    open(f"{MODEL_DIR}/.extract_ok", "w").write("ok")
print(sorted(os.listdir(MODEL_DIR))[:8], "...", f"{sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR)) / 2**30:.1f} ГБ")
''')

md("## Код решения (генерируется из `solution/`, совпадает с архивом сабмита)")
code("!mkdir -p /content/work/solution/scripts /content/work/solution/src/guardian")
files = [SOL / "pyproject.toml", SOL / "scripts" / "predict.py"] + sorted(
    p for p in (SOL / "src" / "guardian").iterdir() if p.is_file() and p.suffix in {".py", ".md", ".json"})
for p in files:
    rel = p.relative_to(SOL).as_posix()
    code(f"%%writefile /content/work/solution/{rel}\n" + p.read_text(encoding="utf-8"))

code(r'''
# ===== 6. Сборка рабочей папки + проверка правил на valid (CPU) =====
import sys, json, pandas as pd, numpy as np
if not os.path.islink(f"{SOL}/model"):
    os.symlink(MODEL_DIR, f"{SOL}/model")
sys.path.insert(0, f"{SOL}/src")
from guardian.rules import analyze
from guardian.devtools import build_pool, prf, neg_weight, tune, tune_gate, replay_labels
CFG = f"{SOL}/src/guardian/config.json"
dv = pd.read_parquet(f"{WORK}/valid.parquet")
hard = np.array([bool(analyze(r.id, r.prompt, r.response)["hard"]) for r in dv.itertuples()])
print(f"valid: {len(dv)} примеров, позитивов {dv.label.sum()} | hard-правила: {hard.sum()} срабатываний, "
      f"из них верных {int(dv.label[hard].sum())}")
pool = build_pool(dv)
pool.to_parquet(f"{WORK}/pool.parquet", index=False)
print(f"пул промежуточных ходов: {len(pool)} примеров (для замера скорости и доли позитивов)")
''')

code(r'''
# ===== 7. Dev-сервер vLLM (один раз на все dev-прогоны; ≈3-8 мин на загрузку) =====
import time, urllib.request
from guardian.server import start_vllm, tail, stop_vllm
cfg = json.load(open(CFG))
DEV_LOG = f"{WORK}/vllm_dev.log"
t0 = time.time()
dev_proc = start_vllm(f"{SOL}/model", cfg["served_name"], cfg["port"], cfg["max_model_len"],
                      cfg["gpu_memory_utilization"], cfg["max_num_seqs"], DEV_LOG, cfg.get("vllm_extra_args"))
BASE = f"http://127.0.0.1:{cfg['port']}/v1"
while True:
    if dev_proc.poll() is not None:
        print(tail(DEV_LOG, 80)); raise RuntimeError("vLLM упал — лог выше")
    try:
        if urllib.request.urlopen(f"{BASE}/models", timeout=5).status == 200: break
    except Exception:
        pass
    last = tail(DEV_LOG, 1)[-160:]
    print(f"\r[{time.time()-t0:5.0f}s] загрузка... {last}", end="", flush=True)
    time.sleep(10)
SERVER_LOAD_S = time.time() - t0
print(f"\nvLLM готов за {SERVER_LOAD_S:.0f}s")
''')

code(r'''
# ===== 8. Реплика бейзлайна на valid -> оценка доли позитивов в паблике =====
# На паблике бейзлайн: TP=94 из 130, FP=160. Его FPR на valid-негативах даёт N негативов паблика.
from guardian.devtools import estimate_public_prevalence
PI_EST = {}
if RUN_BASELINE_REPLICA:
    !cd {SOL} && python scripts/predict.py --mode baseline --input {WORK}/valid.parquet --output {RES}/valid_baseline_pred.parquet --dump {RES}/valid_baseline.parquet --base-url {BASE} --cache {CACHE}/valid_baseline.jsonl 2>&1
    b = pd.read_parquet(f"{RES}/valid_baseline.parquet")
    pb = (b.p_nothink.fillna(0) >= 0.5).astype(int)
    r = prf(b.label, pb)
    fpr = ((pb == 1) & (b.label == 0)).sum() / (b.label == 0).sum()
    est = estimate_public_prevalence(fpr)
    PI_EST["baseline"] = est["pi_public"]
    print(f"Реплика бейзлайна на valid: F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} (на паблике R=0.723) | FPR={fpr:.2f}")
    print(f"=> оценка паблика: N≈{est['n_public']:.0f}, доля позитивов π≈{est['pi_public']:.2f}")
''')

code(r'''
# ===== 9. Наше решение на valid (dev-режим: LLM и по hard-примерам, кэш на Диске) =====
t0 = time.time()
!cd {SOL} && python scripts/predict.py --input {WORK}/valid.parquet --output {RES}/valid_pred.parquet --dump {RES}/valid_judge.parquet --base-url {BASE} --cache {CACHE}/valid_judge.jsonl --llm-on-hard 2>&1
print(f"valid: {time.time()-t0:.0f}s")
d = pd.read_parquet(f"{RES}/valid_judge.parquet")
y = d.label.values
PI = float(np.median(list(PI_EST.values()))) if PI_EST else 0.27
def show(name, pred):
    r, r2 = prf(y, pred), prf(y, pred, neg_weight(y, PI))
    print(f"{name:34s} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} | F1 при π={PI:.2f}: {r2['f1']:.3f}")
H = d.hard.values
show("только hard-правила", H.astype(int))
show("фаза A (без рассуждений) >=0.5", ((d.p_nothink.fillna(0) >= 0.5) | H).astype(int))
show("фаза B (рассуждение) >=0.5", ((d.p_think.fillna(d.p_nothink).fillna(0) >= 0.5) | H).astype(int))
# итог считаем повтором решения по дампу без потолка доли позитивов (на valid 50/50 потолок искусственно режет)
cfg = json.load(open(CFG))
d["pred_nocap"] = replay_labels(d, cfg, cap=False)
show(f"итог: правило {cfg.get('decision', 'fused')}", d.pred_nocap.values)
d["domain"] = d.id.str.split("__").str[0]
print(d.assign(ok=(d.pred_nocap == d.label)).groupby("domain").agg(n=("id", "size"), pos=("label", "sum"), pred_pos=("pred_nocap", "sum"), acc=("ok", "mean")))
d[d.pred_nocap != d.label][["id", "label", "pred", "p_nothink", "p_think", "hard_reasons", "reasoning"]].to_csv(f"{RES}/valid_errors.csv", index=False)
print("Ошибки с рассуждениями модели сохранены:", f"{RES}/valid_errors.csv")
''')

code(r'''
# ===== 10. Пул промежуточных ходов: скорость под нагрузкой + доля позитивов (≈10-25 мин) =====
if RUN_POOL:
    t0 = time.time()
    !cd {SOL} && python scripts/predict.py --input {WORK}/pool.parquet --output {RES}/pool_pred.parquet --dump {RES}/pool_judge.parquet --base-url {BASE} --cache {CACHE}/pool_judge.jsonl --deadline 7200 2>&1
    POOL_S = time.time() - t0
    pd_ = pd.read_parquet(f"{RES}/pool_judge.parquet")
    rate = float(replay_labels(pd_, json.load(open(CFG)), cap=False).mean())
    PI_EST["pool_flag_rate"] = rate
    print(f"пул: {len(pd_)} примеров за {POOL_S:.0f}s ({POOL_S/len(pd_):.2f} c/пример при уже поднятом сервере)")
    print(f"доля позитивов по модели на пуле: {rate:.2f}; средняя длина рассуждения: {pd_.think_tokens.mean():.0f} ток.")
''')

code(r'''
# ===== 11. Подбор gate_pa (правило «голос рассуждения 1 И P_A >= gate_pa») -> config.json =====
# Доля позитивов в паблике по оценкам ~0.15 (реплика бейзлайна, пересчёт флагов пула). Флаги на пуле
# (тестоподобные ходы) не должны сильно превышать 0.2-0.25, иначе precision на паблике рухнет.
PI = float(np.clip(PI_EST.get("baseline", 0.15), 0.1, 0.3))
pis = [round(PI * k, 3) for k in (0.8, 1.0, 1.33)]
pool_d = pd.read_parquet(f"{RES}/pool_judge.parquet") if RUN_POOL else None
cfg = json.load(open(CFG))
g = tune_gate(d, pool_d, cfg, pis=pis)
display(g.round(3))
best = g.iloc[0]
cfg.update({"decision": "gate", "gate_pa": float(best.gate_pa), "b_min_pa": float(best.gate_pa)})
json.dump(cfg, open(CFG, "w"), indent=2, ensure_ascii=False)
json.dump(cfg, open(f"{DV}/config.json", "w"), indent=2, ensure_ascii=False)
print(f"Выбрано: gate_pa={best.gate_pa} | TP={best.tp:.0f} FP={best.fp:.0f} на valid | F1 при π-сценариях={best.f1_shifted:.3f} | флаги на пуле={best.pool_flag_rate:.2f}")
print("config.json сохранён в решение и на Диск:", f"{DV}/config.json")
''')

code(r'''
# ===== 12. Остановка dev-сервера =====
stop_vllm(dev_proc)
time.sleep(10)
print(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]).decode())
''')

code(r'''
# ===== 13. Полный прогон «как на платформе»: predict.py сам поднимает vLLM (≈15-30 мин) =====
E2E = {}
if RUN_E2E:
    t0 = time.time()
    !cd {SOL} && python scripts/predict.py --input {WORK}/pool.parquet --output {RES}/e2e_pred.parquet --dump {RES}/e2e_dump.parquet 2>&1 | tee {LOGS}/e2e.log
    wall = time.time() - t0
    e = pd.read_parquet(f"{RES}/e2e_dump.parquet")
    cover_b = float(e.p_think.notna().mean())
    per_ex = wall / len(e)
    E2E = {"n": len(e), "wall_s": wall, "think_coverage": cover_b, "a100_factor": A100_FACTOR}
    print(f"E2E: {len(e)} примеров за {wall:.0f}s; рассуждение успело для {cover_b:.0%} примеров")
    for n in (300, 500, 800, 1200):
        print(f"  прогноз на A100 для N={n}: ~{(SERVER_LOAD_S + (wall - SERVER_LOAD_S) / len(e) * n) * A100_FACTOR / 60:.1f} мин "
              f"(дедлайн решения {cfg['hard_deadline_s']/60:.0f} мин — дальше тайм-гард режет фазу B)")
''')

code(r'''
# ===== 14. Сводка версии на Диск =====
summary = {"version": VERSION, "gpu": GPU_NAME, "config": json.load(open(CFG)), "pi_estimates": PI_EST,
           "server_load_s": SERVER_LOAD_S, "e2e": E2E,
           "valid": {k: prf(y, v) for k, v in {"final": d.pred.values, "hard_only": d.hard.astype(int).values}.items()},
           "notebook_hours": (time.time() - T_NOTEBOOK) / 3600}
summary["cu_estimate"] = summary["notebook_hours"] * CU_PER_HOUR
json.dump(summary, open(f"{DV}/summary.json", "w"), indent=2, ensure_ascii=False, default=float)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=float)[:3000])
print(f"\n≈ {summary['notebook_hours']:.2f} ч рантайма ≈ {summary['cu_estimate']:.0f} CU")
print("Дальше: скачай с Диска", f"{DV}/config.json", "-> положи в solution/src/guardian/ и собери архив tools/build_submission.py")
''')

code(r'''
# ===== 15. (Опционально) полный архив сабмита прямо в Colab -> Диск (~35 ГБ, нужно место на Диске) =====
if BUILD_FULL_ZIP_IN_COLAB:
    stage = f"{WORK}/submission"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(SOL, stage, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "model"))
    os.symlink(MODEL_DIR, f"{stage}/model")
    !cd {stage} && chmod -R a+rX . && zip -r -0 -q {WORK}/guardian_{VERSION}.zip pyproject.toml scripts src model/ -x "model/.extract_ok" && ls -lh {WORK}/guardian_{VERSION}.zip
    shutil.copy(f"{WORK}/guardian_{VERSION}.zip", f"{DV}/guardian_{VERSION}.zip")
    print("Архив на Диске:", f"{DV}/guardian_{VERSION}.zip")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"-> {OUT} ({len(cells)} ячеек)")
