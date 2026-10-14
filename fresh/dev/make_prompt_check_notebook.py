"""Генерирует notebooks_extended/guardian_v5d_prompts.ipynb: прежние формулировки промптов против переписанных.

Два кода решения прогоняются один в один как на платформе (scripts/predict.py --dump, конфиг H1, общий vLLM):
  old — код и промпты архива H1 до переписывания (собирается из fresh, как build_zip);
  new — код и промпты из текущего архива D:/downloads/guardian_v5_H1_final.zip (переписан пользователем 08.10).
Запросы к модели у них различаются только тремя файлами промптов (проверено побайтно на dev, 2198 запросов).
Наборы — только размеченные ходы: dev (текст 188 + hard 37) и τ² (текст 623 + вызовы 264 + hard 16).
Сравнение и вердикт — dev/prompt_eval.py (метки, веса и прежние оценки упакованы в labels.json).
python dev/make_prompt_check_notebook.py [--new-archive D:/downloads/guardian_v5_H1_final.zip]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "dev"))
import build_zip  # noqa: E402
import eval_text_rankers as etr  # noqa: E402

NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v5d_prompts.ipynb"
WATCHDOG_HOURS = 1.5


def text_rows(name: str) -> list:
    """Текстовые ходы набора с весами под состав паблика (рейкинг по P_base и группам; петли τ² — вес 0)."""
    import pandas as pd
    frame = etr.add_combinations(etr.load_text_frame(name))
    kept = frame[~frame.loop].reset_index(drop=True)
    weights = dict(zip(kept.id, etr.rake_weights(kept)))
    rows = []
    for row in frame.itertuples():
        rows.append({"id": row.id, "set": name, "dialog": row.dialog, "label": int(row.label), "group": row.group,
                     "loop": bool(row.loop), "weight": float(weights.get(row.id, 0.0)),
                     "shift": float(etr.GROUP_LOGIT_SHIFT.get(row.group, 0.0)),
                     "p_base_prev": float(row.p_base), "atoms_prev": float(row.atoms_mean),
                     "category": None if pd.isna(row.category) else str(row.category), "domain": row.domain})
    return rows


def call_rows() -> list:
    import pandas as pd
    paths = etr.SETS["tau2"]
    units = pd.DataFrame(json.loads(paths["units"].read_text(encoding="utf-8")))
    labels = etr.load_labels(paths["labels"])
    signals = pd.read_parquet(paths["signals"])[["id", "calls_score"]]
    frame = units[units.kind == "call"].merge(labels, on="id").merge(signals, on="id", how="left")
    return [{"id": row.id, "dialog": row.dialog, "label": int(row.label),
             "calls_prev": None if pd.isna(row.calls_score) else float(row.calls_score)} for row in frame.itertuples()]


def unit_ids(units_path: Path) -> list:
    return [unit["id"] for unit in json.loads(units_path.read_text(encoding="utf-8"))]


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    archive.addfile(info, io.BytesIO(data))


def pack(new_archive: Path) -> tuple:
    labels = {"text": text_rows("dev") + text_rows("tau2"), "calls": call_rows(),
              "group_shift": etr.GROUP_LOGIT_SHIFT}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data, mode in build_zip.code_files("H1_final"):          # old: ровно код архива до переписывания
            add_bytes(archive, f"code_old/{name}", data, mode)
        with zipfile.ZipFile(new_archive) as source:                       # new: код из текущего архива
            for info in source.infolist():
                if not info.filename.startswith("model/") and not info.is_dir():
                    mode = 0o755 if info.filename.startswith("scripts/") else 0o644
                    add_bytes(archive, f"code_new/{info.filename}", source.read(info), mode)
        add_bytes(archive, "code_old/dev/make_pool.py", (ROOT / "dev" / "make_pool.py").read_bytes())
        add_bytes(archive, "prompt_eval.py", (ROOT / "dev" / "prompt_eval.py").read_bytes())
        add_bytes(archive, "valid.parquet", (PROJECT / "valid.parquet").read_bytes())
        add_bytes(archive, "labels.json", json.dumps(labels, ensure_ascii=False).encode("utf-8"))
        add_bytes(archive, "ids_dev.json", json.dumps(unit_ids(PROJECT / "labeling_v5" / "units.json")).encode())
        add_bytes(archive, "ids_tau2.json", json.dumps(unit_ids(PROJECT / "labeling_v6" / "units.json")).encode())
    stats = {"text": len(labels["text"]), "calls": len(labels["calls"])}
    return base64.b64encode(buffer.getvalue()).decode(), stats


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def build(payload: str) -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v5d — прежние формулировки промптов против переписанных

**Что сравниваем.** В архивах H1/H2 переписаны три системных промпта: `baseline.md` (P_base, нужен H1 и H2),
`text_standard.md` (текстовые атомы, только H1) и `standard.md` (атомы вызовов). Остальной запрос к модели у
двух версий совпадает побайтно. Ноутбук прогоняет **оба кода решения как на платформе** (`scripts/predict.py`,
конфиг H1, общий vLLM) на размеченных ходах и сравнивает, какие формулировки лучше находят ошибки:
- dev: 188 текстовых ходов (+37 hard); τ²: 623 текстовых (петли не учитываются) + 264 хода с вызовами.

**Правило решения (записано до прогона)** — для каждого файла промпта по главному набору τ²:
разница new − old (бутстрэп по диалогам, 5–95%) **целиком ниже нуля → новый хуже, возвращаем прежний**;
целиком выше нуля → новый лучше; иначе разница в пределах шума — можно оставить новый.
Для масштаба печатается и шум повторного прогона тех же прежних промптов (FP8).

**Как запускать:** Colab, **A100 80GB**, High-RAM → «Среда выполнения → Выполнить все». Всё сохраняется на Диск
(`MyDrive/guardian_of_truth/v5d_prompts/`), повторный запуск досчитывает только недостающее. **В конце среда
отключится сама** (сторожевой таймер — через {WATCHDOG_HOURS} ч в любом случае). Нужен файл
`MyDrive/guardian_of_truth/v5_tau2gen/examples_v5.parquet` (он уже есть после v5_tau2gen); секрет `HF_TOKEN` желателен.

**Оценка:** ≈ 35–45 мин ≈ **5–6 CU**. Итог печатается в шаге 7 и лежит в `summary.txt`;
архив для Claude — `MyDrive/guardian_of_truth/v5d_prompts/results_v5d_prompts.zip`.
"""),
        code(f"""
# 1. Диск, папки, GPU + сторожевой таймер (отключит среду через {WATCHDOG_HOURS} ч, если ноутбук не дойдёт до конца)
import os, sys, json, time, subprocess, threading
from pathlib import Path
from google.colab import drive, runtime
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v5d_prompts'); OUT.mkdir(parents=True, exist_ok=True)
GENERATED = Path('/content/drive/MyDrive/guardian_of_truth/v5_tau2gen/examples_v5.parquet')
LOG = OUT / 'run_log.txt'

def note(message):
    line = time.strftime('%H:%M:%S ') + str(message)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\\n')

def watchdog(seconds):
    time.sleep(seconds)
    note('сторожевой таймер: отключаю среду')
    try:
        drive.flush_and_unmount()
    finally:
        runtime.unassign()

threading.Thread(target=watchdog, args=({WATCHDOG_HOURS} * 3600,), daemon=True).start()
note('старт; GPU: ' + subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                     capture_output=True, text=True).stdout.strip())
note('диалоги τ² на Диске: ' + ('есть' if GENERATED.exists() else 'НЕТ — сравнение будет только на dev'))
"""),
        code("""
# 2. vLLM 0.19.1 — как на платформе  (⏱ 3–6 мин)
import importlib.metadata as md
try:
    vllm_version = md.version('vllm')
except md.PackageNotFoundError:
    vllm_version = None
try:
    if vllm_version != '0.19.1':
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'vllm==0.19.1'], check=False)
    note('vllm ' + md.version('vllm'))
except Exception as error:
    note(f'установка vLLM: {error!r}')
"""),
        code("""
# 3. Модель Qwen3.6-35B-A3B-FP8  (⏱ 2–10 мин)
from huggingface_hub import snapshot_download
MODEL = Path('/content/models/user')
try:
    if not (MODEL / '.complete').exists():
        try:
            from google.colab import userdata
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
        except Exception as error:
            note(f'HF_TOKEN не найден, качаю без него: {error}')
        snapshot_download('Qwen/Qwen3.6-35B-A3B-FP8', local_dir=str(MODEL),
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'merges.txt'])
        (MODEL / '.complete').write_text('ok')
    note('модель готова')
except Exception as error:
    note(f'модель не скачалась: {error!r}')
"""),
        code(f"""
# 4. Два кода решения (old / new), метки, наборы из размеченных ходов  (⏱ <1 мин)
import base64, io, tarfile
import pandas as pd
PAYLOAD_B64 = '{payload}'
WORK = Path('/content/v5d'); WORK.mkdir(exist_ok=True)
INPUTS = {{}}
try:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD_B64)), mode='r:gz') as archive:
        archive.extractall(WORK)
    subprocess.run([sys.executable, 'code_old/dev/make_pool.py', '--valid', 'valid.parquet', '--out', 'data'],
                   cwd=WORK, check=True, capture_output=True, text=True)
    sources = {{'dev': WORK / 'data' / 'dev.parquet', 'tau2': GENERATED}}
    for name, source in sources.items():
        if not Path(source).exists():
            continue
        wanted = set(json.loads((WORK / f'ids_{{name}}.json').read_text()))
        frame = pd.read_parquet(source)
        frame = frame[frame['id'].isin(wanted)][['id', 'prompt', 'response']].reset_index(drop=True)
        INPUTS[name] = WORK / f'{{name}}_labeled.parquet'
        frame.to_parquet(INPUTS[name], index=False)
        note(f'{{name}}: размеченных ходов {{len(frame)}} из {{len(wanted)}}')
    for variant in ('old', 'new'):
        config = json.loads((WORK / f'code_{{variant}}' / 'src' / 'veritas' / 'config.json').read_text())
        note(f'код {{variant}}: стратегия {{config["strategy"]}}, ранжировщик {{config["text_ranker"]}}')
except Exception as error:
    note(f'подготовка не удалась: {{error!r}}')
"""),
        code("""
# 5. Сервер vLLM  (⏱ 4–6 мин)
import urllib.request
subprocess.run('pkill -f "vllm.entrypoints.openai.api_server" || true', shell=True)
time.sleep(5)
server_log = open('/content/vllm_judge.log', 'w')
JUDGE, ready = None, False
try:
    JUDGE = subprocess.Popen([sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(MODEL),
                              '--served-model-name', 'm', '--port', '8000', '--max-model-len', '131072',
                              '--gpu-memory-utilization', '0.90', '--max-num-seqs', '128', '--enable-prefix-caching',
                              '--language-model-only'], stdout=server_log, stderr=subprocess.STDOUT)
    started = time.time()
    while time.time() - started < 1500 and JUDGE.poll() is None:
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5); ready = True; break
        except Exception:
            time.sleep(5)
    note(f'vLLM готов: {ready}, {time.time() - started:.0f} с')
    if not ready:
        note(open('/content/vllm_judge.log').read()[-3000:])
except Exception as error:
    note(f'vLLM не запустился: {error!r}')
"""),
        code("""
# 6. Прогоны: dev old → new, τ² new → old (порядок чередуется)  (⏱ dev 2×2 мин, τ² 2×9 мин)
ORDER = [('dev', 'old'), ('dev', 'new'), ('tau2', 'new'), ('tau2', 'old')]
for name, variant in ORDER:
    dump = OUT / f'dump_{name}_{variant}.json'
    if not ready or name not in INPUTS:
        continue
    if dump.exists():
        note(f'{name}/{variant}: уже посчитано'); continue
    t = time.time()
    try:
        result = subprocess.run([sys.executable, f'code_{variant}/scripts/predict.py', '--input', str(INPUTS[name]),
                                 '--output', str(WORK / f'pred_{name}_{variant}.parquet'),
                                 '--base-url', 'http://127.0.0.1:8000/v1', '--dump', str(WORK / f'dump_{name}_{variant}.json')],
                                cwd=WORK, capture_output=True, text=True)
        lines = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
        note(f'{name}/{variant}: код {result.returncode}, {time.time() - t:.0f} с')
        note('\\n'.join(lines[-6:]))
        if result.returncode == 0:
            dump.write_text((WORK / f'dump_{name}_{variant}.json').read_text(encoding='utf-8'), encoding='utf-8')
    except Exception as error:
        note(f'{name}/{variant}: {error!r}')
"""),
        code("""
# 7. Сравнение и вердикт  (⏱ 1–2 мин)
try:
    result = subprocess.run([sys.executable, 'prompt_eval.py', '--labels', 'labels.json', '--dumps', str(OUT),
                             '--out', str(OUT / 'summary.json')], cwd=WORK, capture_output=True, text=True)
    print(result.stdout); print(result.stderr[-3000:])
    (OUT / 'summary.txt').write_text(result.stdout + '\\n' + result.stderr[-3000:], encoding='utf-8')
    note('итог сохранён: ' + str(OUT / 'summary.txt'))
except Exception as error:
    note(f'сравнение не удалось: {error!r}')
"""),
        code("""
# 8. Архив результатов на Диск и отключение среды  (⏱ <1 мин)
try:
    import shutil, zipfile
    with zipfile.ZipFile(OUT / 'results_v5d_prompts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.glob('*')):
            if path.suffix in ('.json', '.txt'):
                archive.write(path, path.name)
    note('архив: ' + str(OUT / 'results_v5d_prompts.zip'))
except Exception as error:
    note(f'архив не собран: {error!r}')
note('готово, отключаю среду')
try:
    if JUDGE is not None:
        JUDGE.terminate()
    drive.flush_and_unmount()
finally:
    runtime.unassign()
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm",
                                                                         "gpuType": "A100"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--new-archive", default="D:/downloads/guardian_v5_H1_final.zip")
    parser.add_argument("--out", default=str(NOTEBOOK))
    args = parser.parse_args()
    payload, stats = pack(Path(args.new_archive))
    out = Path(args.out)
    out.write_text(json.dumps(build(payload), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} МБ); меток: текст {stats['text']}, вызовы {stats['calls']}")


if __name__ == "__main__":
    main()
