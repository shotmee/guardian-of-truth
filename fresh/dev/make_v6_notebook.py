"""Генерирует notebooks_extended/guardian_v6_meta.ipynb: метаданные τ² (B), примеры организаторов (C), судья GLM (D).

На размеченных ходах dev (430, из них 46 с метками организаторов) и τ² (903) считаются сигналы base / meta / shots:
сначала на Qwen3.6-35B-A3B-FP8 (как на платформе), затем на GLM-4.7-Flash FP8 (две сборки по очереди, если первая
не поднимется на vLLM 0.19.1). В конце — сравнение (dev/eval_v6.py), архив на Диск и отключение среды.
python dev/make_v6_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "dev"))
from make_prompt_check_notebook import add_bytes, call_rows, code, markdown, text_rows, unit_ids  # noqa: E402

import pandas as pd  # noqa: E402

NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v6_meta.ipynb"
WATCHDOG_HOURS = 3.0
GLM_CANDIDATES = ["unsloth/GLM-4.7-Flash-FP8-Dynamic", "mratsim/GLM-4.7-Flash-FP8"]


def pack() -> tuple:
    valid = pd.read_parquet(PROJECT / "valid.parquet")
    labels = {"text": text_rows("dev") + text_rows("tau2"), "calls": call_rows(),
              "valid": [{"id": i, "label": int(l)} for i, l in zip(valid.id, valid.label)]}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted((ROOT / "src" / "veritas").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                add_bytes(archive, path.relative_to(ROOT).as_posix(), path.read_bytes())
        for name in ("score_v6.py", "eval_v6.py", "make_pool.py"):
            add_bytes(archive, f"dev/{name}", (ROOT / "dev" / name).read_bytes())
        add_bytes(archive, "valid.parquet", (PROJECT / "valid.parquet").read_bytes())
        add_bytes(archive, "labels.json", json.dumps(labels, ensure_ascii=False).encode("utf-8"))
        add_bytes(archive, "ids_tau2.json", json.dumps(unit_ids(PROJECT / "labeling_v6" / "units.json")).encode())
    return base64.b64encode(buffer.getvalue()).decode(), {k: len(v) for k, v in labels.items()}


def build(payload: str) -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v6 — метаданные задач τ², примеры организаторов, судья GLM

**Что считаем** (только на размеченных ходах: dev 430, из них 46 с метками организаторов; τ² 903) — три сигнала:
- `base` — промпт бейзлайна организаторов (как P_base в решении);
- `meta` — тот же промпт + **метаданные задачи τ²** (сценарий клиента, эталонные действия; это видел разметчик);
- `shots` — тот же промпт + **примеры разметки организаторов** из valid (23 ошибки с объяснениями + 23 не-ошибки).

Сначала на **Qwen3.6-35B-A3B-FP8** (как на платформе), потом на **GLM-4.7-Flash FP8** (семейство разметчика):
заодно проверяем, поднимается ли GLM на vLLM 0.19.1 — без этого в решение его не взять.

**Как запускать:** Colab, **A100 80GB** или H100 (на A100 40GB модель не помещается — ноутбук это проверит в шаге 1 и сразу отключится), High-RAM → «Среда выполнения → Выполнить все». Всё пишется на Диск
(`MyDrive/guardian_of_truth/v6_meta/`), повторный запуск досчитывает недостающее; **в конце среда отключится сама**
(сторожевой таймер — через {WATCHDOG_HOURS:g} ч). Нужен `MyDrive/guardian_of_truth/v5_tau2gen/examples_v5.parquet`;
секрет `HF_TOKEN` желателен.

**Оценка:** ≈ 1.5–2 ч ≈ **12–16 CU**. Итог — шаг 9 и `summary.txt`; архив для Claude —
`MyDrive/guardian_of_truth/v6_meta/results_v6_meta.zip`.
"""),
        code(f"""
# 1. Диск, папки, GPU + сторожевой таймер ({WATCHDOG_HOURS:g} ч)
import os, sys, json, time, subprocess, threading, shutil
from pathlib import Path
from google.colab import drive, runtime
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v6_meta'); (OUT / 'signals').mkdir(parents=True, exist_ok=True)
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
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True).stdout.strip()
note('старт; GPU: ' + gpu)
if int(gpu.split(',')[-1]) < 75000:      # веса Qwen 35B FP8 ≈ 35 ГБ + кэш: на A100 40GB сервер не поднимается
    note('GPU меньше 80 ГБ — модель не поместится. Отключаю среду, чтобы не тратить CU; перезапусти на A100 80GB или H100')
    drive.flush_and_unmount()
    runtime.unassign()
    raise SystemExit('нужна видеокарта на 80 ГБ')
note('диалоги τ² на Диске: ' + ('есть' if GENERATED.exists() else 'НЕТ — будет только dev'))
"""),
        code("""
# 2. vLLM 0.19.1 — как на платформе  (⏱ 3–6 мин)
import importlib.metadata as md
try:
    if md.version('vllm') != '0.19.1':
        raise md.PackageNotFoundError
except md.PackageNotFoundError:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'vllm==0.19.1'], check=False)
for package in ('vllm', 'transformers', 'torch'):
    try:
        note(f'{package} {md.version(package)}')
    except Exception as error:
        note(f'{package}: {error!r}')
"""),
        code("""
# 3. Модель Qwen3.6-35B-A3B-FP8  (⏱ 2–10 мин)
from huggingface_hub import snapshot_download
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception as error:
    note(f'HF_TOKEN не найден, качаю без него: {error}')
def download(repo, target):
    # скачать модель в отдельном процессе без кэша кусков Hugging Face (xet): он удваивает место на диске
    target = Path(target)
    if (target / '.complete').exists():
        return True
    script = (f"from huggingface_hub import snapshot_download; snapshot_download({repo!r}, local_dir={str(target)!r}, "
              "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', 'merges.txt'])")
    result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'HF_HUB_DISABLE_XET': '1'},
                            capture_output=True, text=True)
    shutil.rmtree(Path.home() / '.cache' / 'huggingface' / 'xet', ignore_errors=True)
    if result.returncode != 0:
        note(f'{repo} не скачалась:\\n{result.stderr[-1500:]}')
        shutil.rmtree(target, ignore_errors=True)
        return False
    (target / '.complete').write_text('ok')
    return True

QWEN = Path('/content/models/qwen')
note('Qwen готова' if download('Qwen/Qwen3.6-35B-A3B-FP8', QWEN) else 'Qwen НЕ скачалась')
note(subprocess.run('df -h /content | tail -1', shell=True, capture_output=True, text=True).stdout.strip())
"""),
        code(f"""
# 4. Код, метки, наборы из размеченных ходов  (⏱ <1 мин)
import base64, io, tarfile
import pandas as pd
PAYLOAD_B64 = '{payload}'
WORK = Path('/content/v6'); WORK.mkdir(exist_ok=True)
INPUTS = {{}}
try:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD_B64)), mode='r:gz') as archive:
        archive.extractall(WORK)
    subprocess.run([sys.executable, 'dev/make_pool.py', '--valid', 'valid.parquet', '--out', 'data'],
                   cwd=WORK, check=True, capture_output=True, text=True)
    INPUTS['dev'] = WORK / 'data' / 'dev.parquet'
    if GENERATED.exists():
        wanted = set(json.loads((WORK / 'ids_tau2.json').read_text()))
        frame = pd.read_parquet(GENERATED)
        frame = frame[frame['id'].isin(wanted)][['id', 'prompt', 'response']].reset_index(drop=True)
        INPUTS['tau2'] = WORK / 'tau2_labeled.parquet'
        frame.to_parquet(INPUTS['tau2'], index=False)
    note('наборы: ' + ', '.join(f'{{k}} {{len(pd.read_parquet(v))}}' for k, v in INPUTS.items()))
except Exception as error:
    note(f'подготовка не удалась: {{error!r}}')
"""),
        code("""
# Общие функции: сервер vLLM, прогон сигналов, сколько уже посчитано
import urllib.request, re

def coverage(model):
    # (все сигналы посчитаны ≥ 98%, строки отчёта) по jsonl-чекпоинтам на Диске
    lines, ok = [], True
    for name, path in INPUTS.items():
        rows = len(pd.read_parquet(path))
        for signal in ('base', 'meta', 'shots'):
            f = OUT / 'signals' / f'{name}_{model}_{signal}.jsonl'
            k = sum(1 for line in open(f, encoding='utf-8') if line.strip()) if f.exists() else 0
            lines.append(f'{name}/{signal}: {k}/{rows}')
            ok = ok and k >= 0.98 * rows
    return ok, lines

def stop_servers():
    subprocess.run('pkill -f "vllm.entrypoints.openai.api_server" || true', shell=True)
    time.sleep(15)
    subprocess.run('pkill -9 -f EngineCore || true', shell=True)       # движок, если пережил свой сервер
    time.sleep(3)

def start_server(model_dir, extra, log_name, timeout=1500):
    stop_servers()
    log = open(f'/content/{log_name}.log', 'w')
    process = subprocess.Popen([sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model_dir),
                                '--served-model-name', 'm', '--port', '8000', '--max-model-len', '131072',
                                '--gpu-memory-utilization', '0.90', '--max-num-seqs', '128', '--enable-prefix-caching',
                                *extra], stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    while time.time() - started < timeout and process.poll() is None:
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)
            note(f'{log_name}: сервер готов за {time.time() - started:.0f} с')
            return True
        except Exception:
            time.sleep(5)
    lines = open(f'/content/{log_name}.log', encoding='utf-8', errors='replace').read().splitlines()
    first = next((i for i, l in enumerate(lines) if re.search(r'\\w+Error\\b|Traceback|out of memory', l)
                  and 'Warning' not in l), max(0, len(lines) - 30))
    cause = '\\n'.join(lines[first:first + 30])
    note(f'{log_name}: сервер НЕ поднялся (код {process.poll()}). Первая ошибка:\\n{cause}')
    shutil.copy(f'/content/{log_name}.log', OUT / f'{log_name}.log')
    return False

def run_signals(model):
    for name, path in INPUTS.items():
        t = time.time()
        result = subprocess.run([sys.executable, 'dev/score_v6.py', '--input', str(path), '--name', name, '--model', model,
                                 '--base-url', 'http://127.0.0.1:8000/v1', '--out', str(OUT / 'signals')],
                                cwd=WORK, capture_output=True, text=True)
        lines = [l for l in (result.stdout + result.stderr).splitlines() if l.strip() and '|' not in l]
        note(f'{model}/{name}: код {result.returncode}, {time.time() - t:.0f} с\\n' + '\\n'.join(lines[-8:]))
"""),
        code("""
# 5–6. Qwen: сервер и сигналы base / meta / shots  (⏱ 5 мин + 20–30 мин)
try:
    if start_server(QWEN, ['--language-model-only'], 'vllm_qwen'):
        run_signals('qwen')
    qwen_ok, lines = coverage('qwen')
    note('Qwen посчитан: ' + ', '.join(lines))
except Exception as error:
    note(f'Qwen: {error!r}')
"""),
        code(f"""
# 7–8. GLM-4.7-Flash FP8: скачать, поднять на vLLM 0.19.1, сигналы base / meta / shots  (⏱ 10 мин + 20–30 мин)
GLM_CANDIDATES = {GLM_CANDIDATES!r}
glm_ok = False
stop_servers()
if coverage('qwen')[0]:
    shutil.rmtree(QWEN, ignore_errors=True)          # сигналы Qwen на Диске — веса больше не нужны
    note('веса Qwen удалены, место под GLM')
for repo in GLM_CANDIDATES:
    try:
        free_gb = shutil.disk_usage('/content').free / 1e9
        if free_gb < 40:
            note(f'GLM {{repo}}: на диске {{free_gb:.0f}} ГБ — мало, пропускаю'); continue
        target = Path('/content/models') / repo.replace('/', '__')
        if not download(repo, target):
            continue
        note(f'GLM {{repo}} скачана')
        if start_server(target, [], 'vllm_glm_' + repo.split('/')[0]):
            glm_ok = True
            (OUT / 'glm_model.txt').write_text(repo)
            run_signals('glm')
            break
        shutil.rmtree(target, ignore_errors=True)    # не поднялась — освобождаем место под следующую сборку
    except Exception as error:
        note(f'GLM {{repo}}: {{error!r}}')
note('GLM на vLLM 0.19.1: ' + ('ПОДНЯЛАСЬ' if glm_ok else 'не поднялась ни одна сборка'))
stop_servers()
"""),
        code("""
# 9. Сравнение  (⏱ 1–3 мин)
try:
    result = subprocess.run([sys.executable, 'dev/eval_v6.py', '--labels', 'labels.json', '--signals', str(OUT / 'signals'),
                             '--out', str(OUT / 'summary.json')], cwd=WORK, capture_output=True, text=True)
    print(result.stdout); print(result.stderr[-3000:])
    (OUT / 'summary.txt').write_text(result.stdout + '\\n' + result.stderr[-3000:], encoding='utf-8')
except Exception as error:
    note(f'сравнение не удалось: {error!r}')
"""),
        code("""
# 10. Архив результатов и отключение среды
try:
    import zipfile
    with zipfile.ZipFile(OUT / 'results_v6_meta.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.parquet', '.log'):
                archive.write(path, path.relative_to(OUT).as_posix())
    note('архив: ' + str(OUT / 'results_v6_meta.zip'))
except Exception as error:
    note(f'архив не собран: {error!r}')
note('готово, отключаю среду')
try:
    stop_servers()
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
    payload, stats = pack()
    NOTEBOOK.write_text(json.dumps(build(payload), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{NOTEBOOK} ({NOTEBOOK.stat().st_size / 1e6:.1f} МБ); метки: {stats}")


if __name__ == "__main__":
    main()
