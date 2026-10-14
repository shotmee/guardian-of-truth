"""Генерирует notebooks_extended/guardian_v6_policy.ipynb: фокусная проверка регламента (v6p) на Qwen3.6.

Сигналы на размеченных ходах dev (430) и τ² (903): slice (фрагмент политики + последняя реплика клиента + ответ),
window (фрагмент + последние сообщения диалога с результатами инструментов + ответ), focus (полный промпт организаторов
+ фрагмент перед ответом). Сравнение — dev/eval_policy.py, правило выбора записано в его описании.
python dev/make_policy_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "dev"))
from make_prompt_check_notebook import add_bytes, call_rows, code, markdown, text_rows, unit_ids  # noqa: E402

NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v6_policy.ipynb"
WATCHDOG_HOURS = 1.5


def pack() -> tuple:
    valid = pd.read_parquet(PROJECT / "valid.parquet")
    labels = {"text": text_rows("dev") + text_rows("tau2"), "calls": call_rows(),
              "valid": [{"id": i, "label": int(l)} for i, l in zip(valid.id, valid.label)]}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted((ROOT / "src" / "veritas").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                add_bytes(archive, path.relative_to(ROOT).as_posix(), path.read_bytes())
        for name in ("score_v6.py", "eval_v6.py", "eval_policy.py", "make_pool.py"):
            add_bytes(archive, f"dev/{name}", (ROOT / "dev" / name).read_bytes())
        add_bytes(archive, "valid.parquet", (PROJECT / "valid.parquet").read_bytes())
        add_bytes(archive, "labels.json", json.dumps(labels, ensure_ascii=False).encode("utf-8"))
        add_bytes(archive, "ids_tau2.json", json.dumps(unit_ids(PROJECT / "labeling_v6" / "units.json")).encode())
    return base64.b64encode(buffer.getvalue()).decode(), {k: len(v) for k, v in labels.items()}


def build(payload: str) -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v6p — фокусная проверка регламента (фрагмент политики)

**Гипотеза:** на длинном промпте (до 230 тыс. символов) модель теряет точечные правила политики; короткий запрос
с нужным фрагментом политики поймает процессные ошибки, которые P_base пропускает (зона P_base < 0.5).
Три варианта на Qwen3.6-35B-A3B-FP8 (как на платформе), на размеченных ходах dev (430) и τ² (903):
- `slice` — фрагмент политики + последняя реплика клиента + ответ агента (≈ 4 тыс. символов);
- `window` — фрагмент + последние сообщения диалога с результатами инструментов + ответ (≈ 10 тыс.);
- `focus` — полный промпт организаторов, но перед ответом агента ещё раз нужный фрагмент политики.

**Правило (записано до прогона):** вариант берём, если на τ² «H1 + вариант» даёт P@160 ≥ ×1.05 к H1 с нижней
границей бутстрэпа > ×1.0 и на dev не меньше ×1.0; или если на вызовах прирост AUC надёжно выше нуля и ≥ +2 ошибки
в топ-20%. Иначе не берём.

**Как запускать:** Colab, **A100 80GB или H100**, «Выполнить все» (на карте меньше 80 ГБ ноутбук сразу отключится).
**Оценка:** ≈ 30–40 мин ≈ **4–5 CU**. В конце среда отключится сама (сторожевой таймер — {WATCHDOG_HOURS:g} ч).
**Результат:** вывод шага 6 и архив `MyDrive/guardian_of_truth/v6_policy/results_v6_policy.zip` — пришли Claude.
"""),
        code(f"""
# 1. Диск, папки, GPU (нужно 80 ГБ) + сторожевой таймер
import os, sys, json, time, subprocess, threading, shutil, re
from pathlib import Path
from google.colab import drive, runtime
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v6_policy'); (OUT / 'signals').mkdir(parents=True, exist_ok=True)
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
if int(gpu.split(',')[-1]) < 75000:
    note('GPU меньше 80 ГБ — отключаю среду; перезапусти на A100 80GB или H100')
    drive.flush_and_unmount(); runtime.unassign()
    raise SystemExit('нужна видеокарта на 80 ГБ')
note('диалоги τ² на Диске: ' + ('есть' if GENERATED.exists() else 'НЕТ — будет только dev'))
"""),
        code("""
# 2. vLLM 0.19.1  (⏱ 3–6 мин)
import importlib.metadata as md
try:
    if md.version('vllm') != '0.19.1':
        raise md.PackageNotFoundError
except md.PackageNotFoundError:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'vllm==0.19.1'], check=False)
for package in ('vllm', 'transformers', 'torch'):
    note(f'{package} {md.version(package)}')
"""),
        code("""
# 3. Qwen3.6-35B-A3B-FP8 (без кэша кусков Hugging Face)  (⏱ 2–5 мин)
QWEN = Path('/content/models/qwen')
def download(repo, target):
    if (target / '.complete').exists():
        return True
    script = (f"from huggingface_hub import snapshot_download; snapshot_download({repo!r}, local_dir={str(target)!r}, "
              "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'merges.txt'])")
    result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'HF_HUB_DISABLE_XET': '1'},
                            capture_output=True, text=True)
    if result.returncode != 0:
        note(f'{repo} не скачалась:\\n{result.stderr[-1500:]}'); return False
    (target / '.complete').write_text('ok'); return True
note('Qwen готова' if download('Qwen/Qwen3.6-35B-A3B-FP8', QWEN) else 'Qwen НЕ скачалась')
"""),
        code(f"""
# 4. Код, метки, наборы  (⏱ <1 мин)
import base64, io, tarfile
import pandas as pd
PAYLOAD_B64 = '{payload}'
WORK = Path('/content/v6p'); WORK.mkdir(exist_ok=True)
INPUTS = {{}}
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD_B64)), mode='r:gz') as archive:
    archive.extractall(WORK)
subprocess.run([sys.executable, 'dev/make_pool.py', '--valid', 'valid.parquet', '--out', 'data'], cwd=WORK, check=True,
               capture_output=True, text=True)
INPUTS['dev'] = WORK / 'data' / 'dev.parquet'
if GENERATED.exists():
    wanted = set(json.loads((WORK / 'ids_tau2.json').read_text()))
    frame = pd.read_parquet(GENERATED)
    frame = frame[frame['id'].isin(wanted)][['id', 'prompt', 'response']].reset_index(drop=True)
    INPUTS['tau2'] = WORK / 'tau2_labeled.parquet'; frame.to_parquet(INPUTS['tau2'], index=False)
note('наборы: ' + ', '.join(f'{{k}} {{len(pd.read_parquet(v))}}' for k, v in INPUTS.items()))
"""),
        code("""
# 5. Сервер Qwen и сигналы slice / window / focus  (⏱ 5 мин + 15–20 мин)
import urllib.request

def stop_servers():
    subprocess.run('pkill -f "vllm.entrypoints.openai.api_server" || true', shell=True); time.sleep(15)
    subprocess.run('pkill -9 -f EngineCore || true', shell=True); time.sleep(3)

def start_server(timeout=1500):
    stop_servers()
    log = open('/content/vllm_qwen.log', 'w')
    process = subprocess.Popen([sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(QWEN),
                                '--served-model-name', 'm', '--port', '8000', '--max-model-len', '131072',
                                '--gpu-memory-utilization', '0.90', '--max-num-seqs', '128', '--enable-prefix-caching',
                                '--language-model-only'], stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    while time.time() - started < timeout and process.poll() is None:
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)
            note(f'сервер готов за {time.time() - started:.0f} с'); return True
        except Exception:
            time.sleep(5)
    shutil.copy('/content/vllm_qwen.log', OUT / 'vllm_qwen.log')
    lines = open('/content/vllm_qwen.log', encoding='utf-8', errors='replace').read().splitlines()
    first = next((i for i, l in enumerate(lines) if re.search(r'\\w+Error\\b|Traceback|out of memory', l) and 'Warning' not in l),
                 max(0, len(lines) - 30))
    note('сервер НЕ поднялся. Первая ошибка:\\n' + '\\n'.join(lines[first:first + 30])); return False

if start_server():
    for name, path in INPUTS.items():
        t = time.time()
        result = subprocess.run([sys.executable, 'dev/score_v6.py', '--input', str(path), '--name', name, '--model', 'qwen',
                                 '--base-url', 'http://127.0.0.1:8000/v1', '--out', str(OUT / 'signals'),
                                 '--signals', 'slice,window,focus'], cwd=WORK, capture_output=True, text=True)
        lines = [l for l in (result.stdout + result.stderr).splitlines() if l.strip() and '|' not in l]
        note(f'qwen/{name}: код {result.returncode}, {time.time() - t:.0f} с\\n' + '\\n'.join(lines[-6:]))
stop_servers()
"""),
        code("""
# 6. Сравнение и вердикт  (⏱ 1–2 мин)
result = subprocess.run([sys.executable, 'dev/eval_policy.py', '--labels', 'labels.json', '--signals', str(OUT / 'signals'),
                         '--out', str(OUT / 'summary.json')], cwd=WORK, capture_output=True, text=True)
print(result.stdout); print(result.stderr[-3000:])
(OUT / 'summary.txt').write_text(result.stdout + '\\n' + result.stderr[-3000:], encoding='utf-8')
"""),
        code("""
# 7. Архив и отключение
import zipfile
with zipfile.ZipFile(OUT / 'results_v6_policy.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.parquet', '.log'):
            archive.write(path, path.relative_to(OUT).as_posix())
note('архив: ' + str(OUT / 'results_v6_policy.zip') + '; отключаю среду')
try:
    stop_servers(); drive.flush_and_unmount()
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
