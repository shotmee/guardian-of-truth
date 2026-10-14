"""Генерирует notebooks_extended/guardian_v6_glm_debug.ipynb: почему GLM-4.7-Flash не отвечал и сколько он стоит.

1) Поднимает GLM-4.7-Flash FP8 на vLLM 0.19.1 и по шагам проверяет запросы: chat без logprobs → с logprobs →
   наш прежний запрос → длинный реальный пример → /v1/completions с нашей отрисовкой шаблона GLM (как в решении).
   Лог сервера сохраняется на Диск всегда. Если сервер умер на запросе — перезапуск с --enforce-eager.
2) Если хоть один путь работает, считает сигналы base и shots на dev (430) и τ² (903) и сравнивает с Qwen из v6_meta.
python dev/make_glm_debug_notebook.py
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
from make_prompt_check_notebook import add_bytes, code, markdown, unit_ids  # noqa: E402
from make_v6_notebook import pack as pack_v6  # noqa: E402

NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v6_glm_debug.ipynb"
WATCHDOG_HOURS = 1.5
GLM_REPO = "unsloth/GLM-4.7-Flash-FP8-Dynamic"


def build(payload: str) -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v6 — отладка GLM-4.7-Flash (судья из семейства разметчика)

В прошлом прогоне GLM поднялся на vLLM 0.19.1, но **все запросы вернули ошибку** (0 оценок, лог не сохранился).
Этот ноутбук:
1. поднимает GLM и **по шагам** проверяет запросы — от простейшего chat до нашего пути `/v1/completions`
   с отрисовкой шаблона GLM (так работает решение на платформе); лог сервера сохраняется на Диск всегда;
2. если путь найден — считает `base` (промпт организаторов) и `shots` (+ примеры организаторов) на dev и τ²
   и сравнивает с Qwen из прошлого прогона.

**Как запускать:** Colab, **A100 80GB или H100**, «Выполнить все». На карте меньше 80 ГБ ноутбук сразу отключится.
**Оценка:** ≈ 30–40 мин ≈ **4–5 CU**. В конце среда отключится сама (сторожевой таймер — {WATCHDOG_HOURS:g} ч).
**Результат:** вывод шагов 6 и 8 и архив `MyDrive/guardian_of_truth/v6_glm/results_v6_glm.zip` — пришли Claude.
"""),
        code(f"""
# 1. Диск, папки, GPU (нужно 80 ГБ) + сторожевой таймер
import os, sys, json, time, subprocess, threading, shutil, re
from pathlib import Path
from google.colab import drive, runtime
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v6_glm'); (OUT / 'signals').mkdir(parents=True, exist_ok=True)
QWEN_SIGNALS = Path('/content/drive/MyDrive/guardian_of_truth/v6_meta/signals')
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
note('сигналы Qwen для сравнения: ' + ('есть' if (QWEN_SIGNALS / 'tau2_qwen_signals.parquet').exists() else 'НЕТ'))
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
        code(f"""
# 3. GLM-4.7-Flash FP8 (без кэша кусков Hugging Face)  (⏱ 3–5 мин)
GLM = Path('/content/models/glm')
def download(repo, target):
    if (target / '.complete').exists():
        return True
    script = (f"from huggingface_hub import snapshot_download; snapshot_download({{repo!r}}, local_dir={{str(target)!r}}, "
              "allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', 'merges.txt'])")
    result = subprocess.run([sys.executable, '-c', script], env={{**os.environ, 'HF_HUB_DISABLE_XET': '1'}},
                            capture_output=True, text=True)
    if result.returncode != 0:
        note(f'{{repo}} не скачалась:\\n{{result.stderr[-1500:]}}'); return False
    (target / '.complete').write_text('ok'); return True
note('GLM готова' if download({GLM_REPO!r}, GLM) else 'GLM НЕ скачалась')
"""),
        code(f"""
# 4. Код, метки, наборы  (⏱ <1 мин)
import base64, io, tarfile
import pandas as pd
PAYLOAD_B64 = '{payload}'
WORK = Path('/content/v6glm'); WORK.mkdir(exist_ok=True)
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
# 5. Сервер и пробные запросы
import urllib.request, urllib.error
sys.path.insert(0, str(WORK / 'src'))
from veritas.chat import ChatTemplate, judge_prompt
from veritas import baseline
URL = 'http://127.0.0.1:8000/v1'
SERVER = None

def stop_servers():
    subprocess.run('pkill -f "vllm.entrypoints.openai.api_server" || true', shell=True); time.sleep(15)
    subprocess.run('pkill -9 -f EngineCore || true', shell=True); time.sleep(3)

def save_log(name):
    shutil.copy(f'/content/{name}.log', OUT / f'{name}.log')
    lines = open(f'/content/{name}.log', encoding='utf-8', errors='replace').read().splitlines()
    errors = [i for i, l in enumerate(lines) if re.search(r'\\w+Error\\b|Traceback|out of memory', l) and 'Warning' not in l]
    return '\\n'.join(lines[errors[0]:errors[0] + 30]) if errors else '(ошибок в логе нет)'

def start_server(extra, name, timeout=1500):
    global SERVER
    stop_servers()
    log = open(f'/content/{name}.log', 'w')
    SERVER = subprocess.Popen([sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(GLM),
                               '--served-model-name', 'm', '--port', '8000', '--max-model-len', '131072',
                               '--gpu-memory-utilization', '0.90', '--max-num-seqs', '128', '--enable-prefix-caching',
                               *extra], stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    while time.time() - started < timeout and SERVER.poll() is None:
        try:
            urllib.request.urlopen(URL + '/models', timeout=5); note(f'{name}: сервер готов за {time.time() - started:.0f} с'); return True
        except Exception:
            time.sleep(5)
    note(f'{name}: сервер НЕ поднялся (код {SERVER.poll()}). Первая ошибка:\\n' + save_log(name)); return False

def alive():
    try:
        urllib.request.urlopen(URL + '/models', timeout=5); return True
    except Exception:
        return False

def post(path, body, timeout=600):
    request = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode('utf-8', 'replace')
    except Exception as error:
        return None, repr(error)

TEMPLATE = ChatTemplate(GLM / 'chat_template.jinja')
SYSTEM = (WORK / 'src' / 'veritas' / 'prompts' / 'baseline.md').read_text(encoding='utf-8').strip()
dev = pd.read_parquet(INPUTS['dev'])
short = dev.iloc[(dev.prompt.str.len()).argsort().iloc[0]]
long = dev.iloc[(dev.prompt.str.len()).argsort().iloc[len(dev) // 2]]

def probes():
    results = {}
    msg = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': baseline.user_message(short.prompt, short.response)}]
    msg_long = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': baseline.user_message(long.prompt, long.response)}]
    tests = {
        'chat, без logprobs': ('/chat/completions', {'model': 'm', 'messages': [{'role': 'user', 'content': 'Ответь одной цифрой: 1'}], 'max_tokens': 1}),
        'chat + logprobs': ('/chat/completions', {'model': 'm', 'messages': [{'role': 'user', 'content': 'Ответь одной цифрой: 1'}], 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 20}),
        'chat, прежний запрос (короткий пример)': ('/chat/completions', {'model': 'm', 'messages': msg, 'max_tokens': 1, 'temperature': 0.0, 'logprobs': True, 'top_logprobs': 20, 'seed': 1, 'chat_template_kwargs': {'enable_thinking': False}}),
        'chat, прежний запрос (средний пример)': ('/chat/completions', {'model': 'm', 'messages': msg_long, 'max_tokens': 1, 'temperature': 0.0, 'logprobs': True, 'top_logprobs': 20, 'seed': 1, 'chat_template_kwargs': {'enable_thinking': False}}),
        'completions, наш шаблон (средний пример)': ('/completions', {'model': 'm', 'prompt': judge_prompt(TEMPLATE, SYSTEM, baseline.user_message(long.prompt, long.response), thinking=False), 'max_tokens': 1, 'temperature': 0.0, 'logprobs': 20, 'seed': 1, 'skip_special_tokens': False}),
    }
    for name, (path, body) in tests.items():
        status, text = post(path, body)
        results[name] = status == 200
        note(f'проба «{name}»: {status} | {text[:400]}')
        if not alive():
            note('СЕРВЕР УМЕР после этой пробы'); break
    return results
"""),
        code("""
# 6. Диагностика: пробы на обычном сервере, при падении — с --enforce-eager
API, EXTRA = None, []
for extra, name in (([], 'vllm_glm'), (['--enforce-eager'], 'vllm_glm_eager')):
    if not start_server(extra, name):
        continue
    result = probes()
    note(f'{name}: первая ошибка в логе сервера:\\n' + save_log(name))
    if result.get('completions, наш шаблон (средний пример)'):
        API, EXTRA = 'completions', extra; break
    if result.get('chat, прежний запрос (средний пример)'):
        API, EXTRA = 'chat', extra; break
note(f'ИТОГ ДИАГНОСТИКИ: рабочий путь — {API or "НЕ НАЙДЕН"}, флаги сервера {EXTRA}')
(OUT / 'diagnosis.json').write_text(json.dumps({'api': API, 'extra': EXTRA}), encoding='utf-8')
"""),
        code("""
# 7. Сигналы GLM base / shots на dev и τ² (если путь найден)  (⏱ 15–20 мин)
if API:
    if not alive():
        start_server(EXTRA, 'vllm_glm_run')
    for name, path in INPUTS.items():
        t = time.time()
        command = [sys.executable, 'dev/score_v6.py', '--input', str(path), '--name', name, '--model', 'glm', '--api', API,
                   '--base-url', URL, '--out', str(OUT / 'signals'), '--signals', 'base,shots']
        if API == 'completions':
            command += ['--template', str(GLM / 'chat_template.jinja')]
        result = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
        lines = [l for l in (result.stdout + result.stderr).splitlines() if l.strip() and '|' not in l]
        note(f'glm/{name}: код {result.returncode}, {time.time() - t:.0f} с\\n' + '\\n'.join(lines[-6:]))
    save_log('vllm_glm_run' if Path('/content/vllm_glm_run.log').exists() else ('vllm_glm' if not EXTRA else 'vllm_glm_eager'))
stop_servers()
"""),
        code("""
# 8. Сравнение с Qwen (сигналы Qwen — из прогона v6_meta)
for path in QWEN_SIGNALS.glob('*_qwen_signals.parquet'):
    shutil.copy(path, OUT / 'signals' / path.name)
result = subprocess.run([sys.executable, 'dev/eval_v6.py', '--labels', 'labels.json', '--signals', str(OUT / 'signals'),
                         '--out', str(OUT / 'summary.json')], cwd=WORK, capture_output=True, text=True)
print(result.stdout); print(result.stderr[-3000:])
(OUT / 'summary.txt').write_text(result.stdout + '\\n' + result.stderr[-3000:], encoding='utf-8')
"""),
        code("""
# 9. Архив и отключение
import zipfile
with zipfile.ZipFile(OUT / 'results_v6_glm.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.parquet', '.log'):
            archive.write(path, path.relative_to(OUT).as_posix())
note('архив: ' + str(OUT / 'results_v6_glm.zip') + '; отключаю среду')
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
    payload, stats = pack_v6()
    NOTEBOOK.write_text(json.dumps(build(payload), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{NOTEBOOK} ({NOTEBOOK.stat().st_size / 1e6:.1f} МБ); метки: {stats}")


if __name__ == "__main__":
    main()
