"""Генерирует notebooks_extended/guardian_v6_glm2.ipynb: два обхода падения GLM-4.7-Flash на A100 и оценка GLM.

Причина падения (v6_glm): на запросах длиннее одного куска предзаполнения vLLM 0.19.1 идёт в
mla_attention._compute_prefill_context, где FP8-слой kv_b_proj считается ядром Marlin и падает
(«unsupported `a` scalar_type»). Обходы:
  A — та же FP8-сборка, но без разбиения предзаполнения и без кэша префиксов (этот путь не вызывается);
  B — «разжать» только kv_b_proj в bf16 во всех слоях (вес FP8 × масштаб канала) и добавить его в ignore
      квантования; остальная модель остаётся FP8, размер растёт на ~0.2 ГБ, кэш префиксов работает.
Каждый обход проверяется пробами (короткий / средний / длинный реальный запрос); рабочий (B важнее) используется
для сигналов base и shots на dev и τ² с замером скорости, затем сравнение с Qwen из v6_meta.
python dev/make_glm2_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "dev"))
from make_prompt_check_notebook import code, markdown  # noqa: E402
from make_v6_notebook import pack as pack_v6  # noqa: E402

NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v6_glm2.ipynb"
WATCHDOG_HOURS = 2.0
GLM_REPO = "unsloth/GLM-4.7-Flash-FP8-Dynamic"


def build(payload: str) -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v6 — GLM-4.7-Flash: два обхода падения на A100 и оценка судьи

Прошлый прогон нашёл причину: на запросах длиннее 2048 токенов vLLM 0.19.1 на A100 падает в FP8-слое `kv_b_proj`
механизма MLA (`unsupported 'a' scalar_type`). Здесь проверяются два обхода:
- **A** — та же FP8-сборка, но без разбиения предзаполнения и без кэша префиксов (сломанный путь не вызывается;
  без кэша медленнее — скорость замеряется);
- **B** — слой `kv_b_proj` «разжат» в bf16 во всех слоях (вес FP8 × масштаб), остальное FP8; кэш префиксов работает.

Рабочий обход (B важнее) используется для сигналов `base` и `shots` на dev и τ²; затем сравнение с Qwen.

**Как запускать:** Colab, **A100 80GB** (именно A100 — как на платформе), «Выполнить все». На карте меньше 80 ГБ
ноутбук сразу отключится. **Оценка:** ≈ 50–70 мин ≈ **7–9 CU**. В конце среда отключится сама
(сторожевой таймер — {WATCHDOG_HOURS:g} ч). **Результат:** вывод шагов 7 и 9 и архив
`MyDrive/guardian_of_truth/v6_glm2/results_v6_glm2.zip` — пришли Claude.
"""),
        code(f"""
# 1. Диск, папки, GPU (нужно 80 ГБ) + сторожевой таймер
import os, sys, json, time, subprocess, threading, shutil, re
from pathlib import Path
from google.colab import drive, runtime
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v6_glm2'); (OUT / 'signals').mkdir(parents=True, exist_ok=True)
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
    note('GPU меньше 80 ГБ — отключаю среду; перезапусти на A100 80GB')
    drive.flush_and_unmount(); runtime.unassign()
    raise SystemExit('нужна видеокарта на 80 ГБ')
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
# 3. GLM-4.7-Flash FP8 (без кэша кусков Hugging Face)  (⏱ 3–8 мин)
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
WORK = Path('/content/v6glm2'); WORK.mkdir(exist_ok=True)
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
# 5. Сервер (флаги задаются целиком) и пробные запросы
import urllib.request, urllib.error
sys.path.insert(0, str(WORK / 'src'))
from veritas.chat import ChatTemplate, judge_prompt
from veritas import baseline
URL = 'http://127.0.0.1:8000/v1'
NORMAL_FLAGS = ['--enable-prefix-caching']                       # как в решении
NO_CHUNK_FLAGS = ['--no-enable-prefix-caching', '--no-enable-chunked-prefill', '--max-num-batched-tokens', '131072']

def stop_servers():
    subprocess.run('pkill -f "vllm.entrypoints.openai.api_server" || true', shell=True); time.sleep(15)
    subprocess.run('pkill -9 -f EngineCore || true', shell=True); time.sleep(3)

def save_log(name):
    shutil.copy(f'/content/{name}.log', OUT / f'{name}.log')
    lines = open(f'/content/{name}.log', encoding='utf-8', errors='replace').read().splitlines()
    errors = [i for i, l in enumerate(lines) if re.search(r'\\w+Error\\b|Traceback|out of memory', l) and 'Warning' not in l]
    return '\\n'.join(lines[errors[0]:errors[0] + 40]) if errors else '(ошибок в логе нет)'

def start_server(model_dir, flags, name, timeout=1500):
    stop_servers()
    log = open(f'/content/{name}.log', 'w')
    process = subprocess.Popen([sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model_dir),
                                '--served-model-name', 'm', '--port', '8000', '--max-model-len', '131072',
                                '--gpu-memory-utilization', '0.90', '--max-num-seqs', '128', *flags],
                               stdout=log, stderr=subprocess.STDOUT)
    started = time.time()
    while time.time() - started < timeout and process.poll() is None:
        try:
            urllib.request.urlopen(URL + '/models', timeout=5); note(f'{name}: сервер готов за {time.time() - started:.0f} с'); return True
        except Exception:
            time.sleep(5)
    note(f'{name}: сервер НЕ поднялся (код {process.poll()}). Первая ошибка:\\n' + save_log(name)); return False

def alive():
    try:
        urllib.request.urlopen(URL + '/models', timeout=5); return True
    except Exception:
        return False

def post(path, body, timeout=900):
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
by_length = dev.iloc[dev.prompt.str.len().argsort().values]
EXAMPLES = {'короткий': by_length.iloc[0], 'средний': by_length.iloc[len(dev) // 2], 'длинный': by_length.iloc[int(len(dev) * 0.95)]}

def probes(name, log_name):
    results = {}
    for label, row in EXAMPLES.items():
        body = {'model': 'm', 'prompt': judge_prompt(TEMPLATE, SYSTEM, baseline.user_message(row.prompt, row.response), thinking=False),
                'max_tokens': 1, 'temperature': 0.0, 'logprobs': 20, 'seed': 1, 'skip_special_tokens': False}
        started = time.time()
        status, text = post('/completions', body)
        results[label] = status == 200
        note(f'{name}: проба «{label}» ({len(row.prompt)} симв.): {status}, {time.time() - started:.1f} с | {text[:300]}')
        if not alive():
            note(f'{name}: СЕРВЕР УМЕР после пробы «{label}»'); break
    note(f'{name}: первая ошибка в логе сервера:\\n' + save_log(log_name))
    return results
"""),
        code("""
# 6. Обход B: «разжать» kv_b_proj в bf16 во всех слоях (из FP8-сборки, без полного переквантования)  (⏱ 8–12 мин)
import torch
from safetensors import safe_open
from safetensors.torch import save_file
GLM_B = Path('/content/models/glm_kvbf16')

def build_kv_bf16(src, dst):
    if (dst / '.complete').exists():
        return True
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix != '.safetensors' and path.name not in ('model.safetensors.index.json', '.complete'):
            shutil.copy(path, dst / path.name)
    config = json.loads((src / 'config.json').read_text())
    quant = config['quantization_config']
    ignore = set(quant.get('ignore', []))
    for i in range(config['num_hidden_layers']):
        ignore.add(f'model.layers.{i}.self_attn.kv_b_proj')
    quant['ignore'] = sorted(ignore)
    (dst / 'config.json').write_text(json.dumps(config, indent=2))
    state = {'shard': {}, 'bytes': 0, 'number': 1, 'index': {}, 'converted': 0}

    def flush():
        if state['shard']:
            name = f"model-{state['number']:05d}.safetensors"
            save_file(state['shard'], str(dst / name), metadata={'format': 'pt'})
            for key in state['shard']:
                state['index'][key] = name
            state['shard'], state['bytes'], state['number'] = {}, 0, state['number'] + 1

    for path in sorted(src.glob('*.safetensors')):
        with safe_open(str(path), framework='pt') as f:
            for key in list(f.keys()):
                if key.endswith('self_attn.kv_b_proj.weight_scale') or key.endswith('self_attn.kv_b_proj.input_scale'):
                    continue
                tensor = f.get_tensor(key)
                if key.endswith('self_attn.kv_b_proj.weight') and tensor.dtype == torch.float8_e4m3fn:
                    scale = f.get_tensor(key[:-len('weight')] + 'weight_scale').to(torch.float32)
                    tensor = (tensor.to(torch.float32) * scale).to(torch.bfloat16)
                    state['converted'] += 1
                state['shard'][key] = tensor.contiguous()
                state['bytes'] += tensor.numel() * tensor.element_size()
                if state['bytes'] > 4e9:
                    flush()
    flush()
    (dst / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': state['index']}))
    size = sum(p.stat().st_size for p in dst.glob('*.safetensors')) / 1e9
    note(f"обход B: kv_b_proj разжат в bf16 в {state['converted']} слоях; сборка {size:.1f} ГБ")
    (dst / '.complete').write_text('ok')
    return state['converted'] > 0

try:
    B_READY = build_kv_bf16(GLM, GLM_B)
except Exception as error:
    B_READY = False
    note(f'обход B: сборка не удалась: {error!r}')
"""),
        code("""
# 7. Пробы обоих обходов и выбор  (⏱ 10–15 мин)
RESULTS = {}
if B_READY and start_server(GLM_B, NORMAL_FLAGS, 'vllm_glm_B'):
    RESULTS['B'] = probes('B', 'vllm_glm_B')
if start_server(GLM, NO_CHUNK_FLAGS, 'vllm_glm_A'):
    RESULTS['A'] = probes('A', 'vllm_glm_A')
CHOICE = None
for key, model_dir, flags in (('B', GLM_B, NORMAL_FLAGS), ('A', GLM, NO_CHUNK_FLAGS)):
    if RESULTS.get(key) and all(RESULTS[key].get(label) for label in EXAMPLES):
        CHOICE = (key, model_dir, flags); break
note(f'ИТОГ ПРОБ: {RESULTS}; выбран обход: {CHOICE[0] if CHOICE else "НИ ОДИН"}')
(OUT / 'diagnosis.json').write_text(json.dumps({'results': RESULTS, 'choice': CHOICE[0] if CHOICE else None}), encoding='utf-8')
"""),
        code("""
# 8. Сигналы GLM base / shots на dev и τ² рабочим обходом, с замером скорости  (⏱ 15–30 мин)
if CHOICE:
    key, model_dir, flags = CHOICE
    if start_server(model_dir, flags, 'vllm_glm_run'):
        for name, path in INPUTS.items():
            t = time.time()
            result = subprocess.run([sys.executable, 'dev/score_v6.py', '--input', str(path), '--name', name, '--model', 'glm',
                                     '--api', 'completions', '--template', str(model_dir / 'chat_template.jinja'),
                                     '--base-url', URL, '--out', str(OUT / 'signals'), '--signals', 'base,shots'],
                                    cwd=WORK, capture_output=True, text=True)
            lines = [l for l in (result.stdout + result.stderr).splitlines() if l.strip() and '|' not in l]
            note(f'glm/{name} (обход {key}): код {result.returncode}, {time.time() - t:.0f} с\\n' + '\\n'.join(lines[-6:]))
        save_log('vllm_glm_run')
stop_servers()
"""),
        code("""
# 9. Сравнение с Qwen (сигналы Qwen — из прогона v6_meta)
for path in QWEN_SIGNALS.glob('*_qwen_signals.parquet'):
    shutil.copy(path, OUT / 'signals' / path.name)
result = subprocess.run([sys.executable, 'dev/eval_v6.py', '--labels', 'labels.json', '--signals', str(OUT / 'signals'),
                         '--out', str(OUT / 'summary.json')], cwd=WORK, capture_output=True, text=True)
print(result.stdout); print(result.stderr[-3000:])
(OUT / 'summary.txt').write_text(result.stdout + '\\n' + result.stderr[-3000:], encoding='utf-8')
"""),
        code("""
# 10. Архив и отключение
import zipfile
with zipfile.ZipFile(OUT / 'results_v6_glm2.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.parquet', '.log'):
            archive.write(path, path.relative_to(OUT).as_posix())
note('архив: ' + str(OUT / 'results_v6_glm2.zip') + '; отключаю среду')
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
