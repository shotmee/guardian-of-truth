"""Генерирует notebooks_extended/guardian_v5_tau2gen.ipynb: новые диалоги τ²-bench «как у организаторов».

Слабый агент (Qwen3.5-9B без рассуждений) обслуживает клиентов, которых играет Qwen3.6-35B-A3B-FP8; инструкция,
политики, русская база знаний банка и формат транскрипта — как в данных организаторов. Результат — независимый
dev-набор для выбора текстового ранжировщика (размечается потом субагентами turn-labeler).
Код из dev/tau2gen упаковывается в ноутбук (base64 tar.gz).
python dev/make_tau2gen_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dev" / "tau2gen"
NOTEBOOK = ROOT.parent / "notebooks_extended" / "guardian_v5_tau2gen.ipynb"
TAU2_COMMIT = "4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699"
TASKS_PER_DOMAIN = {"airline": 30, "retail": 35, "telecom": 40, "banking_knowledge": 45}


def pack_package() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(PACKAGE.iterdir()):
            if path.suffix in (".py", ".json"):
                archive.add(path, arcname=path.name)
    return base64.b64encode(buffer.getvalue()).decode()


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def build() -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v5 — новые диалоги τ²-bench «как у организаторов» (независимый dev-набор)

**Зачем.** Текстовые ранжировщики сейчас выбираются на 165 ходах из 39 диалогов — этого не хватает, чтобы
отличить ×1.1 от ×1.3, а старые промпты писались по этим же диалогам. Здесь генерируются ~150 новых диалогов
тем же способом, что у организаторов:
- инструкция агента τ² + «Always communicate with the user in Russian…» (дословно из valid);
- политики доменов дословно из valid; база знаний банка переведена на русский, как у организаторов;
- клиент — Qwen3.6-35B-A3B-FP8, агент — слабая модель **Qwen3.5-9B** без рассуждений;
- задачи — только те, которых нет в valid ({", ".join(f"{d} {n}" for d, n in TASKS_PER_DOMAIN.items())}).

**Среда:** Colab, **A100 80GB**, High-RAM. Секрет Colab `HF_TOKEN` желателен (скачивание моделей быстрее).

**Оценка:** ≈ 1.5–2 ч ≈ **11–15 CU** (A100 ≈ 7.5 CU/ч): установка и модели ~20 мин, перевод базы ~3 мин,
диалоги ~60–80 мин. RAM ≈ 20 ГБ, диск ≈ 70 ГБ.

**Чекпоинты:** всё на Диске в `MyDrive/guardian_of_truth/v5_tau2gen/`. Готовые модели, перевод и домены
пропускаются. Если сессия упала — запусти всё сверху ещё раз, повторится только незаконченный домен.

**Результат:** `MyDrive/guardian_of_truth/v5_tau2gen/results_v5_tau2gen.zip` — пришли его Claude.
"""),
        code("""
# 1. Диск, папки, GPU  (⏱ <1 мин)
import os, sys, json, time, subprocess, shutil, glob
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v5_tau2gen'); RUNS = OUT / 'runs'
RUNS.mkdir(parents=True, exist_ok=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'],
                     capture_output=True, text=True).stdout)
print(sys.version)
"""),
        code("""
# 2. vLLM 0.19.1 (как на платформе) + τ²-bench в отдельном окружении, чтобы зависимости не мешали vLLM  (⏱ 4–7 мин)
import importlib.metadata as md
try:
    vllm_version = md.version('vllm')
except md.PackageNotFoundError:
    vllm_version = None
if vllm_version != '0.19.1':
    !pip -q install vllm==0.19.1
print('vllm', md.version('vllm'))

TAU2 = Path('/content/tau2-bench'); TAU2_PY = '/content/tau2env/bin/python'
if not (TAU2 / '.ready2').exists():
    if not TAU2.exists():
        !git clone -q https://github.com/sierra-research/tau2-bench.git {TAU2}
    !cd {TAU2} && git checkout -q """ + TAU2_COMMIT + """
    !pip -q install uv
    !test -x {TAU2_PY} || uv venv -q --python 3.12 /content/tau2env
    # tau2 при импорте тянет голосовой модуль: websockets, scipy, elevenlabs, pyaudio (pyaudio — через portaudio)
    !apt-get -qq install -y portaudio19-dev > /dev/null
    !uv pip install -q --python {TAU2_PY} -e "{TAU2}[knowledge]" websockets scipy "elevenlabs>=1.54.0" pyaudio
    (TAU2 / '.ready2').write_text('ok')
check = subprocess.run([TAU2_PY, '-c', 'import tau2.registry, tau2.run, litellm; print("tau2 ok, litellm", litellm.__version__)'],
                       capture_output=True, text=True)
print(check.stdout, check.stderr[-2000:])
"""),
        code("""
# 3. Модели: клиент Qwen3.6-35B-A3B-FP8, агент Qwen3.5-9B  (⏱ 8–15 мин в первый раз)
from huggingface_hub import snapshot_download
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception as error:
    print('HF_TOKEN не найден, качаю без него:', error)
MODELS = {'user': 'Qwen/Qwen3.6-35B-A3B-FP8', 'agent': 'Qwen/Qwen3.5-9B'}
for role, repo in MODELS.items():
    target = Path('/content/models') / role
    if not (target / '.complete').exists():
        t = time.time()
        snapshot_download(repo, local_dir=str(target), allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja',
                                                                       '*.model', '*.tiktoken', 'merges.txt'])
        (target / '.complete').write_text(repo)
        print(f'✓ {role}: {repo} за {time.time() - t:.0f} с')
    print(role, sorted(p.name for p in target.iterdir())[:8], '…')
"""),
        code(f"""
# 4. Код генерации (упакован в ноутбук)  (⏱ <1 мин)
import base64, io, tarfile
CODE_B64 = '{pack_package()}'
GEN = Path('/content/tau2gen'); GEN.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz') as archive:
    archive.extractall(GEN)
print(sorted(p.name for p in GEN.iterdir()))
"""),
        code("""
# 5. Два сервера vLLM: клиент (порт 8002) и агент (порт 8001), с разбором вызовов инструментов  (⏱ 5–8 мин)
import urllib.request

def start_server(role, port, memory_share, parser='qwen3_coder'):
    log = open(f'/content/vllm_{role}.log', 'w')
    command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', f'/content/models/{role}',
               '--served-model-name', role, '--port', str(port), '--max-model-len', '131072',
               '--gpu-memory-utilization', str(memory_share), '--max-num-seqs', '64', '--enable-prefix-caching',
               '--language-model-only', '--enable-auto-tool-choice', '--tool-call-parser', parser]
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

def wait_ready(role, port, process, limit=1200):
    started = time.time()
    while time.time() - started < limit:
        if process.poll() is not None:
            print(open(f'/content/vllm_{role}.log').read()[-4000:]); raise RuntimeError(f'vLLM {role} упал')
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5); print(f'✓ {role} готов за {time.time()-started:.0f} с'); return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f'vLLM {role} не поднялся')

def alive(port):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5); return True
    except Exception:
        return False

SERVERS = globals().get('SERVERS', {})
for role, port, share in (('user', 8002, 0.55), ('agent', 8001, 0.33)):   # при перезапуске ячейки живые не трогаем
    if alive(port):
        print(f'✓ {role} уже работает'); continue
    SERVERS[role] = start_server(role, port, share); wait_ready(role, port, SERVERS[role])

def chat(port, model, messages, tools=None):
    body = {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': 200,
            'chat_template_kwargs': {'enable_thinking': False}}
    if tools: body['tools'] = tools
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(request, timeout=300).read())['choices'][0]['message']

probe_tool = [{'type': 'function', 'function': {'name': 'get_user_details', 'description': 'Get user details.',
               'parameters': {'type': 'object', 'properties': {'user_id': {'type': 'string'}}, 'required': ['user_id']}}}]
for role, port in (('agent', 8001), ('user', 8002)):
    answer = chat(port, role, [{'role': 'user', 'content': 'Найди пользователя sara_doe_496.'}], probe_tool)
    print(role, '→', answer.get('tool_calls') or answer.get('content'))
"""),
        code("""
# 6. База знаний банка → русский (как у организаторов), кэш на Диске  (⏱ 2–4 мин в первый раз)
DOCS = TAU2 / 'data/tau2/domains/banking_knowledge/documents'
!{sys.executable} {GEN}/translate_kb.py --docs {DOCS} --url http://127.0.0.1:8002/v1 --model user --cache {OUT}/kb_ru.json
sample = json.loads(sorted(DOCS.glob('*.json'))[5].read_text()); print(sample['title']); print(sample['content'][:300])
"""),
        code("""
# 7. Пробный диалог (1 задача airline) — проверка всей цепочки до большого прогона  (⏱ 2–4 мин)
tools_check = subprocess.run([TAU2_PY, '-c', 'import sys, json; sys.path.insert(0, "/content/tau2gen"); '
                              'from patches import apply_patches, check_banking_tools; '
                              'apply_patches(json.load(open("/content/tau2gen/organizer_policies.json"))); '
                              'print(check_banking_tools())'], cwd=str(TAU2), capture_output=True, text=True)
print('инструменты агента в банке:', tools_check.stdout.strip()[-600:], tools_check.stderr[-1500:])
assert all(name in tools_check.stdout for name in ('KB_search_bm25', 'KB_search_dense', "'grep'")), 'набор банка не тот'
SMOKE = Path('/content/smoke'); SMOKE.mkdir(exist_ok=True)
common = ['--exclude', f'{GEN}/valid_tasks.json', '--policies', f'{GEN}/organizer_policies.json',
          '--agent-url', 'http://127.0.0.1:8001/v1', '--agent-model', 'agent',
          '--user-url', 'http://127.0.0.1:8002/v1', '--user-model', 'user']
result = subprocess.run([TAU2_PY, f'{GEN}/run_simulations.py', '--domain', 'airline', '--tasks', '1', '--seed', '99',
                         '--concurrency', '1', '--out', str(SMOKE), *common], cwd=str(TAU2), capture_output=True, text=True)
print(result.stdout[-3000:]); print(result.stderr[-3000:])
smoke = json.loads((SMOKE / 'airline.json').read_text())
for message in smoke['simulations'][0]['messages'][:8]:
    print(message['role'], '|', (message.get('content') or '')[:150].replace(chr(10), ' '), '|',
          [c['name'] for c in message.get('tool_calls') or []])
"""),
        code(f"""
# 8. Большой прогон: домены по очереди, готовые пропускаются  (⏱ 60–80 мин)
TASKS = {json.dumps(TASKS_PER_DOMAIN)}
for domain, count in TASKS.items():
    if (RUNS / f'{{domain}}.json').exists():
        print('✓ уже есть', domain); continue
    t = time.time()
    process = subprocess.run([TAU2_PY, f'{{GEN}}/run_simulations.py', '--domain', domain, '--tasks', str(count),
                              '--concurrency', '32', '--out', str(RUNS), *common], cwd=str(TAU2),
                             capture_output=True, text=True)
    print(process.stdout[-1500:]); print(process.stderr[-1500:])
    print(f'{{domain}}: {{time.time() - t:.0f}} с, файл есть: {{(RUNS / (domain + ".json")).exists()}}', flush=True)
"""),
        code("""
# 9. Ходы в формате соревнования + сверка формата с valid  (⏱ 1–2 мин)
sys.path.insert(0, str(GEN))
from serialize import examples_frame, render_tools
examples = examples_frame(RUNS)
examples.to_parquet(OUT / 'examples_v5.parquet', index=False)
print(examples.groupby('domain').agg(dialogs=('dialog', 'nunique'), turns=('id', 'size'),
                                     text_turns=('n_calls', lambda s: int((s == 0).sum()))))
organizer_tools = json.loads((GEN / 'organizer_tools.json').read_text())
policies = json.loads((GEN / 'organizer_policies.json').read_text())
for domain in TASKS:
    run = json.loads((RUNS / f'{domain}.json').read_text())
    setup = next(iter(run['agent_setups'].values()))
    ours = render_tools(setup['tools'])
    same_names = sorted(l.split(' — ')[0] for l in ours.splitlines() if l.startswith('- ')) == \\
                 sorted(l.split(' — ')[0] for l in organizer_tools[domain].splitlines() if l.startswith('- '))
    print(f'{domain}: политика как у организаторов: {policies[domain] in setup["system_prompt"]}, '
          f'инструменты те же: {same_names}, схема дословно: {ours == organizer_tools[domain]}')
"""),
        code("""
# 10. Архив результатов → пришли его Claude  (⏱ <1 мин)
archive = OUT / 'results_v5_tau2gen.zip'
!cd {OUT} && rm -f {archive} && zip -q -r {archive} runs examples_v5.parquet kb_ru.json
print(archive, f'{archive.stat().st_size / 1e6:.1f} МБ')
for server in SERVERS.values():
    server.terminate()
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm",
                                                                         "gpuType": "A100"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> None:
    NOTEBOOK.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{NOTEBOOK} ({NOTEBOOK.stat().st_size / 1e3:.0f} КБ)")


if __name__ == "__main__":
    main()
