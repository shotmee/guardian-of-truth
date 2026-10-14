"""Генерирует notebooks_extended/guardian_v4_dev.ipynb: dev-прогон трёх стратегий v4 в Colab.

Код решения (fresh/) упаковывается в ноутбук (base64 tar.gz), поэтому ноутбук самодостаточен.
python dev/make_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT.parent / "notebooks_extended" / "guardian_v4_dev.ipynb"


def pack_code() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_dir() or "__pycache__" in rel or rel.startswith(("dev/data", "dev/out")) \
                    or rel.endswith((".pyc", ".ipynb", ".zip")):
                continue
            tar.add(p, arcname=rel)
    return base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def build():
    b64 = pack_code()
    cells = [
        md("""
# Guardian of Truth v4 — dev-прогон трёх стратегий (A «Дублёр», B «Аудитор», C «Атомы»)

**Что делает:** поднимает vLLM 0.19.1 с Qwen3.6-35B-A3B-FP8 (как на платформе), проверяет, что наш рендер
шаблона чата совпадает с `tokenizer.apply_chat_template`, прогоняет все три стратегии на dev-наборе
(46 ходов valid + 384 промежуточных хода из тех же диалогов = 430, распределение как в паблике),
затем делает «платформенный» прогон: `pip install .` + `scripts/predict.py` со своим стартом vLLM.

**Среда:** Colab, **A100 80GB**, High-RAM. Нужны секреты Colab `KAGGLE_USERNAME` и `KAGGLE_KEY` (датасет `shotme/aij-minecraft-kulbit`).

**Оценка:** ≈ 50–60 мин ≈ **7–8 CU** (A100 ≈ 7.5 CU/ч). RAM ≈ 20–30 ГБ, диск ≈ 75 ГБ (части архива удаляются после распаковки).

**Чекпоинты:** установка vLLM, модель и каждая стратегия пропускаются, если уже готовы (дампы лежат на Диске:
`MyDrive/guardian_of_truth/v4_dev/`). Если сессия упала — просто запусти всё сверху ещё раз.

**Результат:** `MyDrive/guardian_of_truth/v4_dev/results_v4_dev.zip` — пришли его Claude.
"""),
        code("""
# 1. Диск, папки, GPU  (⏱ <1 мин)
import os, sys, json, time, subprocess, shutil, glob
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
VER = 'v4_dev'
OUT = Path('/content/drive/MyDrive/guardian_of_truth') / VER
OUT.mkdir(parents=True, exist_ok=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'],
                     capture_output=True, text=True).stdout)
"""),
        code("""
# 2. vLLM 0.19.1 — та же версия, что на платформе  (⏱ 3–6 мин в первый раз)
import importlib.metadata as md
try:
    v = md.version('vllm')
except md.PackageNotFoundError:
    v = None
print('vllm сейчас:', v)
if v != '0.19.1':
    !pip -q install vllm==0.19.1
    print('vllm установлен:', md.version('vllm'))
"""),
        code("""
# 3. Веса модели и valid из Kaggle  (⏱ 10–15 мин в первый раз, потом пропускается)
MODEL = Path('/content/model'); DATA = Path('/content/data'); DATA.mkdir(exist_ok=True)
KAGGLE_DATASET = 'shotme/aij-minecraft-kulbit'

def kaggle_auth():
    try:
        from google.colab import userdata
        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    except Exception as e:
        print('секреты Kaggle не найдены, пробую без них:', e)

if not (MODEL / '.complete').exists():
    kaggle_auth(); import kagglehub
    parts = []
    for i in range(7):
        f = f'baseline.tar.gz.0{i}'
        parts.append(kagglehub.dataset_download(KAGGLE_DATASET, path=f)); print('✓', f, flush=True)
    MODEL.mkdir(exist_ok=True)
    t = time.time()
    cmd = 'cat ' + ' '.join(sorted(parts)) + f' | tar xzf - -C {MODEL} --strip-components=2 --wildcards "baseline/model/*"'
    subprocess.run(cmd, shell=True, check=True)
    print(f'распаковано за {time.time() - t:.0f} с')
    (MODEL / '.complete').write_text('ok')
    for p in parts:
        try: os.remove(p)
        except OSError: pass
if not (DATA / 'valid.parquet').exists():
    kaggle_auth(); import kagglehub
    shutil.copy(kagglehub.dataset_download(KAGGLE_DATASET, path='valid.parquet'), DATA / 'valid.parquet')
print(sorted(x.name for x in MODEL.iterdir()))
"""),
        code(f"""
# 4. Код решения v4 (упакован в ноутбук) + dev-набор  (⏱ <1 мин)
import base64, io, tarfile
CODE_B64 = '{b64}'
FRESH = Path('/content/fresh')
if FRESH.exists():
    shutil.rmtree(FRESH)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz').extractall(FRESH)
os.symlink(MODEL, FRESH / 'model')
sys.path.insert(0, str(FRESH / 'src'))
subprocess.run([sys.executable, str(FRESH / 'dev/make_pool.py'), '--valid', str(DATA / 'valid.parquet'),
                '--out', str(DATA)], check=True)
import pandas as pd
dev = pd.read_parquet(DATA / 'dev.parquet'); print(dev.src.value_counts().to_dict())
"""),
        code("""
# 5. Наш рендер шаблона == tokenizer.apply_chat_template?  (⏱ <1 мин)
from transformers import AutoTokenizer
from veritas.transcript import parse_context, parse_agent_turn
from veritas.chat import ChatTemplate, agent_messages, agent_prompt, judge_prompt
from veritas.tool_schema import parse_available_tools, to_function_schemas
tok = AutoTokenizer.from_pretrained(str(MODEL))
R = ChatTemplate(MODEL / 'chat_template.jinja')
bad = 0; tot = 0
for r in dev.sample(25, random_state=0).itertuples():
    d = parse_context(r.prompt, r.id)
    for th in (False, True):
        mine = agent_prompt(R, d, th)
        ref = tok.apply_chat_template(agent_messages(d), tools=to_function_schemas(parse_available_tools(d.tools_text)),
                                      tokenize=False, add_generation_prompt=True, enable_thinking=th)
        tot += 1
        if mine != ref:
            bad += 1
            k = next((i for i, (x, y) in enumerate(zip(mine, ref)) if x != y), min(len(mine), len(ref)))
            print('НЕСОВПАДЕНИЕ', r.id, th, repr(mine[max(0, k - 120):k + 120]), '\\n   vs', repr(ref[max(0, k - 120):k + 120]))
m2 = judge_prompt(R, 'SYS', 'USER', True)
r2 = tok.apply_chat_template([{'role': 'system', 'content': 'SYS'}, {'role': 'user', 'content': 'USER'}],
                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
print(f'несовпадений агентского шаблона: {bad} из {tot}; шаблон судьи совпадает: {m2 == r2}')
"""),
        code("""
# 6. Старт vLLM (те же флаги, что в решении)  (⏱ 4–6 мин)
from veritas.engine import VllmServer, CompletionClient
srv = VllmServer(str(MODEL), port=8000, log_path='/content/vllm.log')
t = time.time(); srv.start(); OK = srv.wait_until_ready(time.time() + 1500)
T_START = time.time() - t
print('vLLM поднят:', OK, f'за {T_START:.0f} с')
"""),
        code("""
# 7. Смоук: сырые ответы модели по одному примеру на стратегию  (⏱ ≈1 мин)
from veritas.audit import audit_request, INSTRUCTIONS_TAIL as A_TAIL, PROMPTS_DIR as PROMPTS
cl = CompletionClient('http://127.0.0.1:8000/v1', model='m')
ex = dev[dev.id == 'retail__106::t3'].iloc[0]          # позитив valid: zip из email
d = parse_context(ex.prompt, ex.id); resp = parse_agent_turn(ex.response)
print('ОТВЕТ АГЕНТА:', ex.response[:300])
o = cl.complete(agent_prompt(R, d, False), n=3, temperature=0.8, top_p=0.95, top_k=20, max_tokens=384, skip_special_tokens=False)
for ch in o['choices']: print('A·быстрый ->', repr(ch['text'][:400]))
o = cl.complete(agent_prompt(R, d, True), n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, skip_special_tokens=False)
print('A·рассуждение (хвост) ->', repr(o['choices'][0]['text'][-600:]))
system = (PROMPTS / 'standard.md').read_text() + A_TAIL
base = judge_prompt(R, system, audit_request(ex.prompt, resp, d, parse_available_tools(d.tools_text), 330000), True)
o = cl.complete(base, n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=3000, stop=['</think>'], skip_special_tokens=False)
th = o['choices'][0]['text']; print('B·рассуждение (хвост) ->', repr(th[-800:]))
v = cl.complete(base + th.rstrip() + '\\n</think>\\n\\nVERDICT:', max_tokens=1, temperature=0.0, logprobs=20, skip_special_tokens=False)
print('B·вердикт top ->', v['choices'][0]['logprobs']['top_logprobs'][0])
"""),
        code("""
# 8. Три стратегии на dev (430 ходов) против уже поднятого vLLM  (⏱ ≈15–20 мин; готовые пропускаются)
def run_strategy(s):
    dump = OUT / f'dump_{s}.json'
    if dump.exists():
        print(f'{s}: уже есть {dump.name} — пропускаю'); return
    logf = OUT / f'log_{s}.txt'
    cmd = [sys.executable, str(FRESH / 'scripts/predict.py'), '--input', str(DATA / 'dev.parquet'),
           '--output', f'/content/pred_{s}.parquet', '--base-url', 'http://127.0.0.1:8000/v1',
           '--strategy', s, '--dump', f'/content/dump_{s}.json', '--config', str(FRESH / 'configs' / CFG[s])]
    t = time.time()
    with open(logf, 'w') as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end=''); lf.write(line)
        p.wait()
    print(f'== {s}: код {p.returncode}, {time.time() - t:.0f} с')
    if p.returncode == 0:
        shutil.copy(f'/content/dump_{s}.json', dump)

CFG = {'replay': 'A_replay.json', 'audit': 'B_audit.json', 'atoms': 'C_atoms.json'}
for s in ['atoms', 'replay', 'audit']:
    run_strategy(s)
"""),
        code("""
# 9. «Как на платформе»: pip install . + predict.py со своим стартом vLLM на valid (46)  (⏱ ≈8 мин)
srv.stop(); time.sleep(10)
E2E = OUT / 'e2e.json'
if not E2E.exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', str(FRESH)], check=True)
    t = time.time()
    p = subprocess.run([sys.executable, 'scripts/predict.py', '--input', str(DATA / 'valid.parquet'),
                        '--output', '/content/pred_e2e.csv'], cwd=FRESH, capture_output=True, text=True)
    el = time.time() - t
    print(p.stdout[-6000:]); print(p.stderr[-3000:])
    import pyarrow.parquet as pq
    info = dict(code=p.returncode, seconds=el)
    if p.returncode == 0:
        tb = pq.read_table('/content/pred_e2e.csv'); info.update(schema=str(tb.schema), rows=tb.num_rows,
                                                                  flags=int(sum(tb.column('label').to_pylist())))
    print(info); E2E.write_text(json.dumps(info, ensure_ascii=False))
else:
    print(E2E.read_text())
"""),
        code("""
# 10. Сводка и архив результатов для Claude  (⏱ <1 мин)
subprocess.run([sys.executable, str(FRESH / 'dev/analyze.py'), '--dev', str(DATA / 'dev.parquet'),
                '--dumps', str(OUT), '--out', str(OUT / 'summary.json')])
info = dict(t_vllm_start=globals().get('T_START'), gpu=subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
            '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
(OUT / 'env.json').write_text(json.dumps(info))
shutil.copy('/content/vllm.log', OUT / 'vllm.log') if Path('/content/vllm.log').exists() else None
zp = shutil.make_archive('/content/results_v4_dev', 'zip', OUT)
shutil.copy(zp, OUT / 'results_v4_dev.zip')
print('Готово. Пришли Claude файл:', OUT / 'results_v4_dev.zip')
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    NB.parent.mkdir(parents=True, exist_ok=True)
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{NB} ({NB.stat().st_size / 1e3:.0f} КБ, код {len(b64) / 1e3:.0f} КБ base64)")


if __name__ == "__main__":
    build()
