"""Генерирует notebooks_extended/guardian_v5_signals.ipynb: все сигналы на dev-наборах одним сервером vLLM.

Наборы: dev (valid + пул из тех же диалогов, 430 ходов) и новые диалоги τ² (examples_v5.parquet из ноутбука
guardian_v5_tau2gen). Сигналы — dev/score_signals.py: p_base, p_v1, p_v2 (старые судьи v2.1, запросы
один в один), текстовые атомы, атомы по вызовам. Код и valid.parquet упаковываются в ноутбук (base64 tar.gz).
python dev/make_signals_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v5_signals.ipynb"
PACKED = ["src/veritas", "dev/legacy_v2", "dev/score_signals.py", "dev/make_pool.py"]


def pack() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for entry in PACKED:
            path = ROOT / entry
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in files:
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.add(file, arcname=file.relative_to(ROOT).as_posix())
        archive.add(PROJECT / "valid.parquet", arcname="valid.parquet")
    return base64.b64encode(buffer.getvalue()).decode()


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def build() -> dict:
    cells = [
        markdown("""
# Guardian of Truth v5 — сигналы для выбора текстового ранжировщика

**Что считает.** Одна модель Qwen3.6-35B-A3B-FP8 (как на платформе), без рассуждений, один токен на вопрос:
- `p_base` — промпт бейзлайна организаторов;
- `p_v1`, `p_v2` — старые судьи (запросы один в один как в v2.1 — сверим со старыми дампами);
- текстовые атомы — 6 узких вопросов к ходам без вызовов (новое);
- атомы по вызовам (стратегия C) — для ходов с вызовами.

**Наборы:** dev (valid + пул, 430 ходов) и новые диалоги τ² из ноутбука `guardian_v5_tau2gen`
(`MyDrive/guardian_of_truth/v5_tau2gen/examples_v5.parquet`). **Запускай после окончания генерации**
(отдельная сессия Colab). Если файла ещё нет, шаг 7 скажет об этом — перезапусти его потом.

**Среда:** Colab, **A100 80GB**, High-RAM. Секрет `HF_TOKEN` желателен.

**Оценка:** ≈ 60–75 мин ≈ **8–9 CU** (модель качается ~5–10 мин, dev ~10 мин, новые диалоги ~30–45 мин).
RAM ≈ 15 ГБ, диск ≈ 40 ГБ.

**Чекпоинты:** каждая оценка пишется в jsonl на Диске (`MyDrive/guardian_of_truth/v5_signals/`); после
обрыва перезапусти всё — досчитается только недостающее.

**Результат:** `MyDrive/guardian_of_truth/v5_signals/results_v5_signals.zip` — пришли его Claude.
"""),
        code("""
# 1. Диск, папки, GPU  (⏱ <1 мин)
import os, sys, json, time, subprocess, shutil
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v5_signals'); OUT.mkdir(parents=True, exist_ok=True)
GENERATED = Path('/content/drive/MyDrive/guardian_of_truth/v5_tau2gen/examples_v5.parquet')
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv'],
                     capture_output=True, text=True).stdout)
"""),
        code("""
# 2. vLLM 0.19.1 — как на платформе  (⏱ 3–6 мин в первый раз)
import importlib.metadata as md
try:
    vllm_version = md.version('vllm')
except md.PackageNotFoundError:
    vllm_version = None
if vllm_version != '0.19.1':
    !pip -q install vllm==0.19.1
print('vllm', md.version('vllm'))
"""),
        code("""
# 3. Модель Qwen3.6-35B-A3B-FP8  (⏱ 5–10 мин в первый раз)
from huggingface_hub import snapshot_download
MODEL = Path('/content/models/user')
if not (MODEL / '.complete').exists():
    try:
        from google.colab import userdata
        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    except Exception as error:
        print('HF_TOKEN не найден, качаю без него:', error)
    t = time.time()
    snapshot_download('Qwen/Qwen3.6-35B-A3B-FP8', local_dir=str(MODEL),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'merges.txt'])
    (MODEL / '.complete').write_text('Qwen/Qwen3.6-35B-A3B-FP8')
    print(f'скачано за {time.time() - t:.0f} с')
print(sorted(p.name for p in MODEL.iterdir())[:10])
"""),
        code(f"""
# 4. Код (veritas + старые судьи) и dev-набор из valid  (⏱ <1 мин)
import base64, io, tarfile
CODE_B64 = '{pack()}'
FRESH = Path('/content/fresh'); FRESH.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz') as archive:
    archive.extractall(FRESH)
!cd {{FRESH}} && {{sys.executable}} dev/make_pool.py --valid valid.parquet --out dev/data
print(sorted(str(p.relative_to(FRESH)) for p in FRESH.glob('*/*'))[:12])
"""),
        code("""
# 5. Сервер vLLM для судей  (⏱ 4–6 мин)
import urllib.request
subprocess.run('pkill -f "vllm.entrypoints.openai.api_server" || true', shell=True)   # на случай перезапуска ячейки
time.sleep(5)
log = open('/content/vllm_judge.log', 'w')
JUDGE = subprocess.Popen([sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(MODEL),
                          '--served-model-name', 'm', '--port', '8000', '--max-model-len', '131072',
                          '--gpu-memory-utilization', '0.90', '--max-num-seqs', '128', '--enable-prefix-caching',
                          '--language-model-only'], stdout=log, stderr=subprocess.STDOUT)
started = time.time()
while True:
    if JUDGE.poll() is not None:
        print(open('/content/vllm_judge.log').read()[-4000:]); raise RuntimeError('vLLM упал')
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5); break
    except Exception:
        time.sleep(5)
print(f'vLLM готов за {time.time() - started:.0f} с')
"""),
        code("""
# 6. Сигналы на dev (430 ходов)  (⏱ 8–12 мин)
SIGNALS = OUT / 'signals'
!cd {FRESH} && {sys.executable} dev/score_signals.py --input dev/data/dev.parquet --name dev \\
    --base-url http://127.0.0.1:8000/v1 --out {SIGNALS}
"""),
        code("""
# 7. Сигналы на новых диалогах τ²  (⏱ 30–45 мин)
if GENERATED.exists():
    !cd {FRESH} && {sys.executable} dev/score_signals.py --input {GENERATED} --name tau2 \\
        --base-url http://127.0.0.1:8000/v1 --out {SIGNALS}
else:
    print('examples_v5.parquet ещё нет — дождись конца генерации и перезапусти эту ячейку')
"""),
        code("""
# 8. Архив результатов → пришли его Claude  (⏱ <1 мин)
archive = OUT / 'results_v5_signals.zip'
!cd {OUT} && rm -f {archive} && zip -q -r {archive} signals
print(archive, f'{archive.stat().st_size / 1e6:.1f} МБ')
JUDGE.terminate()
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm",
                                                                         "gpuType": "A100"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> None:
    NOTEBOOK.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{NOTEBOOK} ({NOTEBOOK.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
