"""Генерирует notebooks_extended/guardian_v5b_spare_time.ipynb: эксперимент «на что потратить запас времени».

Два новых сигнала для ходов без вызовов, только на размеченных ходах (dev 188 + новые диалоги 623):
  claims — проверка по утверждениям (veritas.claims), think — судья с рассуждением (veritas.deliberate).
Ноутбук рассчитан на запуск без присмотра: всё пишется на Диск по ходу, ошибки шагов не останавливают
ноутбук, а в конце (или по сторожевому таймеру) среда Colab отключается, чтобы не тратить CU.
python dev/make_experiment_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
NOTEBOOK = PROJECT / "notebooks_extended" / "guardian_v5b_spare_time.ipynb"
PACKED = ["src/veritas", "dev/legacy_v2", "dev/score_signals.py", "dev/make_pool.py"]
WATCHDOG_HOURS = 1.5


def labeled_text_ids(units_path: Path) -> list:
    return [unit["id"] for unit in json.loads(units_path.read_text(encoding="utf-8")) if unit["kind"] == "text"]


def pack() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for entry in PACKED:
            path = ROOT / entry
            for file in (sorted(path.rglob("*")) if path.is_dir() else [path]):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.add(file, arcname=file.relative_to(ROOT).as_posix())
        archive.add(PROJECT / "valid.parquet", arcname="valid.parquet")
        for name, units in (("dev_ids.json", PROJECT / "labeling_v5" / "units.json"),
                            ("tau2_ids.json", PROJECT / "labeling_v6" / "units.json")):
            data = json.dumps(labeled_text_ids(units)).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue()).decode()


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def build() -> dict:
    cells = [
        markdown(f"""
# Guardian of Truth v5b — на что потратить запас времени (ночной прогон, отключается сам)

**Что проверяем.** На платформе у нас остаётся ~13 минут из 30. Два способа их потратить, считаем на размеченных
текстовых ходах (dev 188 + новые диалоги τ² 623), чтобы потом сравнить с P_base и атомами офлайн:
- **claims** — модель выписывает из ответа до 6 конкретных фактов, каждый проверяется по контексту отдельным
  однотокенным вопросом (цель — выдуманные факты, основная масса ошибок в банке);
- **think** — судья с рассуждением: сначала думает (до 1536 токенов), потом P('1') из logprobs ответа.

**Как запускать:** Colab, **A100 80GB**, High-RAM → «Среда выполнения → Выполнить все» и можно уходить.
Всё сохраняется на Диск по ходу (`MyDrive/guardian_of_truth/v5b_spare_time/`), упавший шаг не останавливает
ноутбук. **В конце среда отключится сама**; если что-то зависнет — сторожевой таймер отключит её через
{WATCHDOG_HOURS} ч.

**Оценка:** ≈ 40–60 мин ≈ **5–8 CU**. Секрет `HF_TOKEN` желателен.

**Результат:** `MyDrive/guardian_of_truth/v5b_spare_time/results_v5b_spare_time.zip` — пришли его Claude.
"""),
        code(f"""
# 1. Диск, папки, GPU + сторожевой таймер (отключит среду через {WATCHDOG_HOURS} ч, если ноутбук не дойдёт до конца)
import os, sys, json, time, subprocess, threading
from pathlib import Path
from google.colab import drive, runtime
drive.mount('/content/drive')
OUT = Path('/content/drive/MyDrive/guardian_of_truth/v5b_spare_time'); OUT.mkdir(parents=True, exist_ok=True)
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
if not GENERATED.exists():                       # без новых диалогов считать нечего — сразу отключаемся
    note('нет examples_v5.parquet на Диске — отключаю среду')
    drive.flush_and_unmount(); runtime.unassign()
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
# 4. Код (veritas + скрипт оценок), dev-набор из valid, списки размеченных ходов  (⏱ <1 мин)
import base64, io, tarfile
CODE_B64 = '{pack()}'
FRESH = Path('/content/fresh'); FRESH.mkdir(exist_ok=True)
try:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz') as archive:
        archive.extractall(FRESH)
    subprocess.run([sys.executable, 'dev/make_pool.py', '--valid', 'valid.parquet', '--out', 'dev/data'], cwd=FRESH, check=False)
    note('код и данные готовы: ' + ', '.join(sorted(p.name for p in FRESH.iterdir())))
except Exception as error:
    note(f'код не распакован: {{error!r}}')
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
# 6–7. Сигналы claims и think: dev (188 ходов), затем новые диалоги (623 хода)  (⏱ 20–40 мин)
SIGNALS = OUT / 'signals'
for name, source, ids in (('dev', 'dev/data/dev.parquet', 'dev_ids.json'), ('tau2', str(GENERATED), 'tau2_ids.json')):
    if not ready:
        break
    t = time.time()
    result = subprocess.run([sys.executable, 'dev/score_signals.py', '--input', source, '--ids', ids, '--name', name,
                             '--base-url', 'http://127.0.0.1:8000/v1', '--out', str(SIGNALS),
                             '--signals', 'claims,think'], cwd=FRESH, capture_output=True, text=True)
    note(f'{name}: код {result.returncode}, {time.time() - t:.0f} с')
    note(result.stdout[-2500:]); note(result.stderr[-2500:])
"""),
        code("""
# 8. Архив результатов на Диск и отключение среды  (⏱ <1 мин)
try:
    import shutil
    shutil.make_archive(str(OUT / 'results_v5b_spare_time'), 'zip', root_dir=str(OUT), base_dir='signals')
    note('архив: ' + str(OUT / 'results_v5b_spare_time.zip'))
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
    NOTEBOOK.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{NOTEBOOK} ({NOTEBOOK.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
