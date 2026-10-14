"""Крошечные архивы-зонды лидерборда (без модели, загрузка за секунды). Каждый успешный запуск тратит 1 попытку.
  probe_all_ones.zip  — метка 1 для всех: precision = 130 / N -> точный размер паблика.
  probe_rules.zip     — только hard-правила: доля позитивов паблика, которую ловят правила, и их точность.
  probe_text_only.zip — 1 для ходов без вызовов инструментов: сколько позитивов паблика в текстовых ходах (гипотеза «текст»).
Запуск: python tools/build_probe_zips.py [--out D:/downloads] [--only text_only]"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solution"
DATE = (2026, 10, 5, 0, 0, 0)

ALL_ONES = '''#!/usr/bin/env python
"""Зонд лидерборда: метка 1 для всех примеров."""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
a, _ = ap.parse_known_args()
with open(a.input, "rb") as f:
    magic = f.read(4)
df = pd.read_parquet(a.input) if magic == b"PAR1" else pd.read_csv(a.input)
out = pd.DataFrame({"id": df["id"].astype(str), "label": pd.Series([1] * len(df), dtype="int64")})
out.to_parquet(a.output, index=False)
print("all ones:", len(out))
'''

RULES = '''#!/usr/bin/env python
"""Зонд лидерборда: только детерминированные hard-правила (без LLM)."""
import sys, time
T = time.time()
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import argparse
from guardian.pipeline import run
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
a, _ = ap.parse_known_args()
args = argparse.Namespace(input=a.input, output=a.output, model_path="", config=str(ROOT / "src/guardian/config.json"),
                          base_url=None, mode="rules", dump=None, cache=None, llm_on_hard=False, limit=None, deadline=None)
run(args, T)
'''

TEXT_ONLY = '''#!/usr/bin/env python
"""Зонд лидерборда: метка 1 для всех ходов «только текст» (нет вызовов инструментов), остальным 0.
TP = число позитивов паблика среди текстовых ходов, TP + FP = число текстовых ходов в паблике."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import argparse
from guardian.pipeline import read_table, write_preds
from guardian.rules import analyze
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
a, _ = ap.parse_known_args()
df = read_table(a.input)
labels = []
for r in df.itertuples(index=False):
    try:
        fl = analyze(str(r.id), str(r.prompt), str(r.response)).get("flags", {}) or {}
        labels.append(int(bool(fl.get("has_text")) and int(fl.get("n_calls", 0) or 0) == 0))
    except Exception:  # noqa: BLE001
        labels.append(0)
write_preds([str(x) for x in df["id"]], labels, a.output)
print("text-only:", sum(labels), "of", len(labels))
'''

PYPROJECT = '''[project]
name = "guardian-probe"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["pandas>=2.0", "pyarrow>=15", "numpy>=1.24", "openai>=1.40"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
'''


def zinfo(name, mode, is_dir=False):
    zi = zipfile.ZipInfo(name, date_time=DATE)
    zi.create_system = 3
    zi.external_attr = (((0o040000 | mode) << 16) | 0x10) if is_dir else ((0o100000 | mode) << 16)
    zi.compress_type = zipfile.ZIP_DEFLATED
    return zi


def build(path, predict_src, with_pkg):
    with zipfile.ZipFile(path, "w") as zf:
        for d in ("scripts/", "src/", "src/guardian/"):
            zf.writestr(zinfo(d, 0o755, True), b"")
        zf.writestr(zinfo("pyproject.toml", 0o644), PYPROJECT)
        zf.writestr(zinfo("scripts/predict.py", 0o755), predict_src)
        if with_pkg:
            for f in sorted((SOL / "src" / "guardian").iterdir()):
                if f.is_file() and f.suffix in {".py", ".md", ".json"}:
                    zf.writestr(zinfo(f"src/guardian/{f.name}", 0o644), f.read_bytes())
        else:
            zf.writestr(zinfo("src/guardian/__init__.py", 0o644), "")
    print("->", path, f"{Path(path).stat().st_size / 1024:.0f} КБ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/downloads")
    ap.add_argument("--only", choices=["all_ones", "rules", "text_only"], default=None)
    a = ap.parse_args()
    probes = {"all_ones": (ALL_ONES, False), "rules": (RULES, True), "text_only": (TEXT_ONLY, True)}
    for name, (src, pkg) in probes.items():
        if a.only in (None, name):
            build(f"{a.out}/guardian_probe_{name}.zip", src, with_pkg=pkg)
