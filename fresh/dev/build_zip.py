"""Сборка архивов сабмита v4 из частей baseline.tar.gz.* без распаковки на диск.

За один проход по tar.gz веса пишутся сразу во все запрошенные zip (ZIP_STORED, zip64),
потом в каждый дописывается код (fresh/) со своим config.json. Права Linux: каталоги 755,
файлы 644, scripts/*.py 755. Файлы в корне архива, каталога metrics нет.

Сборка:  python dev/build_zip.py --parts D:/downloads/baseline_guardian --out-dir D:/downloads \
             --configs A_replay,B_audit,C_atoms
Готовый код из папок (по одной на конфиг):  ... --configs H1_final,H2_final --code-dirs <H1>,<H2>
Проверка готового архива:  python dev/build_zip.py --check D:/downloads/guardian_v5_H1_final.zip
Патч кода в готовом архиве (секунды, веса не трогаются):
         python dev/build_zip.py --patch D:/downloads/guardian_v4_A_replay.zip --configs A_replay
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import shutil
import struct
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]          # fresh/
CHUNK = 16 * 1024 * 1024
MODEL_PREFIX = "baseline/model/"


class Concat(io.RawIOBase):
    def __init__(self, paths):
        self.paths = list(paths)
        self.i = 0
        self.f = open(self.paths[0], "rb")
        self.total = sum(os.path.getsize(p) for p in self.paths)
        self.read_bytes = 0

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            n = self.f.readinto(b)
            if n:
                self.read_bytes += n
                return n
            self.f.close()
            self.i += 1
            if self.i >= len(self.paths):
                return 0
            self.f = open(self.paths[self.i], "rb")


def zinfo(name: str, mode: int, is_dir: bool = False) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    zi.create_system = 3
    zi.compress_type = zipfile.ZIP_STORED
    if is_dir:
        zi.external_attr = ((0o40000 | mode) << 16) | 0x10
    else:
        zi.external_attr = (0o100000 | mode) << 16
    return zi


# модули, которые нужны детерминированным зондам (без модели): остальные в их архив не кладём
PROBE_MODULES = {"__init__.py", "cli.py", "console.py", "tables.py", "analysis.py", "probes.py",
                 "transcript.py", "tool_schema.py", "checks.py", "text_checks.py"}
# финальная стратегия: зонды + модель (P_base, атомы вызовов и текста); без replay/audit/claims/deliberate
FINAL_MODULES = PROBE_MODULES | {"final.py", "baseline.py", "atoms.py", "text_atoms.py", "chat.py", "engine.py",
                                 "baseline.md", "standard.md", "text_standard.md", "chat_template.jinja"}


def archive_label(config_name: str) -> str:
    """Имя архива по версии из конфига: v5-H1_final -> guardian_v5_H1_final.zip."""
    version = json.loads((ROOT / "configs" / f"{config_name}.json").read_text(encoding="utf-8")).get("version", "")
    return version.replace("-", "_") or config_name


def code_files(config_name: str) -> list:
    """[(путь в архиве, содержимое, права)] — код решения с выбранным конфигом."""
    config_bytes = (ROOT / "configs" / f"{config_name}.json").read_bytes()
    strategy = json.loads(config_bytes)["strategy"]
    is_probe = strategy.startswith("probe_")
    allowed = PROBE_MODULES if is_probe else (FINAL_MODULES if strategy == "final" else None)
    files = [("pyproject.toml", (ROOT / "pyproject.toml").read_bytes(), 0o644),
             ("README.md", (ROOT / "README.md").read_bytes(), 0o644),
             ("scripts/predict.py", (ROOT / "scripts" / "predict.py").read_bytes(), 0o755)]
    for path in sorted((ROOT / "src").rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative == "src/veritas/config.json":
            continue
        if allowed is not None and path.name not in allowed:
            continue
        if is_probe and path.parent.name != "veritas":
            continue
        files.append((relative, path.read_bytes(), 0o644))
    files.append(("src/veritas/config.json", config_bytes, 0o644))
    return files


def code_dirs(files: list) -> list:
    dirs = set()
    for name, _, _ in files:
        parts = name.split("/")[:-1]
        for k in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:k]) + "/")
    return sorted(dirs)


def code_files_from_dir(code_dir: str) -> list:
    """[(путь в архиве, содержимое, права)] — готовый код решения из папки (например, переписанный вручную)."""
    root = Path(code_dir)
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    first = {"pyproject.toml": 0, "README.md": 1, "scripts/predict.py": 2, "src/veritas/config.json": 9}
    files.sort(key=lambda p: (first.get(p.relative_to(root).as_posix(), 5), p.relative_to(root).as_posix()))
    return [(p.relative_to(root).as_posix(), p.read_bytes(),
             0o755 if p.relative_to(root).parts[0] == "scripts" and p.suffix == ".py" else 0o644) for p in files]


def write_code(zf: zipfile.ZipFile, config_name: str, code_dir: Optional[str] = None):
    files = code_files_from_dir(code_dir) if code_dir else code_files(config_name)
    for d in code_dirs(files):
        if d not in zf.NameToInfo:
            zf.writestr(zinfo(d, 0o755, True), b"")
    for name, data, mode in files:
        zf.writestr(zinfo(name, mode), data)


def build(parts_dir: str, out_dir: str, configs: list, code_dirs: Optional[list] = None):
    parts = sorted(glob.glob(os.path.join(parts_dir, "baseline.tar.gz.*")))
    if not parts:
        sys.exit(f"нет частей в {parts_dir}")
    outs = [Path(out_dir) / f"guardian_{archive_label(c)}.zip" for c in configs]
    need = 1.25 * sum(os.path.getsize(x) for x in parts) * len(outs)   # zip STORED ≈ 1.22 × tar.gz
    free = shutil.disk_usage(out_dir).free + sum(p.stat().st_size for p in outs if p.exists())   # перезапишутся
    if free < need:
        sys.exit(f"мало места: свободно {free / 1e9:.1f} ГБ, нужно ≈{need / 1e9:.0f} ГБ")
    zfs = [zipfile.ZipFile(p, "w", allowZip64=True) for p in outs]
    for zf in zfs:
        zf.writestr(zinfo("model/", 0o755, True), b"")
    src = Concat(parts)
    t0 = time.time()
    n_files = 0
    with tarfile.open(fileobj=io.BufferedReader(src, CHUNK), mode="r|gz") as tar:
        for m in tar:
            if not m.name.startswith(MODEL_PREFIX):
                continue
            rel = "model/" + m.name[len(MODEL_PREFIX):]
            if m.isdir():
                if rel != "model/":
                    for zf in zfs:
                        zf.writestr(zinfo(rel.rstrip("/") + "/", 0o755, True), b"")
                continue
            if not m.isfile():
                continue
            f = tar.extractfile(m)
            ws = []
            for zf in zfs:
                zi = zinfo(rel, 0o644)
                zi.file_size = m.size
                ws.append(zf.open(zi, "w", force_zip64=True))
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                for w in ws:
                    w.write(buf)
            for w in ws:
                w.close()
            n_files += 1
            el = time.time() - t0
            print(f"  {rel} ({m.size / 1e9:.2f} ГБ) | прочитано {src.read_bytes / src.total:.0%} | {el:.0f} с",
                  flush=True)
    for i, (zf, c) in enumerate(zip(zfs, configs)):
        write_code(zf, c, code_dirs[i] if code_dirs else None)
        zf.close()
    for p in outs:
        verify(p)
    print(f"готово за {time.time() - t0:.0f} с, файлов модели {n_files}")


def patch(path: str, config_name: str):
    """Заменяет код в готовом архиве: усечение после весов + новые записи кода."""
    zf = zipfile.ZipFile(path, "a", allowZip64=True)
    code = [zi for zi in zf.filelist if not zi.filename.startswith("model/")]
    if code:
        first = min(zi.header_offset for zi in code)
        model_after = [zi for zi in zf.filelist if zi.filename.startswith("model/") and zi.header_offset > first]
        if model_after:
            zf.close()
            sys.exit("код лежит не в конце архива — патч невозможен, пересоберите")
        zf.filelist = [zi for zi in zf.filelist if zi.filename.startswith("model/")]
        zf.NameToInfo = {zi.filename: zi for zi in zf.filelist}
        zf.start_dir = first
        zf.fp.seek(first)
        zf.fp.truncate()
        zf._didModify = True
    write_code(zf, config_name)
    zf.close()
    verify(Path(path))


def build_code_only(out_dir: str, configs: list):
    """Архив без весов (детерминированные зонды): только код и конфиг."""
    for c in configs:
        p = Path(out_dir) / f"guardian_probe_{c.split('_', 1)[1].lower()}.zip"
        with zipfile.ZipFile(p, "w") as zf:
            write_code(zf, c)
        verify(p, need_model=False)


def local_header_problems(p: Path) -> list:
    """Строгая сверка локальных заголовков с центральным каталогом (как у строгих распаковщиков, напр. libarchive):
    имя, метод, CRC, размеры (в т. ч. из zip64), повторяющиеся extra-поля, пропуски и перекрытия между записями.
    08.10 архив, переписанный сторонней утилитой, нёс в заголовках весов два поля zip64 — платформа его не распаковала."""
    problems = []
    with zipfile.ZipFile(p) as zf, open(p, "rb") as raw:
        infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
        end = 0
        for info in infos:
            if info.header_offset != end:
                problems.append(f"{info.filename}: {'перекрытие' if info.header_offset < end else 'пропуск'} "
                                f"{abs(info.header_offset - end)} байт перед записью")
            raw.seek(info.header_offset)
            head = raw.read(30)
            if head[:4] != b"PK\x03\x04":
                problems.append(f"{info.filename}: нет локального заголовка")
                break
            _, _, flags, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<4sHHHHHIIIHH", head)
            name = raw.read(nlen).decode("utf-8" if flags & 0x800 else "cp437")
            extra = raw.read(xlen)
            tags, values, i = [], {}, 0
            while i + 4 <= len(extra):
                tag, length = struct.unpack("<HH", extra[i:i + 4])
                tags.append(tag)
                values[tag] = extra[i + 4:i + 4 + length]
                i += 4 + length
            if len(tags) != len(set(tags)):
                problems.append(f"{info.filename}: повторяющиеся extra-поля {tags}")
            if usize == 0xFFFFFFFF or csize == 0xFFFFFFFF:
                field = values.get(1, b"")
                expected = 8 * ((usize == 0xFFFFFFFF) + (csize == 0xFFFFFFFF))
                if len(field) < expected:
                    problems.append(f"{info.filename}: поле zip64 короче нужного ({len(field)} < {expected})")
                else:
                    sizes = struct.unpack("<" + "Q" * (expected // 8), field[:expected])
                    usize = sizes[0] if usize == 0xFFFFFFFF else usize
                    csize = sizes[-1] if csize == 0xFFFFFFFF else csize
            if name != info.filename or method != info.compress_type or flags & 0x08 \
                    or (crc, csize, usize) != (info.CRC, info.compress_size, info.file_size):
                problems.append(f"{info.filename}: локальный заголовок не совпадает с каталогом")
            end = info.header_offset + 30 + nlen + xlen + info.compress_size
        if end != zf.start_dir:
            problems.append(f"между последней записью и каталогом {zf.start_dir - end} байт")
    return problems


def verify(p: Path, need_model: bool = True):
    problems = local_header_problems(p)
    if problems:
        print("\n".join(problems[:20]))
        sys.exit(f"{p.name}: локальные заголовки не прошли проверку ({len(problems)} проблем)")
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        bad = [n for n in names if n.startswith("metrics") or n.startswith("/")]
        need = ["pyproject.toml", "scripts/predict.py", "src/veritas/cli.py", "src/veritas/checks.py",
                "src/veritas/config.json"]
        if need_model:
            need += ["model/config.json", "model/chat_template.jinja", "model/tokenizer.json"]
        miss = [n for n in need if n not in names]
        st = sum(1 for n in names if n.endswith(".safetensors"))
        perms = {oct((zf.getinfo(n).external_attr >> 16) & 0o777) for n in names}
        cfg = json.loads(zf.read("src/veritas/config.json"))
        assert zf.testzip is not None
    size = p.stat().st_size
    print(f"{p.name}: {size / 1e9:.2f} ГБ, записей {len(names)}, safetensors {st}, права {sorted(perms)}, "
          f"стратегия {cfg['strategy']} ({cfg.get('version')}), пропущено {miss}, лишнее {bad}")
    if miss or bad or size >= 40e9:
        sys.exit("архив не прошёл проверку")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="D:/downloads/baseline_guardian")
    ap.add_argument("--out-dir", default="D:/downloads")
    ap.add_argument("--configs", default="A_replay,B_audit,C_atoms")
    ap.add_argument("--patch", default=None, help="путь к готовому zip: заменить только код")
    ap.add_argument("--no-model", action="store_true", help="архив без весов (детерминированные зонды)")
    ap.add_argument("--code-dirs", default=None,
                    help="папки с готовым кодом через запятую, по одной на конфиг (вместо кода из fresh)")
    ap.add_argument("--check", default=None, help="только проверить готовый архив (локальные заголовки и состав)")
    a = ap.parse_args()
    if a.check:
        verify(Path(a.check))
        return
    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    code_dirs = [d.strip() for d in a.code_dirs.split(",")] if a.code_dirs else None
    if code_dirs and len(code_dirs) != len(configs):
        sys.exit("--code-dirs: нужно по одной папке на конфиг")
    for code_dir, config in zip(code_dirs or [], configs):
        bundled = json.loads((Path(code_dir) / "src" / "veritas" / "config.json").read_text(encoding="utf-8"))
        expected = json.loads((ROOT / "configs" / f"{config}.json").read_text(encoding="utf-8"))
        if bundled != expected:
            sys.exit(f"{code_dir}: config.json не совпадает с configs/{config}.json")
    if a.no_model:
        build_code_only(a.out_dir, configs)
    elif a.patch:
        patch(a.patch, configs[0])
    else:
        build(a.parts, a.out_dir, configs, code_dirs)


if __name__ == "__main__":
    main()
