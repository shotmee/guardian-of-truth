"""Сборка архива решения на Windows без распаковки бейзлайна.

Веса Qwen3.6-35B-A3B-FP8 берутся потоково прямо из частей baseline.tar.gz.00..06 (диск не нужен под
распакованную копию), код — из папки solution/. В zip выставляются Linux-права (папки 755, файлы 644,
predict.py 755) и create_system=Unix — иначе платформа падает на шаге 08 «Статус распаковки данных».
Структура: файлы в корне архива (pyproject.toml, scripts/, src/, model/), без папки metrics.

Пример:
  python tools/build_submission.py --parts D:/downloads/baseline_guardian --out D:/downloads/guardian_v1.zip \
      [--config путь/к/config.json из Colab] [--verify]
"""
import argparse
import hashlib
import io
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solution"
SHA = {  # из комментария организаторов (28 сентября)
    "baseline.tar.gz.00": "d7e5e2e8a77a08de87d815614ec3a2e4cf3ccc3491925430943b90e96f6764d2",
    "baseline.tar.gz.01": "9c3b57335d99de6b85e94156589babeab9cdf349b41d315bb1b9097c51936517",
    "baseline.tar.gz.02": "c4144a01fb8fa661f5ab95cd225601e9676c632badda14944e7a657ccfdbba97",
    "baseline.tar.gz.03": "5fcd652cc2bba08fd8123c423d0331e6a7e4f94038fedf85e1970b9115921909",
    "baseline.tar.gz.04": "2b27de1289071a3d5fed12bec134a350f4cc870198156a4e3b4b2c84ae3521fa",
    "baseline.tar.gz.05": "5043bcc5149cd5428f4514c842d73252cba581e17a29e1ddee012fdc44e65904",
    "baseline.tar.gz.06": "9569a61b78747b20fae2f9ca9fac7a0db93f4eb90813e6a5c4c23e12a8a31cd4",
}
DATE = (2026, 10, 1, 0, 0, 0)


class Concat(io.RawIOBase):
    """Последовательное чтение нескольких файлов как одного потока (с прогрессом)."""

    def __init__(self, paths):
        self.paths, self.k, self.f = list(paths), 0, None
        self.total = sum(p.stat().st_size for p in self.paths)
        self.done, self.t0, self.last = 0, time.time(), 0.0

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.f is None:
                if self.k >= len(self.paths):
                    return 0
                self.f = open(self.paths[self.k], "rb")
            n = self.f.readinto(b)
            if n:
                self.done += n
                if time.time() - self.last > 5:
                    self.last = time.time()
                    sp = self.done / max(1e-6, self.last - self.t0) / 2**20
                    eta = (self.total - self.done) / max(1, sp * 2**20)
                    print(f"  прочитано {self.done / 2**30:5.1f}/{self.total / 2**30:.1f} ГБ "
                          f"({self.done / self.total:5.1%}), {sp:5.0f} МБ/с, осталось ~{eta / 60:4.1f} мин", flush=True)
                return n
            self.f.close()
            self.f, self.k = None, self.k + 1


def zinfo(name: str, mode: int, is_dir: bool = False) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=DATE)
    zi.create_system = 3
    if is_dir:
        zi.external_attr = ((0o040000 | mode) << 16) | 0x10  # S_IFDIR + MS-DOS directory flag
    else:
        zi.external_attr = (0o100000 | mode) << 16           # S_IFREG
    return zi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="папка с baseline.tar.gz.00..06")
    ap.add_argument("--out", required=True, help="итоговый zip (латиницей, на диске с местом ~40 ГБ)")
    ap.add_argument("--config", default=None, help="config.json с подобранными в Colab порогами")
    ap.add_argument("--verify", action="store_true", help="проверить sha256 частей (≈2-4 мин)")
    a = ap.parse_args()

    parts = sorted(Path(a.parts).glob("baseline.tar.gz.*"))
    assert len(parts) == 7, f"нужно 7 частей, найдено {len(parts)}"
    if a.verify:
        for p in parts:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            ok = h.hexdigest() == SHA.get(p.name)
            print(f"  {p.name}: {'OK' if ok else 'НЕ СОВПАДАЕТ!'}")
            assert ok, f"битая часть {p.name} — перекачай её"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        for d in ("scripts/", "src/", "src/guardian/", "model/"):
            zf.writestr(zinfo(d, 0o755, is_dir=True), b"")
        code = [("pyproject.toml", SOL / "pyproject.toml", 0o644),
                ("scripts/predict.py", SOL / "scripts" / "predict.py", 0o755)]
        for f in sorted((SOL / "src" / "guardian").iterdir()):
            if f.is_file() and f.suffix in {".py", ".md", ".json"}:
                src = Path(a.config) if (a.config and f.name == "config.json") else f
                code.append((f"src/guardian/{f.name}", src, 0o644))
        for arc, src, mode in code:
            zi = zinfo(arc, mode)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, src.read_bytes())
            print(f"  + {arc}")
        print("Потоковое копирование весов из baseline.tar.gz.* ...", flush=True)
        n_model = 0
        with tarfile.open(fileobj=io.BufferedReader(Concat(parts), 1 << 24), mode="r|gz") as tf:
            for m in tf:
                if not m.name.startswith("baseline/model/"):
                    continue
                rel = m.name[len("baseline/model/"):]
                if not rel:
                    continue
                if m.isdir():
                    zf.writestr(zinfo(f"model/{rel.rstrip('/')}/", 0o755, is_dir=True), b"")
                    continue
                if not m.isfile():
                    continue
                zi = zinfo(f"model/{rel}", 0o644)
                big = m.size > 50 * 2**20
                zi.compress_type = zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED
                src = tf.extractfile(m)
                with zf.open(zi, "w", force_zip64=True) as dst:
                    while True:
                        chunk = src.read(1 << 24)
                        if not chunk:
                            break
                        dst.write(chunk)
                n_model += 1
        print(f"  + model/: {n_model} файлов")

    # проверка
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "scripts/predict.py" in names and "pyproject.toml" in names and "model/config.json" in names
        assert not any(n.split("/")[0] == "metrics" for n in names), "в архиве не должно быть папки metrics"
        bad = [i.filename for i in zf.infolist() if ((i.external_attr >> 16) & 0o444) != 0o444]
        assert not bad, f"файлы без прав чтения: {bad[:5]}"
    gb = out.stat().st_size / 2**30
    print(f"Готово: {out} — {gb:.2f} ГБ, {len(names)} файлов, {(time.time() - t0) / 60:.1f} мин")
    assert out.stat().st_size < 40 * 10**9, "архив больше 40 ГБ!"


if __name__ == "__main__":
    main()
