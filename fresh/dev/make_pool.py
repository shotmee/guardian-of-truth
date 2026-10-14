"""Пул промежуточных ходов из valid: каждый прошлый ход агента в промптах valid -> (prompt, response).

Это естественное распределение ходов (как в паблике), но без меток организаторов.
Выход: fresh/dev/data/pool.parquet (id, prompt, response, dialog, turn, src) и dev.parquet = valid + pool.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from veritas.transcript import MARKER_LINE  # noqa: E402

RESP_LINE = re.compile(r"^\t← TOOL_RESPONSE ")


def strip_responses(body: str) -> str:
    out, skip = [], False
    for line in body.split("\n"):
        if RESP_LINE.match(line):
            skip = True
            continue
        if skip and line.startswith("\t  "):
            continue
        skip = False
        out.append(line)
    return "\n".join(out).rstrip("\n")


def pool_from(valid: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for ex_id, prompt in zip(valid["id"], valid["prompt"]):
        dialog = ex_id.split("::")[0]
        ms = list(MARKER_LINE.finditer(prompt))
        for i, m in enumerate(ms):
            mk = m.group(1)
            t = re.search(r"ход\s+(\d+)", mk)
            if not t:
                continue
            end = ms[i + 1].start() if i + 1 < len(ms) else len(prompt)
            body = prompt[m.end():end].strip("\n")
            resp = mk + "\n" + strip_responses(body)
            pid = f"{dialog}::t{t.group(1)}"
            pr = prompt[:m.start()].rstrip("\n")
            if pid not in rows or len(pr) > len(rows[pid]["prompt"]):
                rows[pid] = dict(id=pid, prompt=pr, response=resp, dialog=dialog,
                                 turn=int(t.group(1)), src="pool")
    return pd.DataFrame(list(rows.values()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default=str(ROOT.parent / "valid.parquet"))
    ap.add_argument("--out", default=str(ROOT / "dev" / "data"))
    a = ap.parse_args()
    valid = pd.read_parquet(a.valid)
    pool = pool_from(valid)
    vid = set(valid["id"])
    pool = pool[~pool["id"].isin(vid)].reset_index(drop=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out / "pool.parquet", index=False)
    v = valid.copy()
    v["dialog"] = v["id"].str.split("::").str[0]
    v["turn"] = v["id"].str.extract(r"::t(\d+)")[0].astype(int)
    v["src"] = "valid"
    dev = pd.concat([v[["id", "prompt", "response", "dialog", "turn", "src", "label"]],
                     pool.assign(label=-1)], ignore_index=True)
    dev.to_parquet(out / "dev.parquet", index=False)
    print(f"pool {len(pool)}  dev {len(dev)}  dialogs {pool['dialog'].nunique()}")


if __name__ == "__main__":
    main()
