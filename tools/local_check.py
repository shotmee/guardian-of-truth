"""CPU-проверка правил на valid.parquet (+ пул промежуточных ходов). Запуск: python tools/local_check.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution" / "src"))
from guardian.rules import analyze  # noqa: E402
from guardian.devtools import build_pool  # noqa: E402

df = pd.read_parquet(ROOT / "valid.parquet")
rows = []
for r in df.itertuples():
    a = analyze(r.id, r.prompt, r.response)
    rows.append({"id": r.id[:60], "label": r.label, "hard": len(a["hard"]), "soft": len(a["soft"]),
                 "calls": a["n_calls"], "tools": a["n_tools"], "why": " | ".join(a["hard"])[:160]})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 160, "display.max_rows", 200)
print(res.to_string())
hp = res[res.hard > 0]
print(f"\nhard-правила сработали: {len(hp)}; из них label=1: {int(hp.label.sum())}; "
      f"позитивов всего {int(res.label.sum())}")

if "--soft" in sys.argv:
    for r in df.itertuples():
        a = analyze(r.id, r.prompt, r.response)
        print("=" * 80, r.id, "label", r.label)
        for f in a["facts"]:
            print("  ", f[:300])

pool = build_pool(df)
hits = 0
for r in pool.itertuples():
    a = analyze(r.id, r.prompt, r.response)
    if a["hard"]:
        hits += 1
        if "--pool" in sys.argv:
            print(r.id[:70], "|", " | ".join(a["hard"])[:200])
print(f"\nпул промежуточных ходов: {len(pool)} примеров, hard-срабатываний: {hits} ({hits / max(1, len(pool)):.1%})")
