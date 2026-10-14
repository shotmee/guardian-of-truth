"""Оценка фокусной проверки регламента (v6p): сигналы slice / window / focus против P_base и H1.

Текст (τ² без петель — главный набор, dev — сверка): AUC и P@160 с весами под состав паблика; сигнал сам по себе,
P_base + сигнал, H1 + сигнал (H1 = P_base + текстовые атомы + сдвиг группы, прежний прогон); парный бутстрэп по
диалогам для отношения P@160 к H1; средний ранг ошибок по типам и для ошибок из зоны P_base < 0.5 (на неё и
нацелена проверка). Вызовы τ²: AUC и число ошибок в топ-20% для атомов и атомов + сигнал.
Правило (записано до прогона): сигнал берём, если на τ² H1 + сигнал даёт P@160 ≥ ×1.05 к H1 с нижней границей (10%)
> 1.0 и на dev не меньше ×1.0, или если на вызовах атомы + сигнал дают прирост AUC с интервалом выше нуля и ≥ +2
ошибки в топ-20%.
python dev/eval_policy.py --labels labels.json --signals <папка с *_qwen_signals.parquet> [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_v6 import jitter, load_signals, logit, precision_at, weighted_auc

SIGNALS = ("slice", "window", "focus")
DRAWS = 600


def boot(frame: pd.DataFrame, col: str, ref: str, rng) -> tuple:
    dialogs = frame.dialog.values
    unique = np.unique(dialogs)
    members = {d: np.flatnonzero(dialogs == d) for d in unique}
    total = frame.weight.sum()
    out = []
    for _ in range(DRAWS):
        idx = np.concatenate([members[d] for d in rng.choice(unique, len(unique))])
        s = frame.iloc[idx]
        k = 160 * s.weight.sum() / total
        a = precision_at(jitter(s[col].values), s.label.values, s.weight.values, k)
        b = precision_at(jitter(s[ref].values), s.label.values, s.weight.values, k)
        out.append(a / max(b, 1e-9))
    return tuple(np.percentile(out, [10, 50, 90]))


def mean_rank(frame: pd.DataFrame, col: str, mask) -> float:
    negatives = np.sort(frame.loc[frame.label == 0, col].values)
    values = frame.loc[mask & (frame.label == 1), col].values
    return float(np.mean(np.searchsorted(negatives, values) / max(1, len(negatives)))) if len(values) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    signals = load_signals(Path(args.signals))
    rng = np.random.default_rng(0)
    summary, decisions = {}, {}
    available = [s for s in SIGNALS if f"qwen_{s}" in signals.columns and signals[f"qwen_{s}"].notna().mean() > 0.5]
    print(f"посчитанные сигналы: {available}")
    for name in ("tau2", "dev"):
        text = pd.DataFrame([r for r in labels["text"] if r["set"] == name])
        frame = text.merge(signals[signals.set == name].drop(columns=["set"]), on="id", how="left")
        frame = frame[frame.weight > 0].reset_index(drop=True)
        if frame.empty:
            continue
        frame["P_base"] = logit(frame.p_base_prev)
        frame["H1"] = frame.P_base + logit(frame.atoms_prev) + frame["shift"]
        for s in available:
            values = logit(frame[f"qwen_{s}"].fillna(frame.p_base_prev))
            frame[s] = values
            frame[f"P_base + {s}"] = frame.P_base + values
            frame[f"H1 + {s}"] = frame.H1 + values
        y, w = frame.label.values, frame.weight.values
        print(f"\n=== ТЕКСТ [{name}]: ходов {len(frame)}, ошибок {int(y.sum())}")
        rows = {}
        for col in ["P_base", "H1"] + [c for s in available for c in (s, f"P_base + {s}", f"H1 + {s}")]:
            lo, med, hi = boot(frame, col, "H1", rng)
            rows[col] = {"AUC": weighted_auc(jitter(frame[col].values), y, w),
                         "P@160": precision_at(jitter(frame[col].values), y, w, 160), "к H1": med, "10%": lo, "90%": hi}
        table = pd.DataFrame(rows).T
        print(table.round(3).to_string())
        summary[f"text_{name}"] = table.round(4).to_dict(orient="index")
        low_zone = frame.p_base_prev < 0.5
        print(f"  средний ранг ошибок (доля не-ошибок ниже), зона P_base < 0.5: ошибок {int((low_zone & (frame.label == 1)).sum())}")
        for col in ["P_base", "H1"] + available:
            line = f"    {col:8s} зона <0.5: {mean_rank(frame, col, low_zone):.2f}"
            for cat in ("policy_violation", "false_refusal", "reask_known", "unsupported_claim", "wrong_info"):
                mask = (frame.category == cat).values
                if (mask & (frame.label == 1).values).sum() >= 3:
                    line += f" | {cat} {mean_rank(frame, col, mask):.2f}"
            print(line)
        for s in available:
            r = rows[f"H1 + {s}"]
            if name == "tau2":
                decisions[s] = {"tau2": r["к H1"], "tau2_10%": r["10%"]}
            elif s in decisions:
                decisions[s]["dev"] = r["к H1"]
    calls = pd.DataFrame(labels["calls"])
    if not calls.empty and available:
        frame = calls.merge(signals[signals.set == "tau2"].drop(columns=["set"]), on="id", how="left")
        frame["atoms"] = logit(frame.calls_prev.fillna(0.5))
        y = frame.label.values
        print(f"\n=== ВЫЗОВЫ [tau2]: ходов {len(frame)}, ошибок {int(y.sum())}")
        k = int(round(0.203 * len(frame)))
        base_auc = weighted_auc(jitter(frame.atoms.values), y, np.ones(len(y)))
        base_top = int(y[np.argsort(-jitter(frame.atoms.values))[:k]].sum())
        print(f"  атомы: AUC {base_auc:.3f}, ошибок в топ-{k}: {base_top}")
        dialogs = frame.dialog.values
        unique = np.unique(dialogs)
        members = {d: np.flatnonzero(dialogs == d) for d in unique}
        for s in available:
            combined = frame.atoms + logit(frame[f"qwen_{s}"].fillna(0.5))
            auc = weighted_auc(jitter(combined.values), y, np.ones(len(y)))
            top = int(y[np.argsort(-jitter(combined.values))[:k]].sum())
            deltas = []
            for _ in range(DRAWS):
                idx = np.concatenate([members[d] for d in rng.choice(unique, len(unique))])
                if y[idx].min() == y[idx].max():
                    continue
                deltas.append(weighted_auc(jitter(combined.values[idx]), y[idx], np.ones(len(idx)))
                              - weighted_auc(jitter(frame.atoms.values[idx]), y[idx], np.ones(len(idx))))
            lo, hi = np.percentile(deltas, [5, 95])
            print(f"  атомы + {s}: AUC {auc:.3f} (Δ {auc - base_auc:+.3f}, 5–95% {lo:+.3f}…{hi:+.3f}), ошибок в топ-{k}: {top}")
            decisions.setdefault(s, {}).update({"calls_dAUC_low": lo, "calls_top_gain": top - base_top})
    print("\n=== ВЕРДИКТ (правило записано до прогона)")
    for s, d in decisions.items():
        text_ok = d.get("tau2", 0) >= 1.05 and d.get("tau2_10%", 0) > 1.0 and d.get("dev", 1.0) >= 1.0
        calls_ok = d.get("calls_dAUC_low", -1) > 0 and d.get("calls_top_gain", 0) >= 2
        verdict = "БЕРЁМ" + (" (текст)" if text_ok else "") + (" (вызовы)" if calls_ok else "") if (text_ok or calls_ok) else "не берём"
        print(f"  {s:7s}: τ² ×{d.get('tau2', float('nan')):.3f} (10% ×{d.get('tau2_10%', float('nan')):.3f}), "
              f"dev ×{d.get('dev', float('nan')):.3f}; вызовы: нижняя граница ΔAUC {d.get('calls_dAUC_low', float('nan')):+.3f}, "
              f"ошибок в топе {d.get('calls_top_gain', 0):+d} → {verdict}")
    summary["decisions"] = decisions
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
