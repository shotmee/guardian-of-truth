"""Оценка сигналов v6 (base / meta / shots на Qwen и GLM) на размеченных ходах.

Текст: P@160 и AUC с весами под состав текстовых ходов паблика (рейкинг; петли τ² — вес 0), отношение к P_base,
бутстрэп по диалогам для отношения. Вызовы (τ², без hard): AUC и точность топ-20%. Valid (46 ходов с метками
организаторов): AUC — мало, но это единственные «их» метки.
Ходы без найденной задачи в meta получают значение base (так сработает и решение).
python dev/eval_v6.py --labels labels.json --signals <папка с *_signals.parquet> [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FLAGS, CALL_SHARE, DRAWS = 160, 0.203, 400


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def weighted_auc(scores, labels, weights):
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores[neg], kind="mergesort")
    ns, nw = scores[neg][order], weights[neg][order]
    cum = np.concatenate([[0.0], np.cumsum(nw)])
    lo = np.searchsorted(ns, scores[pos], side="left")
    hi = np.searchsorted(ns, scores[pos], side="right")
    wins = cum[lo] + 0.5 * (cum[hi] - cum[lo])
    return float((weights[pos] * wins).sum() / (weights[pos].sum() * weights[neg].sum()))


def precision_at(scores, labels, weights, flags):
    order = np.argsort(-scores, kind="mergesort")
    chosen = order[np.cumsum(weights[order]) <= flags]
    return float((weights[chosen] * labels[chosen]).sum() / max(weights[chosen].sum(), 1e-9))


def jitter(values):
    values = np.asarray(values, dtype=float)
    return values + 1e-9 * np.arange(len(values)) / max(1, len(values))


def load_signals(directory: Path) -> pd.DataFrame:
    by_model: dict = {}
    for path in sorted(directory.glob("*_signals.parquet")):
        name, model = path.name.split("_")[:2]                 # <набор>_<модель>_signals.parquet
        table = pd.read_parquet(path)
        table["set"] = name
        by_model.setdefault(model, []).append(table)
    merged = None
    for model, tables in sorted(by_model.items(), key=lambda item: item[0] != "qwen"):
        table = pd.concat(tables, ignore_index=True)
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table.drop(columns=["kind", "meta_how"]), on=["id", "set"], how="outer")
    return merged


def columns_for(frame: pd.DataFrame) -> dict:
    """{название: оценки} — сигналы и их сочетания с атомами и группами."""
    cols = {}
    base = frame["qwen_base"].fillna(frame["p_base_prev"]) if "qwen_base" in frame else frame["p_base_prev"]
    cols["P_base (прежний прогон)"] = logit(frame["p_base_prev"])
    cols["H1: base+атомы+группы (прежний)"] = logit(frame["p_base_prev"]) + logit(frame["atoms_prev"]) + frame["shift"]
    for model in ("qwen", "glm"):
        if f"{model}_base" not in frame or frame[f"{model}_base"].isna().mean() > 0.5:
            continue                    # сигнал модели не посчитан — не подставлять чужие оценки
        b = frame[f"{model}_base"].fillna(base)
        cols[f"{model} base"] = logit(b)
        if f"{model}_meta" in frame:
            m = frame[f"{model}_meta"].fillna(b)
            cols[f"{model} meta"] = logit(m)
            cols[f"{model} base+meta"] = logit(b) + logit(m)
            cols[f"{model} meta+атомы+группы"] = logit(m) + logit(frame["atoms_prev"]) + frame["shift"]
            cols[f"{model} base+meta+атомы+группы"] = logit(b) + logit(m) + logit(frame["atoms_prev"]) + frame["shift"]
        if f"{model}_shots" in frame:
            s = frame[f"{model}_shots"].fillna(b)
            cols[f"{model} shots"] = logit(s)
            cols[f"{model} shots+атомы+группы"] = logit(s) + logit(frame["atoms_prev"]) + frame["shift"]
            if f"{model}_meta" in frame:
                cols[f"{model} shots+meta+атомы+группы"] = (logit(s) + logit(frame[f"{model}_meta"].fillna(b))
                                                           + logit(frame["atoms_prev"]) + frame["shift"])
    if "qwen_meta" in frame and "glm_meta" in frame and frame["glm_meta"].isna().mean() <= 0.5:
        cols["qwen meta + glm meta"] = logit(frame["qwen_meta"].fillna(base)) + logit(frame["glm_meta"].fillna(base))
    return cols


def text_report(frame: pd.DataFrame, rng) -> pd.DataFrame:
    frame = frame[frame.weight > 0].reset_index(drop=True)
    labels, weights = frame.label.values, frame.weight.values
    cols = {k: jitter(np.nan_to_num(v, nan=0.0)) for k, v in columns_for(frame).items()}
    reference = precision_at(cols["P_base (прежний прогон)"], labels, weights, FLAGS)
    dialogs = frame.dialog.values
    unique = np.unique(dialogs)
    members = {d: np.flatnonzero(dialogs == d) for d in unique}
    samples = {k: [] for k in cols}
    for _ in range(DRAWS):
        idx = np.concatenate([members[d] for d in rng.choice(unique, len(unique))])
        ref = precision_at(cols["P_base (прежний прогон)"][idx], labels[idx], weights[idx], FLAGS)
        for k, v in cols.items():
            samples[k].append(precision_at(v[idx], labels[idx], weights[idx], FLAGS) / max(ref, 1e-9))
    rows = []
    for k, v in cols.items():
        p = precision_at(v, labels, weights, FLAGS)
        lo, hi = np.percentile(samples[k], [10, 90])
        rows.append({"сигнал": k, "AUC": weighted_auc(v, labels, weights), "P@160": p, "к P_base": p / reference,
                     "10%": lo, "90%": hi})
    return pd.DataFrame(rows).set_index("сигнал")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    signals = load_signals(Path(args.signals))
    if signals is None:
        print("нет сигналов"); return
    rng = np.random.default_rng(0)
    summary = {}
    pd.set_option("display.width", 200)
    print("=" * 110)
    for name in ("tau2", "dev"):
        text = pd.DataFrame([r for r in labels["text"] if r["set"] == name])
        frame = text.merge(signals[signals.set == name].drop(columns=["set"]), on="id", how="left")
        if frame.empty or not any(c.endswith("_base") for c in frame.columns):
            continue
        cover = frame[frame.weight > 0].meta_how.value_counts().to_dict() if "meta_how" in frame else {}
        print(f"\nТЕКСТ [{name}]: ходов {int((frame.weight > 0).sum())}, ошибок {int(frame[frame.weight > 0].label.sum())}; "
              f"метаданные: {cover}")
        table = text_report(frame, rng)
        print(table.round(3).to_string())
        summary[f"text_{name}"] = table.round(4).to_dict(orient="index")
    calls = pd.DataFrame(labels["calls"])
    if not calls.empty:
        frame = calls.merge(signals[signals.set == "tau2"].drop(columns=["set"]), on="id", how="left")
        y = frame.label.values
        out = {}
        variants = {"атомы вызовов (прежний прогон)": logit(frame.calls_prev.fillna(0.5))}
        for model in ("qwen", "glm"):
            for signal in ("base", "meta", "shots"):
                column = f"{model}_{signal}"
                if column in frame and frame[column].isna().mean() <= 0.5:
                    filled = frame[column].fillna(frame.get(f"{model}_base", pd.Series(0.5, index=frame.index))).fillna(0.5)
                    variants[f"{model} {signal}"] = logit(filled)
                    if signal == "meta":
                        variants[f"атомы + {model} meta"] = logit(frame.calls_prev.fillna(0.5)) + logit(filled)
        print(f"\nВЫЗОВЫ [tau2]: ходов {len(frame)}, ошибок {int(y.sum())}")
        for k, v in variants.items():
            v = jitter(np.nan_to_num(v))
            order = np.argsort(-v)
            top = y[order[: max(1, int(round(CALL_SHARE * len(v))))]].mean()
            out[k] = {"AUC": weighted_auc(v, y, np.ones(len(y))), "top20%": float(top)}
            print(f"  {k:36s} AUC {out[k]['AUC']:.3f}   точность топ-20% {top:.3f}")
        summary["calls_tau2"] = out
    valid = pd.DataFrame(labels.get("valid", []))
    if not valid.empty:
        frame = valid.merge(signals[signals.set == "dev"].drop(columns=["set"]), on="id", how="left")
        print(f"\nVALID (метки организаторов): ходов {len(frame)}, ошибок {int(frame.label.sum())}")
        out = {}
        for column in [c for c in frame.columns if c.startswith(("qwen_", "glm_")) and frame[c].isna().mean() <= 0.5]:
            base_col = column.split("_")[0] + "_base"
            v = frame[column].fillna(frame[base_col] if base_col in frame else 0.5).fillna(0.5)
            out[column] = weighted_auc(jitter(v.values), frame.label.values, np.ones(len(frame)))
            print(f"  {column:14s} AUC {out[column]:.3f}")
        summary["valid"] = out
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
