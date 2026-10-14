"""Сравнение двух наборов формулировок промптов (old — прежние, new — переписанные) на размеченных ходах.

Входы:
  --labels  labels.json (собирает dev/make_prompt_check_notebook.py):
            текстовые ходы dev и τ² с метками, весами под состав паблика (рейкинг, петли τ² — вес 0),
            группой зондов и прежними оценками (прогон v5_signals, тот же промпт, что old);
            ходы τ² с вызовами (без срабатываний проверок) с метками и прежней оценкой атомов.
  --dumps   папка с dump_{dev,tau2}_{old,new}.json — дампы финального конвейера (scripts/predict.py --dump).

Сравниваются ранжировщики, которые реально пойдут на платформу:
  H2 текст — P_base (baseline.md); H1 текст — logit P_base + logit(среднее текстовых атомов) + сдвиг группы
  (baseline.md + text_standard.md); вызовы — атомы (standard.md).
Метрики: точность на топ-160 текстовых ходов паблика (P@160) и AUC; для вызовов — AUC и точность топ-20%.
Шум: парный бутстрэп по диалогам (разница new − old) и разница повторного прогона (old − прежний прогон).

Правило решения (записано до прогона), отдельно для каждого файла промпта:
  интервал 5–95% разницы new − old целиком ниже 0 — новый хуже, вернуть прежний;
  целиком выше 0 — новый лучше; иначе — разница в пределах шума, можно оставить новый.
python dev/prompt_eval.py --labels labels.json --dumps <папка> [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FLAGS = 160                            # текстовых флагов на паблике (из 380 текстовых ходов)
CALL_SHARE = 0.203                     # доля флагуемых мягких ходов с вызовами
DRAWS = 1000


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def weighted_auc(scores, labels, weights) -> float:
    positive, negative = labels == 1, labels == 0
    order = np.argsort(scores[negative], kind="mergesort")
    neg_scores, neg_weights = scores[negative][order], weights[negative][order]
    cumulative = np.concatenate([[0.0], np.cumsum(neg_weights)])
    below = np.searchsorted(neg_scores, scores[positive], side="left")
    equal_end = np.searchsorted(neg_scores, scores[positive], side="right")
    wins = cumulative[below] + 0.5 * (cumulative[equal_end] - cumulative[below])
    total = (weights[positive] * wins).sum()
    return float(total / max(weights[positive].sum() * weights[negative].sum(), 1e-12))


def precision_at(scores, labels, weights, flags) -> float:
    order = np.argsort(-scores, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    chosen = order[cumulative <= flags]
    return float((weights[chosen] * labels[chosen]).sum() / max(weights[chosen].sum(), 1e-9))


def top_share_precision(scores, labels, share) -> float:
    order = np.argsort(-scores, kind="mergesort")
    return float(labels[order[: max(1, int(round(share * len(scores))))]].mean())


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def tie_break(scores: np.ndarray) -> np.ndarray:
    return scores + 1e-9 * np.arange(len(scores)) / max(1, len(scores))


def load_dump(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ----------------------------------------------------------------------------- текст
def text_columns(rows: list, dumps: dict) -> dict:
    """{(ранжировщик, вариант): оценки} для строк rows; вариант prev — прежний прогон v5_signals."""
    shift = np.array([row["shift"] for row in rows])
    base = {"prev": np.array([row["p_base_prev"] for row in rows], dtype=float)}
    atoms = {"prev": np.array([row["atoms_prev"] for row in rows], dtype=float)}
    for variant, dump in dumps.items():
        base[variant] = np.array([dump.get("base", {}).get(row["id"], np.nan) for row in rows], dtype=float)
        atoms[variant] = np.array([(dump.get("text_atoms", {}).get(row["id"]) or {}).get("score", np.nan)
                                   for row in rows], dtype=float)
    columns = {}
    for variant in base:
        columns[("H2", variant)] = base[variant]
        columns[("H1", variant)] = logit(base[variant]) + logit(atoms[variant]) + shift
        columns[("atoms", variant)] = atoms[variant]
    if "old" in base and "new" in base:     # смешанные: какой из двух файлов промпта даёт разницу
        columns[("H1", "new_text_standard_only")] = logit(base["old"]) + logit(atoms["new"]) + shift
        columns[("H1", "new_baseline_only")] = logit(base["new"]) + logit(atoms["old"]) + shift
    return columns


def text_report(name: str, rows: list, dumps: dict, rng: np.random.Generator) -> dict:
    rows = [row for row in rows if row["weight"] > 0]
    labels = np.array([row["label"] for row in rows])
    weights = np.array([row["weight"] for row in rows], dtype=float)
    dialogs = np.array([row["dialog"] for row in rows])
    columns = text_columns(rows, dumps)
    missing = {key: int(np.isnan(values).sum()) for key, values in columns.items() if np.isnan(values).any()}
    columns = {key: tie_break(np.where(np.isnan(values), np.nanmedian(values), values))
               for key, values in columns.items()}
    report = {"set": name, "turns": len(rows), "errors": int(labels.sum()), "missing": {f"{k[0]}/{k[1]}": v for k, v in missing.items()},
              "metrics": {}, "delta": {}, "spearman": {}}
    for (ranker, variant), scores in columns.items():
        report["metrics"][f"{ranker}/{variant}"] = {"AUC": weighted_auc(scores, labels, weights),
                                                    "P@160": precision_at(scores, labels, weights, FLAGS)}
    unique = np.unique(dialogs)
    members = {dialog: np.flatnonzero(dialogs == dialog) for dialog in unique}
    pairs = [("H2", "new", "old"), ("H1", "new", "old"), ("H1", "new_text_standard_only", "old"),
             ("H1", "new_baseline_only", "old"), ("atoms", "new", "old"), ("H2", "old", "prev"), ("H1", "old", "prev")]
    pairs = [pair for pair in pairs if (pair[0], pair[1]) in columns and (pair[0], pair[2]) in columns]
    samples = {pair: [] for pair in pairs}
    for _ in range(DRAWS):
        index = np.concatenate([members[d] for d in rng.choice(unique, len(unique))])
        for ranker, a, b in pairs:
            pa = precision_at(columns[(ranker, a)][index], labels[index], weights[index], FLAGS)
            pb = precision_at(columns[(ranker, b)][index], labels[index], weights[index], FLAGS)
            samples[(ranker, a, b)].append(pa - pb)
    for (ranker, a, b), values in samples.items():
        point = report["metrics"][f"{ranker}/{a}"]["P@160"] - report["metrics"][f"{ranker}/{b}"]["P@160"]
        low, high = np.percentile(values, [5, 95])
        report["delta"][f"{ranker}: {a} − {b}"] = {"P@160": point, "low": float(low), "high": float(high),
                                                   "TP_on_public": point * FLAGS}
    for ranker in ("H2", "atoms"):
        for a, b in (("new", "old"), ("old", "prev")):
            if (ranker, a) in columns and (ranker, b) in columns:
                report["spearman"][f"{ranker}: {a} ~ {b}"] = spearman(columns[(ranker, a)], columns[(ranker, b)])
    return report


# ----------------------------------------------------------------------------- вызовы
def call_report(rows: list, dumps: dict, rng: np.random.Generator) -> dict:
    labels = np.array([row["label"] for row in rows])
    dialogs = np.array([row["dialog"] for row in rows])
    columns = {"prev": np.array([row["calls_prev"] if row["calls_prev"] is not None else np.nan for row in rows], dtype=float)}
    for variant, dump in dumps.items():
        columns[variant] = np.array([dump.get("calls", {}).get(row["id"], np.nan) for row in rows], dtype=float)
    missing = {k: int(np.isnan(v).sum()) for k, v in columns.items() if np.isnan(v).any()}
    columns = {k: tie_break(np.where(np.isnan(v), np.nanmedian(v), v)) for k, v in columns.items()}
    flat = np.ones(len(rows))
    report = {"turns": len(rows), "errors": int(labels.sum()), "missing": missing, "metrics": {}, "delta": {}, "spearman": {}}
    for variant, scores in columns.items():
        report["metrics"][variant] = {"AUC": weighted_auc(scores, labels, flat),
                                      "top20%": top_share_precision(scores, labels, CALL_SHARE)}
    unique = np.unique(dialogs)
    members = {dialog: np.flatnonzero(dialogs == dialog) for dialog in unique}
    for a, b in (("new", "old"), ("old", "prev")):
        if a not in columns or b not in columns:
            continue
        values = []
        for _ in range(DRAWS):
            index = np.concatenate([members[d] for d in rng.choice(unique, len(unique))])
            if labels[index].min() == labels[index].max():
                continue
            values.append(weighted_auc(columns[a][index], labels[index], flat[index])
                          - weighted_auc(columns[b][index], labels[index], flat[index]))
        low, high = np.percentile(values, [5, 95])
        report["delta"][f"{a} − {b}"] = {"AUC": report["metrics"][a]["AUC"] - report["metrics"][b]["AUC"],
                                         "low": float(low), "high": float(high)}
        report["spearman"][f"{a} ~ {b}"] = spearman(columns[a], columns[b])
    return report


# ----------------------------------------------------------------------------- вердикт
def verdict(low: float, high: float) -> str:
    if high < 0:
        return "НОВЫЙ ХУЖЕ — вернуть прежний"
    if low > 0:
        return "новый лучше"
    return "в пределах шума — можно оставить новый"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    dumps_dir = Path(args.dumps)
    rng = np.random.default_rng(0)

    summary = {"text": {}, "calls": None}
    for name in ("tau2", "dev"):
        dumps = {variant: load_dump(dumps_dir / f"dump_{name}_{variant}.json") for variant in ("old", "new")}
        dumps = {variant: dump for variant, dump in dumps.items() if dump}
        rows = [row for row in labels["text"] if row["set"] == name]
        if not dumps or not rows:
            print(f"{name}: нет дампов или меток — пропускаю")
            continue
        summary["text"][name] = text_report(name, rows, dumps, rng)
    tau2_dumps = {v: load_dump(dumps_dir / f"dump_tau2_{v}.json") for v in ("old", "new")}
    tau2_dumps = {v: d for v, d in tau2_dumps.items() if d}
    if tau2_dumps and labels["calls"]:
        summary["calls"] = call_report(labels["calls"], tau2_dumps, rng)

    print("=" * 100)
    print("ТЕКСТОВЫЕ ХОДЫ: точность на топ-160 (P@160) и AUC; веса — состав текстовых ходов паблика")
    for name, report in summary["text"].items():
        print(f"\n[{name}] ходов {report['turns']}, ошибок {report['errors']}" +
              (f"; без оценки: {report['missing']}" if report["missing"] else ""))
        print(f"  {'ранжировщик / вариант':38s} {'P@160':>7s} {'AUC':>7s}")
        for key, values in report["metrics"].items():
            print(f"  {key:38s} {values['P@160']:7.3f} {values['AUC']:7.3f}")
        print("  разница P@160 (5% … 95%, бутстрэп по диалогам), в TP на паблике (×160):")
        for key, values in report["delta"].items():
            print(f"    {key:38s} {values['P@160']:+.3f} ({values['low']:+.3f} … {values['high']:+.3f}) "
                  f"≈ {values['TP_on_public']:+.1f} TP")
        print("  согласие порядков (Спирмен): " + ", ".join(f"{k} {v:.3f}" for k, v in report["spearman"].items()))
    if summary["calls"]:
        report = summary["calls"]
        print(f"\nХОДЫ С ВЫЗОВАМИ (τ², без срабатываний проверок): ходов {report['turns']}, ошибок {report['errors']}" +
              (f"; без оценки: {report['missing']}" if report["missing"] else ""))
        for variant, values in report["metrics"].items():
            print(f"  {variant:6s} AUC {values['AUC']:.3f}, точность топ-20% {values['top20%']:.3f}")
        for key, values in report["delta"].items():
            print(f"  разница AUC {key}: {values['AUC']:+.3f} ({values['low']:+.3f} … {values['high']:+.3f})")
        print("  согласие порядков: " + ", ".join(f"{k} {v:.3f}" for k, v in report["spearman"].items()))

    print("\n" + "=" * 100)
    print("ВЕРДИКТ (правило записано до прогона; главный набор — τ², dev — для сверки):")
    decisions = {}
    main_set = summary["text"].get("tau2") or summary["text"].get("dev")
    if main_set:
        for file_name, key in (("baseline.md (P_base, H1 и H2)", "H2: new − old"),
                               ("text_standard.md (текстовые атомы, только H1)", "H1: new_text_standard_only − old")):
            if key in main_set["delta"]:
                values = main_set["delta"][key]
                decisions[file_name] = verdict(values["low"], values["high"])
                check = summary["text"].get("dev", {}).get("delta", {}).get(key)
                extra = f"; dev {check['P@160']:+.3f}" if check and main_set["set"] != "dev" else ""
                print(f"  {file_name:48s} {decisions[file_name]}  [τ² {values['P@160']:+.3f}{extra}]")
    if summary["calls"] and "new − old" in summary["calls"]["delta"]:
        values = summary["calls"]["delta"]["new − old"]
        decisions["standard.md (атомы вызовов)"] = verdict(values["low"], values["high"])
        print(f"  {'standard.md (атомы вызовов)':48s} {decisions['standard.md (атомы вызовов)']}  [AUC {values['AUC']:+.3f}]")
    if main_set and "H1: new − old" in main_set["delta"]:
        values = main_set["delta"]["H1: new − old"]
        print(f"\n  Итог для H1 целиком (все новые против всех прежних): P@160 {values['P@160']:+.3f} "
              f"({values['low']:+.3f} … {values['high']:+.3f}) ≈ {values['TP_on_public']:+.1f} TP в текстовом блоке паблика")
        reference = main_set["delta"].get("H1: old − prev")
        if reference:
            print(f"  Для масштаба — повторный прогон тех же прежних промптов: {reference['P@160']:+.3f} "
                  f"≈ {reference['TP_on_public']:+.1f} TP (это чистый шум FP8)")
    summary["decisions"] = decisions
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
