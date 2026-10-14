"""Сравнение ранжировщиков текстовых ходов на размеченных наборах (dev — разметка v5, tau2 — разметка v6).

Каждый ход взвешивается так, чтобы набор по составу был похож на текстовые ходы паблика: в диапазонах P_base
(≥0.8 / 0.5–0.8 / <0.5) — число ходов паблика (70 / 143 / 167); доля ошибок внутри диапазона — из разметки
(--keep-rates) или подгоняется под измеренную на паблике (0.47 / 0.33 / 0.10, по умолчанию).
Метрики: взвешенный AUC, точность на топ-k (k — число текстовых флагов на паблике), отношение к P_base и
бутстрэп по диалогам для отношения.

python dev/eval_text_rankers.py --set dev|tau2 [--keep-rates] [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from veritas.text_checks import (SIGNAL_NAMES, earlier_error_in_history, find_text_signals,  # noqa: E402
                                 has_process_signal)
from veritas.tool_schema import parse_available_tools  # noqa: E402
from veritas.transcript import AGENT, parse_agent_turn, parse_context  # noqa: E402

PUBLIC_BINS = ((0.8, 9.0, 0.47, 70), (0.5, 0.8, 0.33, 143), (-9.0, 0.5, 0.10, 167))
# группы лёгких зондов v5 на паблике (зонды process / history, 08.10): ходов и доля ошибок у организаторов
PUBLIC_GROUPS = {"process": (50, 0.32), "history": (123, 0.31), "rest": (207, 0.20)}
# сдвиг логита: логарифм отношения шансов группы к «остальным» по зондам паблика
GROUP_LOGIT_SHIFT = {group: float(np.log(rate / (1 - rate)) - np.log(0.20 / 0.80))
                     for group, (_, rate) in PUBLIC_GROUPS.items() if group != "rest"}
LOOP_SIMILARITY = 0.5                  # в данных организаторов похожесть на прошлые реплики не выше 0.35
WORD = re.compile(r"\w{3,}")
SIGNALS_DIR = PROJECT / "colab_results" / "v5_signals" / "v5_signals" / "signals"
EXTRA_SIGNALS_DIR = PROJECT / "colab_results" / "v5b_spare_time" / "v5b_spare_time" / "signals"   # claims, think
SETS = {
    "dev": dict(signals=SIGNALS_DIR / "dev_signals.parquet", examples=ROOT / "dev" / "data" / "dev.parquet",
                labels=PROJECT / "labeling_v5" / "out", units=PROJECT / "labeling_v5" / "units.json"),
    "tau2": dict(signals=SIGNALS_DIR / "tau2_signals.parquet",
                 examples=PROJECT / "colab_results" / "v5_tau2gen" / "v5_tau2gen" / "examples_v5.parquet",
                 labels=PROJECT / "labeling_v6" / "out", units=PROJECT / "labeling_v6" / "units.json"),
}
ATOMS = ("facts", "action_claim", "reask", "refusal", "policy", "overall")
CJK = re.compile(r"[一-鿿]")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_labels(directory: Path) -> pd.DataFrame:
    rows = [json.loads(line) for path in sorted(directory.glob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows).drop_duplicates("id", keep="last")[["id", "label", "category"]]


def load_text_frame(name: str) -> pd.DataFrame:
    paths = SETS[name]
    units = pd.DataFrame(json.loads(paths["units"].read_text(encoding="utf-8")))
    labels = load_labels(paths["labels"])
    frame = units[units.kind == "text"].merge(labels, on="id")
    if name == "dev":                                  # у ходов valid — метки организаторов
        organizer = pd.read_parquet(PROJECT / "valid.parquet").set_index("id")["label"]
        frame["label"] = [int(organizer[i]) if source == "valid" else int(label)
                          for i, source, label in zip(frame.id, frame.src, frame.label)]
    frame["label"] = frame.label.astype(int)
    signals = pd.read_parquet(paths["signals"])
    frame = frame.merge(signals, on="id", how="left")
    extra = EXTRA_SIGNALS_DIR / f"{name}_signals.parquet"
    if extra.exists():
        columns = ["id", "claims_max", "claims_mean", "claims_count", "think_p", "think_tokens", "think_truncated"]
        frame = frame.merge(pd.read_parquet(extra)[columns], on="id", how="left")
    examples = pd.read_parquet(paths["examples"]).set_index("id")
    for column in SIGNAL_NAMES:
        frame[column] = 0
    for position, example_id in enumerate(frame.id):
        found = find_text_signals(parse_context(examples.prompt[example_id], example_id),
                                  parse_agent_turn(examples.response[example_id]))
        for signal in found:
            frame.loc[frame.index[position], signal.name] = 1
    frame["dialog"] = frame.id.str.split("::").str[0]
    frame["domain"] = frame.id.str.split("__").str[0]
    frame["cjk"] = [bool(CJK.search(examples.response[i])) for i in frame.id]
    frame["customer_russian"] = [_customer_russian_share(examples.prompt[i]) for i in frame.id]
    groups = [_probe_group(examples.prompt[i], examples.response[i], i) for i in frame.id]
    frame["group"] = [group for group, _ in groups]
    frame["loop"] = [loop for _, loop in groups]
    return frame.dropna(subset=["p_base"]).reset_index(drop=True)


def _probe_group(prompt: str, response: str, example_id: str) -> tuple:
    """(группа лёгких зондов: process / history / rest, петля — дословный повтор прошлой реплики агента)."""
    transcript, turn = parse_context(prompt, example_id), parse_agent_turn(response)
    if has_process_signal(transcript, turn):
        group = "process"
    elif earlier_error_in_history(transcript, parse_available_tools(transcript.tools_text)):
        group = "history"
    else:
        group = "rest"
    words = set(WORD.findall(turn.text.lower()))
    earlier = [set(WORD.findall(past.text.lower())) for past in transcript.turns
               if past.role == AGENT and past.text and past.number is not None]
    loop = bool(words) and any(len(words & past) / len(words | past) >= LOOP_SIMILARITY for past in earlier if past)
    return group, loop


def _customer_russian_share(prompt: str) -> float:
    """Доля кириллицы среди букв в репликах клиента (новые диалоги τ² бывают на английском)."""
    blocks = [block.split("⟦", 1)[0] for block in prompt.split("⟦USER⟧")[1:]]   # текст до следующего маркера
    text = " ".join(line for block in blocks for line in block.splitlines() if not line.startswith("\t"))
    cyrillic, latin = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    return cyrillic / (cyrillic + latin) if cyrillic + latin else 1.0


def add_combinations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    atoms = frame[[f"text_{a}" for a in ATOMS]].fillna(0.0)
    frame["atoms_mean"] = atoms.mean(axis=1)
    frame["atoms_max"] = atoms.drop(columns=["text_overall"]).max(axis=1)
    process = [n for n in SIGNAL_NAMES if n != "offers_transfer"]
    frame["checks_any"] = frame[process].max(axis=1)
    frame["base+v1+v2"] = logit(frame.p_base) + logit(frame.p_v1) + logit(frame.p_v2)
    frame["base+overall"] = logit(frame.p_base) + logit(frame.text_overall)
    frame["base+atoms"] = logit(frame.p_base) + logit(frame.atoms_mean)
    frame["base+overall+checks"] = frame["base+overall"] + 3.0 * frame.checks_any + 1.0 * frame.offers_transfer
    frame["all_llm"] = frame["base+v1+v2"] + logit(frame.text_overall) + logit(frame.atoms_mean)
    frame["all_llm+checks"] = frame["all_llm"] + 3.0 * frame.checks_any + 1.0 * frame.offers_transfer
    prior = np.zeros(len(frame))
    for group, shift in GROUP_LOGIT_SHIFT.items():
        prior = prior + shift * (frame.group == group).values
    if "claims_max" in frame:
        frame["base+claims"] = logit(frame.p_base) + logit(frame.claims_max.fillna(0.0))
        frame["base+think"] = logit(frame.p_base) + logit(frame.think_p.fillna(0.5))
        frame["base+atoms+claims"] = frame["base+atoms"] + logit(frame.claims_max.fillna(0.0))
        frame["base+atoms+think"] = frame["base+atoms"] + logit(frame.think_p.fillna(0.5))
        frame["base+atoms+groups+claims"] = frame["base+atoms"] + prior + logit(frame.claims_max.fillna(0.0))
    frame["base+groups"] = logit(frame.p_base) + prior
    frame["base+atoms+groups"] = frame["base+atoms"] + prior
    frame["atoms_max+groups"] = logit(frame.atoms_max) + prior
    return frame


def public_weights(frame: pd.DataFrame, keep_rates: bool) -> np.ndarray:
    weights = np.zeros(len(frame))
    for low, high, rate, count in PUBLIC_BINS:
        in_bin = ((frame.p_base >= low) & (frame.p_base < high)).values
        if in_bin.sum() == 0:
            continue
        own_rate = frame.label.values[in_bin].mean()
        positives, negatives = in_bin & (frame.label.values == 1), in_bin & (frame.label.values == 0)
        if keep_rates or own_rate in (0.0, 1.0):
            weights[in_bin] = count / in_bin.sum()
        else:
            weights[positives] = count / in_bin.sum() * rate / own_rate
            weights[negatives] = count / in_bin.sum() * (1 - rate) / (1 - own_rate)
    return weights


def rake_weights(frame: pd.DataFrame, iterations: int = 50) -> np.ndarray:
    """Веса, при которых и диапазоны P_base, и группы зондов совпадают с пабликом по числу ходов и доле ошибок."""
    labels = frame.label.values
    targets = []
    for low, high, rate, count in PUBLIC_BINS:
        cell = ((frame.p_base >= low) & (frame.p_base < high)).values
        targets += [(cell & (labels == 1), count * rate), (cell & (labels == 0), count * (1 - rate))]
    for group, (count, rate) in PUBLIC_GROUPS.items():
        cell = (frame.group == group).values
        targets += [(cell & (labels == 1), count * rate), (cell & (labels == 0), count * (1 - rate))]
    weights = np.ones(len(frame))
    for _ in range(iterations):
        for mask, total in targets:
            current = weights[mask].sum()
            if current > 0:
                weights[mask] *= total / current
    return weights


def make_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    return rake_weights(frame) if mode == "rake" else public_weights(frame, keep_rates=(mode == "keep"))


def weighted_auc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    positive, negative = labels == 1, labels == 0
    order = np.argsort(scores[negative], kind="mergesort")
    neg_scores, neg_weights = scores[negative][order], weights[negative][order]
    cumulative = np.concatenate([[0.0], np.cumsum(neg_weights)])
    total = 0.0
    for score, weight in zip(scores[positive], weights[positive]):
        below = np.searchsorted(neg_scores, score, side="left")
        equal = np.searchsorted(neg_scores, score, side="right") - below
        total += weight * (cumulative[below] + 0.5 * (cumulative[below + equal] - cumulative[below]))
    return total / (weights[positive].sum() * weights[negative].sum())


def precision_at(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, flags: float) -> float:
    order = np.argsort(-scores, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    chosen = order[cumulative <= flags]
    return float((weights[chosen] * labels[chosen]).sum() / max(weights[chosen].sum(), 1e-9))


def ranking_scores(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Оценки сигнала с детерминированным разрывом ничьих (одинаково для всех сигналов и для эталона)."""
    scores = frame[column].fillna(frame[column].median()).values.astype(float)
    return scores + 1e-9 * np.arange(len(scores)) / len(scores)


def evaluate(frame: pd.DataFrame, columns: list, mode: str) -> pd.DataFrame:
    weights = make_weights(frame, mode)
    labels = frame.label.values
    rows = []
    reference = {k: precision_at(ranking_scores(frame, "p_base"), labels, weights, k) for k in (130, 160)}
    for column in columns:
        scores = ranking_scores(frame, column)
        rows.append({"сигнал": column, "AUC": weighted_auc(scores, labels, weights),
                     "P@130": precision_at(scores, labels, weights, 130),
                     "P@160": precision_at(scores, labels, weights, 160),
                     "к P_base @160": precision_at(scores, labels, weights, 160) / reference[160]})
    return pd.DataFrame(rows).set_index("сигнал")


def bootstrap_ratio(frame: pd.DataFrame, columns: list, mode: str, draws: int, flags: int = 160) -> dict:
    rng = np.random.default_rng(0)
    dialogs = frame.dialog.unique()
    groups = {dialog: frame.index[frame.dialog == dialog].values for dialog in dialogs}
    ratios = {column: [] for column in columns}
    for _ in range(draws):
        sample = frame.loc[np.concatenate([groups[d] for d in rng.choice(dialogs, len(dialogs))])].reset_index(drop=True)
        weights = make_weights(sample, mode)
        reference = precision_at(ranking_scores(sample, "p_base"), sample.label.values, weights, flags)
        for column in columns:
            scores = ranking_scores(sample, column)
            ratios[column].append(precision_at(scores, sample.label.values, weights, flags) / max(reference, 1e-9))
    return {column: np.percentile(values, [10, 50, 90]) for column, values in ratios.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=list(SETS), default="dev")
    parser.add_argument("--keep-rates", action="store_true", help="не подгонять долю ошибок в диапазонах P_base")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--clean", action="store_true",
                        help="без ходов с китайским текстом и без диалогов, где клиент пишет не по-русски")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--rake", action="store_true",
                        help="веса сразу под диапазоны P_base и группы лёгких зондов паблика (process/history/rest)")
    parser.add_argument("--no-loops", action="store_true", help="без дословных повторов (у организаторов их нет)")
    args = parser.parse_args()
    mode = "rake" if args.rake else ("keep" if args.keep_rates else "bins")

    frame = add_combinations(load_text_frame(args.set))
    if args.clean:
        frame = frame[~frame.cjk & (frame.customer_russian >= 0.5)].reset_index(drop=True)
    if args.domain:
        frame = frame[frame.domain == args.domain].reset_index(drop=True)
    if args.no_loops:
        frame = frame[~frame.loop].reset_index(drop=True)
    print(f"{args.set}: текстовых ходов {len(frame)}, ошибок {int(frame.label.sum())}, диалогов {frame.dialog.nunique()}")
    for low, high, _, _ in PUBLIC_BINS:
        part = frame[(frame.p_base >= low) & (frame.p_base < high)]
        print(f"  P_base [{max(low, 0):.1f}; {min(high, 1):.1f}): ходов {len(part)}, доля ошибок "
              f"{part.label.mean() if len(part) else float('nan'):.2f}")
    columns = (["p_base", "p_v1", "p_v2"] + [f"text_{a}" for a in ATOMS]
               + ["atoms_mean", "atoms_max", "checks_any", "base+v1+v2", "base+overall", "base+atoms",
                  "base+overall+checks", "all_llm", "all_llm+checks", "base+groups", "base+atoms+groups",
                  "atoms_max+groups"])
    if "claims_max" in frame:
        columns += ["claims_max", "claims_mean", "think_p", "base+claims", "base+think", "base+atoms+claims",
                    "base+atoms+think", "base+atoms+groups+claims"]
    table = evaluate(frame, columns, mode)
    print(table.round(3).to_string())
    print("\nAUC по доменам (без перевзвешивания):")
    main_columns = ["p_base", "p_v2", "text_overall", "text_action_claim", "atoms_mean", "base+overall", "base+v1+v2"]
    print(f"  {'домен':18s} {'ходов':>5s} {'ошибок':>6s} " + " ".join(f"{c[:12]:>12s}" for c in main_columns))
    for domain, part in frame.groupby("domain"):
        if part.label.nunique() < 2:
            continue
        flat = np.ones(len(part))
        aucs = [weighted_auc(ranking_scores(part.reset_index(drop=True), c), part.label.values, flat)
                for c in main_columns]
        print(f"  {domain:18s} {len(part):5d} {int(part.label.sum()):6d} " + " ".join(f"{a:12.3f}" for a in aucs))
    if args.bootstrap:
        print(f"\nбутстрэп по диалогам ({args.bootstrap}), отношение к P_base @160: 10% / медиана / 90%")
        for column, (low, median, high) in bootstrap_ratio(frame, columns[1:], mode, args.bootstrap).items():
            print(f"  {column:22s} ×{low:.2f} / ×{median:.2f} / ×{high:.2f}")


if __name__ == "__main__":
    main()
