"""Согласие первой (labeling/out) и второй (labeling/out2) разметки пула агентами Claude.
Пишет labeling/labels_consensus.json: id -> {"l1", "l2", "final", "w"}, где
  final — метка первого прохода, если проходы согласны; при расхождении — 1 только если оба прохода 1 (иначе 0.5 => «спорный»);
  w — вес в обучении стекинга (1.0 согласие, 0.5 спорный, 1.0 для ходов без второго прохода).
И labeling/labels_final.jsonl — метки первого прохода без спорных ходов; их используют eval_pool, feat_lab,
ноутбуки веток и калибровка симулятора (lbsim.CAL).
Запуск: python tools/label_agreement.py"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "labeling"


def load(d):
    out = {}
    for f in sorted((LAB / d).glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[r["id"]] = r
    return out


def main():
    a, b = load("out"), load("out2")
    common = sorted(set(a) & set(b))
    miss = sorted(set(b) - set(a))
    if miss:
        print("ВНИМАНИЕ: id второго прохода, которых нет в первом:", miss[:10])
    cm = Counter((a[i]["label"], b[i]["label"]) for i in common)
    n = len(common)
    agree = cm[(0, 0)] + cm[(1, 1)]
    po = agree / max(n, 1)
    p1a = (cm[(1, 0)] + cm[(1, 1)]) / max(n, 1)
    p1b = (cm[(0, 1)] + cm[(1, 1)]) / max(n, 1)
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"пересечение: {n} ходов | согласие {agree}/{n} = {po:.3f} | kappa {kappa:.3f}")
    print(f"  1→1 {cm[(1, 1)]}  1→0 {cm[(1, 0)]}  0→1 {cm[(0, 1)]}  0→0 {cm[(0, 0)]}")
    by_dom = Counter()
    dis_dom = Counter()
    for i in common:
        d = i.split("__")[0]
        by_dom[d] += 1
        dis_dom[d] += a[i]["label"] != b[i]["label"]
    for d in sorted(by_dom):
        print(f"  {d:18s} расхождений {dis_dom[d]}/{by_dom[d]}")
    print("\nРасхождения:")
    for i in common:
        if a[i]["label"] != b[i]["label"]:
            print(f"- {i}: 1-й={a[i]['label']} ({a[i].get('cat')}: {a[i].get('why', '')[:90]})")
            print(f"  {' ' * len(i)}  2-й={b[i]['label']} ({b[i].get('cat')}: {b[i].get('why', '')[:90]})")
    cons = {}
    for i, r in a.items():
        if i in b:
            l2 = b[i]["label"]
            same = r["label"] == l2
            cons[i] = {"l1": r["label"], "l2": l2, "final": r["label"] if same else 0.5, "w": 1.0 if same else 0.5}
        else:
            cons[i] = {"l1": r["label"], "l2": None, "final": r["label"], "w": 1.0}
    (LAB / "labels_consensus.json").write_text(json.dumps(cons, ensure_ascii=False, indent=0), encoding="utf-8")
    # итоговые метки для обучения и симулятора: спорные (проходы разошлись) исключаем
    final = [r for i, r in a.items() if cons[i]["w"] == 1.0]
    with open(LAB / "labels_final.jsonl", "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps({k: r[k] for k in ("id", "label", "cat", "why") if k in r}, ensure_ascii=False) + "\n")
    print(f"labels_final.jsonl: {len(final)} ходов (исключено спорных {len(a) - len(final)}), "
          f"позитивов {sum(r['label'] for r in final)}")
    pos1 = sum(r["label"] for r in a.values())
    pos_strict = sum(1 for c in cons.values() if c["final"] == 1)
    pos_soft = sum(1 for c in cons.values() if c["final"] >= 0.5)
    print(f"\nпозитивов: 1-й проход {pos1}/{len(a)}; согласованных {pos_strict}; включая спорные {pos_soft}")
    print("->", LAB / "labels_consensus.json")


if __name__ == "__main__":
    main()
