"""Боевой config.json v2.1 из дампов ветки A (colab_results/v2_main): фазы A (судья v2) + A_base (промпт бейзлайна),
без рассуждений и без проверки по предложениям; стекинг (логрег) на «переносимых» признаках.
Признаки выбраны по tools/v2_lab.py: на пуле ожидаемый паблик ~0.60, перенос пул -> valid AUC 0.88
(признаки «тип хода» — has_text, has_write, длины, число вызовов — на valid перевёрнуты и исключены).
Порог — лучший по симулятору (mix) на OOF (5 сидов, CV по траекториям). Старый config.json -> config_v1_2.json.
Запуск: python tools/make_config_v2.py [--dry]"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "solution" / "src"))
import v2_lab as L  # noqa: E402
from guardian import features  # noqa: E402

FEATS = ["la", "la_base", "transfer", "claimed_action_no_call", "repeat_call",
         "dom_airline", "dom_banking_knowledge", "dom_retail", "dom_telecom"]
C = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true", help="только показать, не записывать config.json")
a = ap.parse_args()

D, F = L.load()
ev = L.evaluate(D, F, FEATS)
print(f"OOF: AUC {ev['auc']:.3f} | паблик mix {ev['mix']:.3f} / ratio {ev['ratio']:.3f} | порог {ev['t_mix']} | "
      f"пул TPR {ev['tpr_pool']:.3f} FPR {ev['fpr_pool']:.3f} -> паблик TP {ev['TP']:.0f} FP {ev['FP']:.0f}")

hard = D["hard"].astype(bool).values
X = F[FEATS].values.astype(float)[~hard]
y = D["label"].values[~hard]
mu, sd = X.mean(0), X.std(0) + 1e-6
clf = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
stack = {"features": FEATS, "mu": mu.round(6).tolist(), "sd": sd.round(6).tolist(),
         "coef": clf.coef_[0].round(6).tolist(), "intercept": float(round(clf.intercept_[0], 6)),
         "threshold": float(ev["t_mix"])}
print("веса:", dict(zip(FEATS, np.round(clf.coef_[0], 3))), "| свободный член", round(float(clf.intercept_[0]), 3))

# сверка: score из пайплайна == predict_proba sklearn
p_pipe = np.array([features.score({k: F.at[i, k] for k in FEATS}, stack) for i in np.where(~hard)[0]])
p_skl = clf.predict_proba((X - mu) / sd)[:, 1]
assert np.allclose(p_pipe, p_skl, atol=1e-4), np.abs(p_pipe - p_skl).max()
pm = D["in_pool"].values
flag = np.where(hard, 1, 0)
flag[~hard] = (p_skl >= stack["threshold"]).astype(int)
print(f"доля флагов (обучение на всём, in-sample): пул {flag[pm].mean():.3f} | valid {flag[~pm].mean():.3f}")

cfg = json.load(open(ROOT / "colab_results/v2_main/config.json", encoding="utf-8"))
cfg.update({
    "version": "2.1",
    "phase_a": True, "evidence": True, "judge_prompt": "judge_system.md",
    "extra_a": {"base": {"system": "baseline_prompt.md", "user": "baseline"}},
    "sentence_probe": False, "phase_b": False, "second_sample": False, "b2_all": False,
    "decision": "stack", "stack": stack, "max_pos_frac": 0.3,
})
print("фазы: A + A_base | рассуждение выкл. | S выкл. | max_pos_frac", cfg["max_pos_frac"])
if a.dry:
    sys.exit(0)
dst = ROOT / "solution/src/guardian/config.json"
bak = ROOT / "solution/src/guardian/config_v1_2.json"
if not bak.exists():
    shutil.copy(dst, bak)
    print("старый боевой конфиг ->", bak.name)
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print("->", dst)
