#!/usr/bin/env python
"""Cross-validated layer selection + FV/TV cross-method correlation.

CV estimator: best layer chosen on seed 0, recovery read at that layer on the
remaining seeds. Measures sensitivity to selecting layers on the evaluation split.

Usage: .venv/bin/python scripts/extra_stats.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fvtv import stats

from analyze import load_rows

rows = load_rows()

sweep = {}  # (model, task, seed, method) -> {layer: rr}
for r in rows:
    if r["method"] in ("fv", "tv") and r.get("recovery_ratio") is not None:
        sweep.setdefault((r["model"], r["task"], r["seed"], r["method"]), {})[r["layer"]] = r["recovery_ratio"]

print("== cross-validated (layer picked on seed 0, evaluated on other seeds) ==")
for model in sorted({k[0] for k in sweep}):
    for method in ("fv", "tv"):
        cv_vals, naive_vals = {}, {}
        tasks_ = {k[1] for k in sweep if k[0] == model and k[3] == method}
        for t in sorted(tasks_):
            s0 = sweep.get((model, t, 0, method))
            if not s0:
                continue
            best_layer = max(s0, key=s0.get)
            for seed in (1, 2):
                sw = sweep.get((model, t, seed, method))
                if sw and best_layer in sw:
                    cv_vals.setdefault(t, []).append(sw[best_layer])
                    naive_vals.setdefault(t, []).append(max(sw.values()))
        if len(cv_vals) >= 2:
            m, lo, hi = stats.cluster_bootstrap_ci(cv_vals)
            nm, _, _ = stats.cluster_bootstrap_ci(naive_vals)
            print(f"{model:16s} {method}: cv={m:.3f} [{lo:.3f},{hi:.3f}] naive={nm:.3f} inflation={nm-m:+.3f} n={sum(map(len, cv_vals.values()))}")

print("\n== C3.1: per-task FV vs TV recovery correlation (mean over seeds) ==")
for model in sorted({k[0] for k in sweep}):
    per_task = {}
    for (m_, t, s, meth), sw in sweep.items():
        if m_ != model:
            continue
        per_task.setdefault(t, {}).setdefault(meth, []).append(max(sw.values()))
    xs, ys = [], []
    for t, d in per_task.items():
        if "fv" in d and "tv" in d:
            xs.append(np.mean(d["fv"]))
            ys.append(np.mean(d["tv"]))
    if len(xs) >= 4:
        r = np.corrcoef(xs, ys)[0, 1]
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        print(f"{model:16s} n_tasks={len(xs)} pearson={r:.2f} spearman={rho:.2f} (p={p:.3f}) fv_var={np.var(xs):.4f}")
