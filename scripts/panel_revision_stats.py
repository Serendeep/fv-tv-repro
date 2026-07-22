#!/usr/bin/env python
"""Panel-revision analyses from released grid results.

Computes:
- paired method-minus-control clustered CIs;
- gap-filtered sensitivity for recovery ratios;
- per-model task subsets for the appendix.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fvtv import stats

METHODS = ("fv", "tv")
CONTROLS = {
    "fv": ("fv_control_random_vector", "fv_control_random_k_heads"),
    "tv": ("tv_control_shuffled_theta",),
}
MODEL_LABELS = {
    "gpt-j-6b": "GPT-J",
    "llama-3.1-8b": "L3.1-8B",
    "gemma-2-9b-it": "Gemma-2",
    "llama-3.1-70b": "L3.1-70B",
}
MODEL_ORDER = ["gpt-j-6b", "llama-3.1-8b", "gemma-2-9b-it", "llama-3.1-70b"]
TASK_CATS = {
    "antonym": "ling.", "synonym": "ling.", "present-past": "ling.", "singular-plural": "ling.",
    "country-capital": "know.", "country-currency": "know.", "person-occupation": "know.", "park-country": "know.",
    "english-french": "trans.", "english-spanish": "trans.", "english-german": "trans.",
    "capitalize": "alg.", "capitalize-first": "alg.", "capitalize_first_letter": "alg.", "lowercase-first": "alg.", "lowercase_first_letter": "alg.", "next_item": "alg.", "previous-item": "alg.", "prev_item": "alg.",
}
GAP_MIN = 0.2


def is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def load_rows():
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results" / "grid_*.json"))):
        try:
            rows.extend(json.load(open(path)))
        except json.JSONDecodeError:
            continue
    dedup = {}
    for r in rows:
        dedup[(r["model"], r["task"], r["seed"], r["method"], r["layer"])] = r
    return list(dedup.values())


def best_method_by_pair(rows, model, method):
    out = {}
    for r in rows:
        if r["model"] != model or r["method"] != method or not is_num(r.get("recovery_ratio")):
            continue
        pair = (r["task"], r["seed"])
        if pair not in out or r["recovery_ratio"] > out[pair]["recovery_ratio"]:
            out[pair] = r
    return out


def controls_by_pair(rows, model, method):
    vals = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["method"] in CONTROLS[method] and is_num(r.get("recovery_ratio")):
            vals[(r["task"], r["seed"])].append(r["recovery_ratio"])
    return {k: sum(v) / len(v) for k, v in vals.items() if v}


def gaps_by_pair(rows):
    z, icl = {}, {}
    for r in rows:
        pair = (r["model"], r["task"], r["seed"])
        if r["method"] == "zero_shot_floor" and is_num(r.get("accuracy")):
            z[pair] = r["accuracy"]
        elif r["method"] == "icl_ceiling" and is_num(r.get("accuracy")):
            icl[pair] = r["accuracy"]
    return {k: icl[k] - z[k] for k in icl.keys() & z.keys()}


def cluster(vals_by_pair):
    by_task = defaultdict(list)
    for (task, seed), val in vals_by_pair.items():
        if is_num(val):
            by_task[task].append(val)
    return dict(by_task)


def fmt(x):
    if not is_num(x):
        return "--"
    return f"{x:.2f}"


def main():
    rows = load_rows()
    gaps = gaps_by_pair(rows)
    outdir = ROOT / "results"

    paired_rows = []
    sensitivity_rows = []
    for model in MODEL_ORDER:
        for method in METHODS:
            meth = best_method_by_pair(rows, model, method)
            if not meth:
                continue
            ctrl = controls_by_pair(rows, model, method)
            common = sorted(set(meth) & set(ctrl))
            diffs = {p: meth[p]["recovery_ratio"] - ctrl[p] for p in common}
            dm, dlo, dhi = stats.cluster_bootstrap_ci(cluster(diffs))
            paired_rows.append({
                "model": model, "method": method, "n_pairs": len(common), "n_tasks": len({p[0] for p in common}),
                "diff_mean": dm, "diff_lo": dlo, "diff_hi": dhi,
            })
            # gap sensitivity: keep cells whose ICL-zero headroom is >= 0.2.
            kept = [p for p in meth if gaps.get((model, p[0], p[1]), 0.0) >= GAP_MIN]
            all_vals = {p: meth[p]["recovery_ratio"] for p in meth}
            kept_vals = {p: meth[p]["recovery_ratio"] for p in kept}
            am, alo, ahi = stats.cluster_bootstrap_ci(cluster(all_vals))
            km, klo, khi = stats.cluster_bootstrap_ci(cluster(kept_vals)) if kept_vals else (float("nan"), float("nan"), float("nan"))
            sensitivity_rows.append({
                "model": model, "method": method,
                "n_all": len(all_vals), "n_gap_ge_0p2": len(kept_vals),
                "all_mean": am, "all_lo": alo, "all_hi": ahi,
                "gap_mean": km, "gap_lo": klo, "gap_hi": khi,
            })

    with (outdir / "paired_differences.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        w.writeheader(); w.writerows(paired_rows)
    with (outdir / "gap_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0]))
        w.writeheader(); w.writerows(sensitivity_rows)

    # Appendix task-subset table.
    task_rows = []
    for model in MODEL_ORDER:
        for method in METHODS:
            pairs = set(best_method_by_pair(rows, model, method))
            if not pairs:
                continue
            tasks = sorted({p[0] for p in pairs})
            cats = defaultdict(int)
            for t in tasks:
                cats[TASK_CATS.get(t, "other")] += 1
            task_rows.append({
                "model": model,
                "method": method,
                "n_tasks": len(tasks),
                "n_pairs": len(pairs),
                "categories": ", ".join(f"{k} {v}" for k, v in sorted(cats.items())),
                "tasks": ", ".join(tasks),
            })
    with (outdir / "task_subsets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0]))
        w.writeheader(); w.writerows(task_rows)

    print("== paired method-control differences ==")
    for r in paired_rows:
        print(f"{MODEL_LABELS[r['model']]:9s} {r['method'].upper():2s} n={r['n_pairs']:2d}/{r['n_tasks']:2d} tasks diff={r['diff_mean']:.2f} [{r['diff_lo']:.2f},{r['diff_hi']:.2f}]")
    print("\n== gap sensitivity (gap >= 0.2) ==")
    for r in sensitivity_rows:
        print(f"{MODEL_LABELS[r['model']]:9s} {r['method'].upper():2s} all={r['all_mean']:.2f} [{r['all_lo']:.2f},{r['all_hi']:.2f}] n={r['n_all']:2d}; gap={fmt(r['gap_mean'])} [{fmt(r['gap_lo'])},{fmt(r['gap_hi'])}] n={r['n_gap_ge_0p2']:2d}")
    print("\n== task subsets ==")
    for r in task_rows:
        print(f"{MODEL_LABELS[r['model']]:9s} {r['method'].upper():2s}: {r['n_tasks']} tasks/{r['n_pairs']} pairs ({r['categories']}): {r['tasks']}")
    print("\nwrote results/paired_differences.csv, results/gap_sensitivity.csv, results/task_subsets.csv")

if __name__ == "__main__":
    main()
