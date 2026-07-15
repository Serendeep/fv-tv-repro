#!/usr/bin/env python
"""Aggregate results/grid_*.json into summary.csv + layer_profiles.csv, print
a per-model verdict table.

Usage: .venv/bin/python scripts/analyze.py
"""
import csv
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fvtv import stats

METHODS = ("fv", "tv")
CONTROLS = {
    # ponytail: fv has two matched controls (does the vector content matter /
    # does head selection matter). Pool both into one control sample so
    # summary.csv can keep a single control_mean/cohens_d per method row
    # instead of forking the schema. Individual control means are still
    # visible in the stdout table.
    "fv": ("fv_control_random_vector", "fv_control_random_k_heads"),
    "tv": ("tv_control_shuffled_theta",),
}


def load_rows():
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results" / "grid_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"WARNING: skipping {path}, mid-write or corrupt ({e})", file=sys.stderr)
            continue
        rows.extend(data)

    # dedupe on (model, task, seed, method, layer), keep last occurrence
    dedup = {}
    for r in rows:
        key = (r["model"], r["task"], r["seed"], r["method"], r["layer"])
        dedup[key] = r
    return list(dedup.values())


def is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def best_layer_recovery(rows, model, method):
    """max recovery_ratio over layers, per (task, seed), for one model+method."""
    by_pair = {}
    for r in rows:
        if r["model"] != model or r["method"] != method:
            continue
        rr = r.get("recovery_ratio")
        if not is_num(rr):
            continue
        pair = (r["task"], r["seed"])
        if pair not in by_pair or rr > by_pair[pair]:
            by_pair[pair] = rr
    return by_pair


def control_recovery(rows, model, control_methods):
    """recovery_ratio per (task, seed) for one or more control methods
    (already evaluated at a single best layer, no max needed). Multiple
    control methods are pooled into one sample."""
    by_pair = {}
    for r in rows:
        if r["model"] != model or r["method"] not in control_methods:
            continue
        rr = r.get("recovery_ratio")
        if not is_num(rr):
            continue
        by_pair[(r["task"], r["seed"], r["method"])] = rr
    return by_pair


def floor_ceiling_means(rows, model):
    icl = [r["accuracy"] for r in rows if r["model"] == model and r["method"] == "icl_ceiling" and is_num(r.get("accuracy"))]
    zero = [r["accuracy"] for r in rows if r["model"] == model and r["method"] == "zero_shot_floor" and is_num(r.get("accuracy"))]
    icl_mean = sum(icl) / len(icl) if icl else float("nan")
    zero_mean = sum(zero) / len(zero) if zero else float("nan")
    return icl_mean, zero_mean


def layer_profiles(rows, models):
    out = []
    for model in models:
        max_layer = max(
            (r["layer"] for r in rows if r["model"] == model and r["method"] in METHODS and r["layer"] is not None),
            default=None,
        )
        if not max_layer:
            continue
        for method in METHODS:
            by_layer = {}
            for r in rows:
                if r["model"] != model or r["method"] != method or r["layer"] is None:
                    continue
                rr = r.get("recovery_ratio")
                if not is_num(rr):
                    continue
                by_layer.setdefault(r["layer"], []).append(rr)
            for layer, values in sorted(by_layer.items()):
                out.append({
                    "model": model,
                    "method": method,
                    "layer": layer,
                    "layer_frac": round(layer / max_layer, 4),
                    "mean_recovery": sum(values) / len(values),
                    "n_pairs": len(values),
                })
    return out


def verdict(cohens_d, mean, control_mean):
    if math.isnan(cohens_d) or math.isnan(mean) or math.isnan(control_mean):
        return "INSUFFICIENT DATA"
    if mean > control_mean and cohens_d >= 0.8:
        return "REPLICATES"
    if mean > control_mean and cohens_d >= 0.2:
        return "WEAK EFFECT"
    return "NO EFFECT"


def main():
    rows = load_rows()
    models = sorted({r["model"] for r in rows})

    summary_rows = []
    print_lines = []

    for model in models:
        icl_mean, zero_mean = floor_ceiling_means(rows, model)
        print_lines.append(f"\n=== {model} ===  icl_ceiling={icl_mean:.3f}  zero_shot_floor={zero_mean:.3f}")
        header = f"{'method':4} {'mean':>7} {'ci_lo':>7} {'ci_hi':>7} {'n':>4}  {'control':>7} {'net':>7} {'d':>6}  verdict"
        print_lines.append(header)

        for method in METHODS:
            method_by_pair = best_layer_recovery(rows, model, method)
            if not method_by_pair:
                continue
            method_values = list(method_by_pair.values())
            mean, lo, hi = stats.bootstrap_ci(method_values)

            control_by_key = control_recovery(rows, model, CONTROLS[method])
            control_values = list(control_by_key.values())
            control_mean, _, _ = stats.bootstrap_ci(control_values) if control_values else (float("nan"), float("nan"), float("nan"))
            d = stats.cohens_d(method_values, control_values)
            net = mean - control_mean if is_num(mean) and is_num(control_mean) else float("nan")

            summary_rows.append({
                "model": model, "method": method, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "n_pairs": len(method_values), "control_mean": control_mean, "cohens_d": d,
            })

            v = verdict(d, mean, control_mean)
            print_lines.append(
                f"{method:4} {mean:7.3f} {lo:7.3f} {hi:7.3f} {len(method_values):4d}  "
                f"{control_mean:7.3f} {net:7.3f} {d:6.2f}  {v}"
            )

    profiles = layer_profiles(rows, models)

    summary_path = ROOT / "results" / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "mean", "ci_lo", "ci_hi", "n_pairs", "control_mean", "cohens_d"])
        w.writeheader()
        w.writerows(summary_rows)

    profiles_path = ROOT / "results" / "layer_profiles.csv"
    with open(profiles_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "layer", "layer_frac", "mean_recovery", "n_pairs"])
        w.writeheader()
        w.writerows(profiles)

    print(f"wrote {summary_path} ({len(summary_rows)} rows)")
    print(f"wrote {profiles_path} ({len(profiles)} rows)")
    print("\n".join(print_lines))


if __name__ == "__main__":
    main()
