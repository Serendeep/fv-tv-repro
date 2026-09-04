#!/usr/bin/env python
"""Revision-pass statistics: clustered (by-task) bootstrap CIs, CV co-primary
estimates, per-model protocol/coverage table, per-task shuffled-vs-real TV
breakdown, and true-depth layer peaks.

Usage: .venv/bin/python scripts/revision_stats.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fvtv import stats

N_LAYERS = {"gpt-j-6b": 28, "llama-3.1-8b": 32, "gemma-2-9b-it": 42, "llama-3.1-70b": 80}

# analyze.load_rows applies the filename filter that keeps reduced-protocol
# probes and the extra-control runs out of the main grid; loading them here
# would print per-task numbers that disagree with the paper.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_rows

rows = load_rows()
models = sorted({r["model"] for r in rows})

sweep = {}
for r in rows:
    if r["method"] in ("fv", "tv") and r.get("recovery_ratio") is not None:
        sweep.setdefault((r["model"], r["task"], r["seed"], r["method"]), {})[r["layer"]] = r["recovery_ratio"]

print("== protocol/coverage per model (from data) ==")
for m in models:
    mrows = [r for r in rows if r["model"] == m]
    tasks_ = sorted({r["task"] for r in mrows})
    seeds = sorted({r["seed"] for r in mrows})
    nevals = sorted({r["n_eval"] for r in mrows if r.get("n_eval")})
    layers = sorted({r["layer"] for r in mrows if r["method"] == "tv" and r["layer"] is not None})
    stride = layers[1] - layers[0] if len(layers) > 1 else "?"
    pairs = {(r["task"], r["seed"]) for r in mrows}
    print(f"{m}: tasks={len(tasks_)} seeds={seeds} n_eval={nevals} stride={stride} pairs={len(pairs)}")
    print(f"   tasks: {', '.join(tasks_)}")

print("\n== clustered (by-task) primary estimates ==")
for m in models:
    for method in ("fv", "tv"):
        by_task = defaultdict(list)
        for (m_, t, s, meth), sw in sweep.items():
            if m_ == m and meth == method:
                by_task[t].append(max(sw.values()))
        if not by_task:
            continue
        mean, lo, hi = stats.cluster_bootstrap_ci(dict(by_task))
        n_pairs = sum(len(v) for v in by_task.values())
        # matched control, clustered the same way
        cname = {"fv": ("fv_control_random_vector", "fv_control_random_k_heads"),
                 "tv": ("tv_control_shuffled_theta",)}[method]
        cb = defaultdict(list)
        for r in rows:
            if r["model"] == m and r["method"] in cname and r.get("recovery_ratio") is not None:
                cb[r["task"]].append(r["recovery_ratio"])
        cmean, clo, chi = stats.cluster_bootstrap_ci(dict(cb))
        print(f"{m:16s} {method}: {mean:.3f} [{lo:.3f},{hi:.3f}] n={n_pairs} ({len(by_task)} tasks) | ctrl {cmean:.3f} [{clo:.3f},{chi:.3f}]")

print("\n== CV (layer picked on seed 0, evaluated on later seeds), clustered ==")
for m in models:
    for method in ("fv", "tv"):
        by_task = defaultdict(list)
        for t in {k[1] for k in sweep if k[0] == m and k[3] == method}:
            s0 = sweep.get((m, t, 0, method))
            if not s0:
                continue
            best = max(s0, key=s0.get)
            for seed in (1, 2):
                sw = sweep.get((m, t, seed, method))
                if sw and best in sw:
                    by_task[t].append(sw[best])
        if sum(len(v) for v in by_task.values()) >= 2:
            mean, lo, hi = stats.cluster_bootstrap_ci(dict(by_task))
            print(f"{m:16s} {method}: cv={mean:.3f} [{lo:.3f},{hi:.3f}] n={sum(len(v) for v in by_task.values())}")

print("\n== per-task shuffled-theta vs real TV (mean over seeds) ==")
for m in models:
    per_task = defaultdict(dict)
    for (m_, t, s, meth), sw in sweep.items():
        if m_ == m and meth == "tv":
            per_task[t].setdefault("tv", []).append(max(sw.values()))
    for r in rows:
        if r["model"] == m and r["method"] == "tv_control_shuffled_theta" and r.get("recovery_ratio") is not None:
            per_task[r["task"]].setdefault("shuf", []).append(r["recovery_ratio"])
    for t, d in sorted(per_task.items()):
        if "tv" in d and "shuf" in d:
            tv_m = sum(d["tv"]) / len(d["tv"])
            sh_m = sum(d["shuf"]) / len(d["shuf"])
            print(f"{m:16s} {t:24s} tv={tv_m:.2f} shuf={sh_m:.2f} ratio={sh_m/tv_m if tv_m else float('nan'):.2f}")

print("\n== TV peak layers at true depth ==")
for m in models:
    by_layer = defaultdict(list)
    for (m_, t, s, meth), sw in sweep.items():
        if m_ == m and meth == "tv":
            for L, rr in sw.items():
                by_layer[L].append(rr)
    if by_layer:
        peak = max(by_layer, key=lambda L: sum(by_layer[L]) / len(by_layer[L]))
        print(f"{m:16s} peak L{peak}/{N_LAYERS[m]} = {peak/N_LAYERS[m]:.2f} of true depth")
