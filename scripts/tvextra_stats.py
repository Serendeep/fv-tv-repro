#!/usr/bin/env python
"""Summarize results/grid_*_tvextra.json against the main-grid TV and
shuffled-theta cells: task-clustered mean [95% CI] per condition, per model,
plus a per-task table for the additive variant.

Usage: uv run python scripts/tvextra_stats.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import best_layer_recovery, control_recovery, load_rows
from fvtv.stats import cluster_bootstrap_ci


def clustered(by_pair):
    c = defaultdict(list)
    for (task, *_), v in by_pair.items():
        c[task].append(v)
    return cluster_bootstrap_ci(c)


def main():
    grid = load_rows()
    for path in sorted(glob.glob(str(ROOT / "results" / "grid_*_tvextra.json"))):
        extra = json.load(open(path))
        model = extra[0]["model"]
        tv = best_layer_recovery(grid, model, "tv")
        shuf = {(t, s): v for (t, s, _m), v in control_recovery(grid, model, ("tv_control_shuffled_theta",)).items()}
        cross = {(r["task"], r["seed"]): r["recovery_ratio"] for r in extra if r["method"] == "tv_control_cross_task"}
        tmpl = {(r["task"], r["seed"]): r["recovery_ratio"] for r in extra if r["method"] == "tv_control_template"}
        add = defaultdict(lambda: -9.0)
        for r in extra:
            if r["method"] == "tv_additive":
                add[(r["task"], r["seed"])] = max(add[(r["task"], r["seed"])], r["recovery_ratio"])
        keys = sorted(cross)
        print(f"\n== {model}: {len(keys)} cells, {len({k[0] for k in keys})} tasks")
        for name, d in [("real theta (replace)", tv), ("shuffled-label theta", shuf), ("cross-task theta", cross),
                        ("arrow-template theta", tmpl), ("additive theta (best layer)", add)]:
            sub = {k: d[k] for k in keys if k in d}
            m, lo, hi = clustered(sub)
            print(f"  {name:28s} {m:5.2f} [{lo:5.2f}, {hi:5.2f}]  n={len(sub)}")
        print("  per task (real / shuf / cross / template / additive):")
        by_task = defaultdict(list)
        for k in keys:
            by_task[k[0]].append(k)
        for task, ks in sorted(by_task.items()):
            f = lambda d: sum(d[k] for k in ks) / len(ks)
            print(f"    {task:24s} {f(tv):5.2f} {f(shuf):5.2f} {f(cross):5.2f} {f(tmpl):5.2f} {f(add):5.2f}")


if __name__ == "__main__":
    main()
