#!/usr/bin/env python
"""Additional compact figures for the FV/TV reproducibility paper.

Outputs:
  paper/figures/effect_summary.pdf
  paper/figures/tv_controls_by_task.pdf
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)

from analyze import best_layer_recovery, control_recovery, load_rows
from fvtv.stats import cluster_bootstrap_ci

MODEL_LABELS = {
    "gpt-j-6b": "GPT-J",
    "llama-3.1-8b": "L3.1-8B",
    "gemma-2-9b-it": "Gemma-2",
    "llama-3.1-70b": "L3.1-70B",
}
METHOD_COLORS = {"fv": "#DDAA33", "tv": "#33518A"}
CONTROL_COLOR = "#9A9A9A"
ORDER = [
    ("gpt-j-6b", "fv"), ("gpt-j-6b", "tv"),
    ("llama-3.1-8b", "fv"), ("llama-3.1-8b", "tv"),
    ("gemma-2-9b-it", "fv"), ("gemma-2-9b-it", "tv"),
    ("llama-3.1-70b", "tv"),
]


def load_summary():
    rows = []
    with (ROOT / "results" / "summary.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "model": r["model"],
                "method": r["method"],
                "mean": float(r["mean"]),
                "ci_lo": float(r["ci_lo"]),
                "ci_hi": float(r["ci_hi"]),
                "control": float(r["control_mean"]),
            })
    return {(r["model"], r["method"]): r for r in rows}


def gemma_crosstask_row(k=40):
    """Todd-protocol (cross-task head selection) FV arm on Gemma-2, as one
    extra row for the effect summary: best-layer recovery per task at this k,
    task-clustered CI, and the matched random-vector control."""
    path = ROOT / "results" / f"grid_gemma-2-9b-it_crosstask.json"
    if not path.exists():
        return None
    best, ctrl = defaultdict(lambda: -9.0), {}
    for r in json.loads(path.read_text()):
        if r["k"] != k:
            continue
        if r["method"] == "fv_xtask":
            best[r["task"]] = max(best[r["task"]], r["recovery_ratio"])
        else:
            ctrl[r["task"]] = r["recovery_ratio"]
    if not best:
        return None
    mean, lo, hi = cluster_bootstrap_ci({t: [v] for t, v in best.items()})
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi,
            "control": sum(ctrl.values()) / len(ctrl)}


def effect_summary():
    rows = load_summary()
    extra = gemma_crosstask_row()
    order = list(ORDER)
    if extra is not None:
        rows[("gemma-2-9b-it", "fv_xtask")] = extra
        order.insert(5, ("gemma-2-9b-it", "fv_xtask"))
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    y = list(range(len(order)))[::-1]
    labels = []
    for yi, key in zip(y, order):
        r = rows[key]
        model, method = key
        labels.append(f"{MODEL_LABELS[model]} " +
                      ("FV (cross-task $k$=40)" if method == "fv_xtask" else method.upper()))
        color = METHOD_COLORS["fv" if method.startswith("fv") else "tv"]
        ax.hlines(yi, r["ci_lo"], r["ci_hi"], color=color, lw=1.6, zorder=3)
        ax.plot(r["mean"], yi, "o", ms=3.5, color=color, zorder=4,
                mfc="white" if method == "fv_xtask" else color)
        ax.plot(r["control"], yi, "x", ms=4.0, color=CONTROL_COLOR, zorder=4)
    ax.axvline(0, color="0.86", lw=0.7, zorder=0)
    ax.axvline(1, color="0.90", lw=0.7, zorder=0)
    ax.set_yticks(y, labels, fontsize=6)
    ax.set_xlabel("recovery ratio", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlim(-0.08, 1.08)
    ax.grid(axis="x", color="0.93", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    out = FIG / "effect_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def tv_controls_by_task():
    """Per model and task (seeds averaged), real TV recovery on x against each
    ablated variant on y: label-shuffled, cross-task donor, template swap.
    Points on the diagonal mean the variant is as good as the real vector;
    points on the floor mean it carries nothing. Main-grid cells only."""
    grid = load_rows()
    models = ["gpt-j-6b", "llama-3.1-8b", "gemma-2-9b-it", "llama-3.1-70b"]
    real, series = {}, {"shuffled": {}, "cross-task": {}, "template": {}}
    for m in models:
        for (task, seed), v in best_layer_recovery(grid, m, "tv").items():
            real.setdefault((m, task), []).append(v)
        for (task, seed, _c), v in control_recovery(grid, m, ("tv_control_shuffled_theta",)).items():
            series["shuffled"].setdefault((m, task), []).append(v)
        path = ROOT / "results" / f"grid_{m}_tvextra.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            key = {"tv_control_cross_task": "cross-task",
                   "tv_control_template": "template"}.get(r["method"])
            if key:
                series[key].setdefault((m, r["task"]), []).append(r["recovery_ratio"])

    avg = lambda d: {k: sum(v) / len(v) for k, v in d.items()}
    real = avg(real)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharex=True, sharey=True)
    markers = {"gpt-j-6b": "o", "llama-3.1-8b": "s", "gemma-2-9b-it": "^", "llama-3.1-70b": "D"}
    colors = {"gpt-j-6b": "#33518A", "llama-3.1-8b": "#6C5CE7",
              "gemma-2-9b-it": "#DDAA33", "llama-3.1-70b": "#2E8B57"}
    titles = {"shuffled": "label-shuffled $\\theta$",
              "cross-task": "cross-task donor $\\theta$",
              "template": "template-swap $\\theta$"}

    for ax, name in zip(axes, ("shuffled", "cross-task", "template")):
        pts = avg(series[name])
        for (m, task), yv in pts.items():
            if (m, task) not in real:
                continue
            ax.scatter(real[(m, task)], yv, s=17, marker=markers[m], color=colors[m],
                       edgecolor="white", linewidth=0.25, alpha=0.9, zorder=3,
                       label=MODEL_LABELS[m] if ax is axes[0] else None)
        ax.plot([-0.1, 1.2], [-0.1, 1.2], color="0.75", lw=0.8, ls="--", zorder=1)
        ax.axhline(0, color="0.90", lw=0.6, zorder=1)
        ax.set_axisbelow(True)
        ax.set_title(titles[name], fontsize=7, pad=3)
        ax.set_xlim(-0.06, 1.20)
        ax.set_ylim(-0.30, 1.20)
        ax.tick_params(labelsize=6, length=2)
        ax.grid(color="0.94", lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    axes[0].legend(h2, l2, fontsize=5.6, frameon=False, loc="upper left",
                   ncol=2, handletextpad=0.2, columnspacing=0.6)
    axes[0].set_ylabel("control recovery", fontsize=7)
    fig.supxlabel("real TV recovery", fontsize=7, y=0.03)
    fig.tight_layout(pad=0.3)
    out = FIG / "tv_controls_by_task.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    effect_summary()
    tv_controls_by_task()
