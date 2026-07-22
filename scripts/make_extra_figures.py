#!/usr/bin/env python
"""Additional compact figures for the FV/TV reproducibility paper.

Outputs:
  paper/figures/effect_summary.pdf
  paper/figures/tv_shuffled_by_task.pdf
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)

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


def effect_summary():
    rows = load_summary()
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    y = list(range(len(ORDER)))[::-1]
    labels = []
    for yi, key in zip(y, ORDER):
        r = rows[key]
        model, method = key
        labels.append(f"{MODEL_LABELS[model]} {method.upper()}")
        ax.hlines(yi, r["ci_lo"], r["ci_hi"], color=METHOD_COLORS[method], lw=1.6)
        ax.plot(r["mean"], yi, "o", ms=3.5, color=METHOD_COLORS[method], label=method.upper())
        ax.plot(r["control"], yi, "x", ms=4.0, color=CONTROL_COLOR)
    ax.axvline(0, color="0.86", lw=0.7, zorder=0)
    ax.axvline(1, color="0.90", lw=0.7, zorder=0)
    ax.set_yticks(y, labels, fontsize=6)
    ax.set_xlabel("recovery ratio", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlim(-0.08, 1.08)
    ax.grid(axis="x", color="0.93", lw=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    out = FIG / "effect_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def iter_result_rows():
    for path in sorted((ROOT / "results").glob("grid_*.json")):
        rows = json.loads(path.read_text())
        if isinstance(rows, list):
            yield from rows


def tv_shuffled_by_task():
    # Per model/task, average across seeds: best real TV recovery vs shuffled-theta recovery.
    buckets = defaultdict(list)
    rows = list(iter_result_rows())
    pairs = sorted({(r["model"], r["task"], r["seed"]) for r in rows if r.get("method") in {"tv", "tv_control_shuffled_theta"}})
    for model, task, seed in pairs:
        cell = [r for r in rows if r.get("model") == model and r.get("task") == task and r.get("seed") == seed]
        tvs = [r for r in cell if r.get("method") == "tv"]
        shuf = [r for r in cell if r.get("method") == "tv_control_shuffled_theta"]
        if not tvs or not shuf:
            continue
        best = max(tvs, key=lambda r: r.get("recovery_ratio", -999))
        buckets[(model, task)].append((best.get("recovery_ratio", 0.0), shuf[0].get("recovery_ratio", 0.0)))

    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    markers = {"gpt-j-6b": "o", "llama-3.1-8b": "s", "gemma-2-9b-it": "^", "llama-3.1-70b": "D"}
    colors = {"gpt-j-6b": "#33518A", "llama-3.1-8b": "#6C5CE7", "gemma-2-9b-it": "#DDAA33", "llama-3.1-70b": "#2E8B57"}
    seen = set()
    for (model, task), vals in buckets.items():
        tv = sum(v[0] for v in vals) / len(vals)
        sh = sum(v[1] for v in vals) / len(vals)
        label = MODEL_LABELS[model] if model not in seen else None
        seen.add(model)
        ax.scatter(tv, sh, s=18, marker=markers[model], color=colors[model], edgecolor="white", linewidth=0.25, label=label, alpha=0.9)
    ax.plot([-0.1, 1.2], [-0.1, 1.2], color="0.75", lw=0.8, ls="--")
    ax.axhline(0, color="0.90", lw=0.6)
    ax.axvline(0, color="0.90", lw=0.6)
    ax.set_xlim(-0.06, 1.16)
    ax.set_ylim(-0.06, 1.16)
    ax.set_xlabel("real TV recovery", fontsize=7)
    ax.set_ylabel("shuffled-$\\theta$ recovery", fontsize=7)
    ax.tick_params(labelsize=6, length=2)
    ax.grid(color="0.94", lw=0.5)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    out = FIG / "tv_shuffled_by_task.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    effect_summary()
    tv_shuffled_by_task()
