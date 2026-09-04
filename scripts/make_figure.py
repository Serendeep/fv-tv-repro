#!/usr/bin/env python
"""Layer-profile figure (paper/figures/layer_profiles.pdf) from
results/layer_profiles.csv. 2x2 small multiples, one panel per model,
TV solid dark blue, FV dashed amber: distinct in grayscale by lightness
and linestyle.

Usage: .venv/bin/python scripts/make_figure.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TV, FV = "#33518A", "#DDAA33"
ORDER = ["gpt-j-6b", "llama-3.1-8b", "gemma-2-9b-it", "llama-3.1-70b"]
N_LAYERS = {"gpt-j-6b": 28, "llama-3.1-8b": 32, "gemma-2-9b-it": 42, "llama-3.1-70b": 80}
TITLES = {"gpt-j-6b": "GPT-J-6B", "llama-3.1-8b": "Llama-3.1-8B",
          "gemma-2-9b-it": "Gemma-2-9b-it", "llama-3.1-70b": "Llama-3.1-70B"}

data = {}
with open(ROOT / "results" / "layer_profiles.csv") as f:
    for r in csv.DictReader(f):
        data.setdefault((r["model"], r["method"]), []).append(
            (int(r["layer"]) / N_LAYERS[r["model"]], float(r["mean_recovery"])))

fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.0), sharex=True, sharey=True)
for ax, model in zip(axes.flat, ORDER):
    for method, color, ls in (("tv", TV, "-"), ("fv", FV, "--")):
        pts = sorted(data.get((model, method), []))
        if pts:
            ax.plot(*zip(*pts), color=color, ls=ls, lw=1.4, zorder=3,
                    label={"tv": "TV", "fv": "FV"}[method])
    ax.set_title(TITLES[model], fontsize=7, pad=2)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.15, 1.0)
    ax.set_xlim(0, 1)

axes[0, 0].legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.6)
fig.supxlabel("layer / depth", fontsize=7, y=0.02)
fig.supylabel("recovery ratio", fontsize=7, x=0.02)
fig.tight_layout(pad=0.4)
out = ROOT / "paper" / "figures" / "layer_profiles.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
