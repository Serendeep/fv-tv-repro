#!/usr/bin/env python
"""Poster-scale figures (poster/fig/*.svg) from the committed results.
Lato text is stored as outlines, so the SVGs render the same anywhere.

Usage: python poster/make_poster_figures.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LATO = Path("/usr/local/texlive/2025/texmf-dist/fonts/truetype/typoland/lato")
for f in LATO.glob("Lato-*.ttf"):
    font_manager.fontManager.addfont(str(f))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_extra_figures import gemma_crosstask_row, load_summary

OUT = ROOT / "poster" / "fig"
OUT.mkdir(exist_ok=True)
TV, FV, CTRL, INK, MUTED, RULE = "#33518A", "#DDAA33", "#8A8F98", "#16202A", "#56606B", "#D5D9DE"
FV_INK = "#8C6410"
plt.rcParams.update({
    "svg.fonttype": "path", "font.family": "Lato", "font.size": 24,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1.5,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 10, "ytick.major.pad": 12,
})
HELDOUT = json.loads((ROOT / "results/heldout/summary.json").read_text())
EXT = json.loads((ROOT / "results/extensions/summary.json").read_text())


def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {OUT / name}.svg")


def pp(x):
    return f"{x:+.1f}".replace("-", "−")


def effect():
    """Test 1: exploratory best-layer recovery vs matched controls."""
    rows = load_summary()
    rows[("gemma-2-9b-it", "fv_xtask")] = gemma_crosstask_row()
    order = [("GPT-J-6B", [("gpt-j-6b", "fv"), ("gpt-j-6b", "tv")]),
             ("Llama-3.1-8B", [("llama-3.1-8b", "fv"), ("llama-3.1-8b", "tv")]),
             ("Gemma-2-9b-it", [("gemma-2-9b-it", "fv"), ("gemma-2-9b-it", "fv_xtask"),
                                ("gemma-2-9b-it", "tv")]),
             ("Llama-3.1-70B", [("llama-3.1-70b", "tv")])]
    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    y, ticks, labels = 0, [], []
    for model, keys in order:
        ax.text(-0.02, y + 0.72, model, fontsize=24, fontweight="bold",
                transform=ax.get_yaxis_transform(), ha="right")
        for key in keys:
            r, method = rows[key], key[1]
            c = FV if method.startswith("fv") else TV
            ax.hlines(y, r["ci_lo"], r["ci_hi"], color=c, lw=5, capstyle="round", zorder=3)
            ax.plot(r["control"], y, "x", ms=15, mew=3.5, color=CTRL, zorder=4)
            ax.plot(r["mean"], y, "o", ms=19, color=c, mec="white" if method != "fv_xtask" else c,
                    mfc="white" if method == "fv_xtask" else c, mew=3 if method == "fv_xtask" else 2, zorder=5)
            ax.text(max(r["ci_hi"], r["mean"]) + 0.04, y, f"{r['mean']:.2f}", va="center",
                    fontsize=22, fontweight="demibold")
            ticks.append(y)
            labels.append({"fv": "FV", "tv": "TV", "fv_xtask": "FV, shared heads"}[method])
            y -= 1
        y -= 0.95
    ax.set_yticks(ticks, labels, fontsize=21, color=MUTED)
    ax.set_xlim(-0.06, 1.2)
    ax.set_ylim(y + 0.55, 1.15)
    ax.axvline(0, color=RULE, lw=1.5, zorder=0)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.grid(axis="x", color="#ECEEF1", lw=1.5)
    ax.set_xlabel("share of the in-context accuracy gap recovered", fontsize=22, labelpad=10)
    clean(ax)
    ax.plot([], [], "o", ms=15, color=TV, label="task vector")
    ax.plot([], [], "o", ms=15, color=FV, label="function vector")
    ax.plot([], [], "x", ms=13, mew=3.5, color=CTRL, label="control")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False,
              fontsize=21, handletextpad=0.25, columnspacing=1.0)
    save(fig, "effect")


def heldout():
    """Test 2: method minus control (pp), layers chosen on dev data."""
    agg = {(a["model"], a["method"]): a for a in HELDOUT["aggregates"]}
    g = EXT["gemma_tv"]
    rows = [("fv", "GPT-J-6B", agg[("gpt-j-6b", "fv")]["mean_difference"] * 100,
             [v * 100 for v in agg[("gpt-j-6b", "fv")]["ci95"]]),
            ("fv", "Llama-3.1-8B", agg[("llama-3.1-8b", "fv")]["mean_difference"] * 100,
             [v * 100 for v in agg[("llama-3.1-8b", "fv")]["ci95"]]),
            ("tv", "GPT-J-6B", agg[("gpt-j-6b", "tv")]["mean_difference"] * 100,
             [v * 100 for v in agg[("gpt-j-6b", "tv")]["ci95"]]),
            ("tv", "Llama-3.1-8B", agg[("llama-3.1-8b", "tv")]["mean_difference"] * 100,
             [v * 100 for v in agg[("llama-3.1-8b", "tv")]["ci95"]]),
            ("tv", "Gemma-2-9b base", g["gemma-2-9b"]["aggregate"]["difference"],
             g["gemma-2-9b"]["aggregate"]["ci95"]),
            ("tv", "Gemma-2-9b-it", g["gemma-2-9b-it"]["aggregate"]["difference"],
             g["gemma-2-9b-it"]["aggregate"]["ci95"])]
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    ys, y, prev = [], 0, None
    for method, name, d, (lo, hi) in rows:
        if method != prev:
            y -= 0.9 if prev else 0
            ax.text(-24, y + 0.78, "FV minus random controls" if method == "fv" else
                    "TV minus shuffled-label controls", ha="left", fontsize=22, fontweight="bold",
                    color=FV_INK if method == "fv" else TV, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=3))
            prev = method
        c = FV if method == "fv" else TV
        ax.hlines(y, lo, hi, color=c, lw=5, capstyle="round", zorder=3)
        ax.plot(d, y, "o", ms=19, color=c, mec="white", mew=2, zorder=4)
        ax.text(hi + 2.5, y, pp(d), va="center", fontsize=22, fontweight="demibold")
        ys.append((y, name))
        y -= 1
    ax.set_yticks([a for a, _ in ys], [b for _, b in ys], fontsize=21)
    ax.axvline(0, color=INK, lw=2, zorder=2)
    ax.set_xlim(-25, 70)
    ax.set_ylim(y + 0.5, 1.35)
    ax.set_xticks([-20, 0, 20, 40, 60], ["−20", "0", "+20", "+40", "+60"])
    ax.grid(axis="x", color="#ECEEF1", lw=1.5)
    ax.set_xlabel("test accuracy above control (percentage points)", fontsize=22, labelpad=10)
    clean(ax)
    ax.axhspan(-6.4, -4.4, color="#F1F3F5", zorder=0, lw=0)
    ax.text(-24, -4.5, "run on Kaggle", ha="left", va="top", fontsize=18, color=MUTED)
    save(fig, "heldout")


def llama_layers():
    """Why controls pick their own layer: Llama-3.1-8B TV, held-out test accuracy."""
    cells = [c for c in HELDOUT["cells"] if (c["model"], c["method"]) == ("llama-3.1-8b", "tv")]

    def task_mean(key):
        by = defaultdict(list)
        for c in cells:
            by[c["task"]].append(c[key])
        return 100 * mean(mean(v) for v in by.values())

    vals = [("zero-shot", task_mean("zero"), "#D5D9DE"),
            ("shuffled θ at the TV's layer", task_mean("conditional_control_mean"), CTRL),
            ("real task vector", task_mean("accuracy"), TV),
            ("shuffled θ at its own layer", task_mean("control_mean"), CTRL)]
    # paper/generated/heldout_accuracy_table.tex: 4.1, 34.7, 50.0, 54.2
    assert [round(v, 1) for _, v, _ in vals] == [4.1, 34.7, 50.0, 54.2], vals
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    for i, (lab, v, c) in enumerate(vals[::-1]):
        ax.barh(i, v, height=0.66, color=c, zorder=3)
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=22,
                fontweight="bold" if c == TV else "normal")
    ax.set_yticks(range(4), [l for l, _, _ in vals[::-1]], fontsize=21)
    ax.set_xlim(0, 68)
    ax.set_xticks([])
    ax.axvline(0, color=INK, lw=2, zorder=4)
    clean(ax)
    ax.spines["bottom"].set_visible(False)
    save(fig, "llama_layers")


def sentiment():
    """Test 3: SST-2 sentiment with arbitrary labels (A/B and B/A pooled)."""
    s = EXT["sentiment"]
    models = [("gpt-j-6b", "GPT-J-6B"), ("llama-3.1-8b", "Llama-3.1-8B"),
              ("gemma-2-9b", "Gemma-2-9b base"), ("gemma-2-9b-it", "Gemma-2-9b-it")]
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.axvline(50, color=INK, lw=2, ls=(0, (4, 3)), zorder=1)
    ax.text(50.8, 3.75, "constant-label baseline", ha="left", fontsize=18, color=MUTED)
    for i, (m, name) in enumerate(models):
        r = s[m]["arbitrary"]
        y = 3 - i
        ax.hlines(y, r["real"], r["icl"], color=RULE, lw=4, zorder=2)
        ax.plot(r["icl"], y, "s", ms=18, color=INK, zorder=4)
        ax.plot(r["control"], y, "x", ms=15, mew=3.5, color=CTRL, zorder=6)
        ax.plot(r["real"], y, "o", ms=19, color=TV, mec="white", mew=2, zorder=5)
        ax.text(r["icl"] + 1.6, y, f"{r['icl']:.1f}", va="center", fontsize=21, fontweight="demibold")
        ax.text(min(r["real"], r["control"]) - 3.4, y, f"{r['real']:.1f}", va="center", ha="right", fontsize=21,
                fontweight="demibold", color=TV)
    ax.set_yticks(range(4)[::-1], [n for _, n in models], fontsize=21)
    ax.set_xlim(36, 102)
    ax.set_ylim(-0.6, 4.1)
    ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
    ax.grid(axis="x", color="#ECEEF1", lw=1.5)
    ax.set_xlabel("test accuracy (%)", fontsize=22, labelpad=10)
    clean(ax)
    ax.plot([], [], "s", ms=15, color=INK, label="10-shot ICL")
    ax.plot([], [], "o", ms=15, color=TV, label="task vector")
    ax.plot([], [], "x", ms=13, mew=3.5, color=CTRL, label="shuffled θ")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=3, frameon=False,
              fontsize=21, handletextpad=0.25, columnspacing=1.0)
    save(fig, "sentiment")


def controls():
    """What survives in a task vector: exploratory controls at the real TV's layer."""
    # paper/generated/tv_control_table.tex
    variants = ["real θ", "template swap", "label-shuffled", "sibling-task θ"]
    vals = {"GPT-J-6B": [0.56, 0.41, 0.28, -0.01],
            "Llama-3.1-8B": [0.83, 0.69, 0.42, 0.06],
            "Llama-3.1-70B": [0.80, 0.55, 0.48, 0.03]}
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 5.6), sharey=True)
    ys = list(range(len(variants)))[::-1]
    for ax, (model, v) in zip(axes, vals.items()):
        for yi, x, i in zip(ys, v, range(len(v))):
            ax.barh(yi, max(x, 0.006), height=0.66, color=TV if i == 0 else "#9DAECB", zorder=3)
            ax.text(max(x, 0) + 0.05, yi, f"{x:.2f}".replace("-", "−"), va="center",
                    fontsize=20, fontweight="bold" if i == 0 else "normal")
        ax.set_title(model, fontsize=21, fontweight="bold", loc="left", pad=8)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([])
        ax.axvline(0, color=INK, lw=2, zorder=4)
        clean(ax)
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(ys, variants, fontsize=21)
    fig.tight_layout(pad=0.3, w_pad=0.4)
    save(fig, "controls")


if __name__ == "__main__":
    effect()
    heldout()
    llama_layers()
    sentiment()
    controls()
