#!/usr/bin/env python
"""Plot the completed sentiment extension from verified saved predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from extension_paper_tables import analyze, DATA

ROOT = Path(__file__).resolve().parents[1]

def main():
    summary = analyze()
    assert summary == json.loads((DATA / 'summary.json').read_text()), 'Stale summary'
    models = ['gpt-j-6b', 'llama-3.1-8b', 'gemma-2-9b', 'gemma-2-9b-it']
    labels = ['GPT-J', 'Llama-8B', 'Gemma base', 'Gemma IT']
    rows = [summary['sentiment'][m]['arbitrary'] for m in models]
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.35))
    fig.subplots_adjust(left=.26, right=.97, top=.87, bottom=.12, hspace=.95)
    y = np.arange(4)
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.set_ylim(3.55, -.55)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='.9', linewidth=.5)
        ax.set_axisbelow(True)
    top, bottom = axes
    top.axvline(50, color='.45', ls='--', lw=.8)
    for key, label, marker, color, offset in [
            ('icl', 'ICL', 's', '#0072B2', -.18),
            ('real', 'TV', 'o', '#D55E00', 0),
            ('control', 'Shuffled TV', 'x', '#333333', .18)]:
        top.scatter([r[key] for r in rows], y + offset, s=19,
                    label=label, marker=marker, color=color, zorder=3)
    top.set_xlim(0, 100)
    top.set_xticks([0, 25, 50, 75, 100])
    top.set_xlabel('Accuracy (%)', labelpad=2)
    top.legend(loc='lower center', bbox_to_anchor=(.48, 1.03),
               ncol=3, frameon=False, handletextpad=.3, columnspacing=.8)
    delta = np.array([r['difference'] for r in rows])
    bounds = np.array([r['ci95'] for r in rows])
    bottom.axvline(0, color='.45', ls='--', lw=.8)
    bottom.errorbar(delta, y, xerr=np.vstack([delta-bounds[:, 0], bounds[:, 1]-delta]),
                    fmt='o', markersize=3.5, color='#D55E00', capsize=2, lw=1)
    bottom.set_xlim(-3, 3)
    bottom.set_xticks([-3, 0, 3])
    bottom.set_xlabel('TV − shuffled TV (percentage points)', labelpad=2)
    destination = ROOT / 'paper/figures/sentiment_transfer.pdf'
    fig.savefig(destination, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(f'Wrote {destination}; all plotted data verified.')

if __name__ == '__main__':
    main()
