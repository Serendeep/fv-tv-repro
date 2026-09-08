#!/usr/bin/env python
"""Generate paper tables from verified held-out results; --check detects drift."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def tables():
    root = ROOT / 'results/heldout'
    summary = json.loads((root / 'summary.json').read_text())
    models = [('gpt-j-6b', 'GPT-J'), ('llama-3.1-8b', 'Llama-3.1-8B')]
    tasks = sorted(['antonym', 'country-capital', 'english-french', 'present-past',
                    'person-occupation', 'singular-plural', 'synonym', 'next_item'])
    expected = {(model, method, task, seed) for model, _ in models
                for method in ['fv', 'tv'] for task in tasks for seed in [100, 101, 102]}
    cells = summary['cells']
    assert len(cells) == 96
    assert {(r['model'], r['method'], r['task'], r['seed']) for r in cells} == expected
    assert all(r['icl_gap'] > 0 for r in cells)
    reference_partitions = None
    headline = [r'\begin{tabular}{llrr}', r'\toprule',
                r'Model & Method & Difference & 95\% CI \\', r'\midrule']
    accuracy_rows = [r'\begin{tabular}{llrrrrr}', r'\toprule',
                     r'Model & Method & Zero-shot & ICL & Real & Control (own) & Control (real) \\', r'\midrule']
    task_means = {}
    for model, label in models:
        for method in ['fv', 'tv']:
            arm = root / model / method
            partitions = json.loads((arm / 'partitions.json').read_text())
            if reference_partitions is None:
                reference_partitions = partitions
            assert partitions == reference_partitions
            rows = [r for r in cells if (r['model'], r['method']) == (model, method)]
            # Recompute contrasts directly from token predictions, independently of summary fields.
            raw_by_task = {t: [] for t in tasks}
            for row in rows:
                cell = json.loads((arm / row['task'] / str(row['seed']) / 'predictions.json').read_text())
                target = np.asarray(cell['targets']['test'])
                values = {v: np.mean(np.asarray(cell['predictions'][f'test/{v}/{layer}']) == target)
                          for v, layer in cell['selected'].items()}
                contrast = values[method] - np.mean([v for k, v in values.items() if k != method])
                assert np.isclose(contrast, row['difference'], rtol=0, atol=1e-12)
                raw_by_task[row['task']].append(contrast)
            means = np.array([np.mean(raw_by_task[t]) for t in tasks])
            aggregate = next(a for a in summary['aggregates'] if (a['model'], a['method']) == (model, method))
            draws = np.random.default_rng(20260905).choice(means, size=(10000, 8)).mean(axis=1)
            interval = np.quantile(draws, [.025, .975])
            assert np.allclose(interval, aggregate['ci95'], rtol=0, atol=1e-12)
            assert np.isclose(means.mean(), aggregate['mean_difference'], rtol=0, atol=1e-12)
            headline.append(f'{label} & {method.upper()} & ${100 * means.mean():+.1f}$ & '
                            f'$[{100 * interval[0]:.1f}, {100 * interval[1]:.1f}]$ ' + r'\\')
            values = [100 * np.mean([r[k] for r in rows]) for k in
                      ['zero', 'icl', 'accuracy', 'control_mean', 'conditional_control_mean']]
            accuracy_rows.append(f'{label} & {method.upper()} & ' + ' & '.join(f'{v:.1f}' for v in values) + r' \\')
            task_means[model, method] = means
    per_task = [r'\begin{tabular}{lrrrr}', r'\toprule',
                r'Task & GPT-J FV & GPT-J TV & Llama-3.1-8B FV & Llama-3.1-8B TV \\', r'\midrule']
    for i, task in enumerate(tasks):
        values = [100 * task_means[model, method][i] for model, _ in models for method in ['fv', 'tv']]
        per_task.append(task.replace('_', '-') + ' & ' + ' & '.join(f'${v:+.1f}$' for v in values) + r' \\')
    return {name: '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', '']) for name, lines in [
        ('heldout_results_table.tex', headline), ('heldout_accuracy_table.tex', accuracy_rows),
        ('heldout_task_table.tex', per_task)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, content in tables().items():
        path = ROOT / 'paper/generated' / name
        if args.check:
            assert path.read_text() == content, f'Stale table: {path}'
        else:
            path.write_text(content)
    print('Verified all 96 cells and three held-out paper tables.')
