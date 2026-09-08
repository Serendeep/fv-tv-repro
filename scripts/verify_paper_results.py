#!/usr/bin/env python
"""Cross-check paper tables directly against raw JSON, independently of analyze.py.

No model access. Uses an explicit whitelist for main-grid filenames and computes
cluster intervals from resampled cluster sums and counts, not stats.py.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'gpt-j-6b': (14, 27), 'llama-3.1-8b': (8, 24),
            'gemma-2-9b-it': (5, 12), 'llama-3.1-70b': (16, 48)}


def interval(values):
    clusters = [values[k] for k in sorted(values)]
    counts = np.array([len(v) for v in clusters])
    sums = np.array([sum(v) for v in clusters])
    picks = np.random.default_rng(0).integers(0, len(clusters), size=(10000, len(clusters)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return sums.sum() / counts.sum(), lo, hi


def numbers(line):
    return [float(v) for v in re.findall(r'(?<![A-Za-z])(?<![A-Za-z0-9]-)-?\d+\.\d+', line)]


def main():
    rows = {}
    paths = [p for p in sorted((ROOT / 'results').glob('grid_*.json'))
             if re.fullmatch(r'grid_.+_(shard\d+|tv_newdata)\.json', p.name)]
    assert len(paths) == 10, len(paths)
    for path in paths:
        for r in json.loads(path.read_text()):
            rows[(r['model'], r['task'], r['seed'], r['method'], r['layer'])] = r
    rows = list(rows.values())
    baselines = {}
    for r in rows:
        if r['method'] in ('icl_ceiling', 'zero_shot_floor'):
            baselines.setdefault((r['model'], r['task'], r['seed']), {})[r['method']] = r['accuracy']
    for model, expected in EXPECTED.items():
        cells = [k for k in baselines if k[0] == model]
        assert (len({k[1] for k in cells}), len(cells)) == expected
    for r in rows:
        if r.get('recovery_ratio') is not None:
            b = baselines[(r['model'], r['task'], r['seed'])]
            gap = b['icl_ceiling'] - b['zero_shot_floor']
            assert gap >= .2 - 1e-12
            assert abs((r['accuracy'] - b['zero_shot_floor']) / gap - r['recovery_ratio']) < 1e-12

    summaries = {}
    pairs_summary = {}
    for model in EXPECTED:
        for method in ('fv', 'tv'):
            sweep = defaultdict(dict)
            ctrl = defaultdict(list)
            for r in rows:
                if r['model'] != model:
                    continue
                key = (r['task'], r['seed'])
                if r['method'] == method:
                    sweep[key][r['layer']] = r['recovery_ratio']
                elif r['method'] in ({'fv_control_random_vector', 'fv_control_random_k_heads'}
                                     if method == 'fv' else {'tv_control_shuffled_theta'}):
                    ctrl[key].append(r['recovery_ratio'])
            if not sweep:
                continue
            best = {k: max(v.values()) for k, v in sweep.items()}
            assert set(best) == set(ctrl)
            clustered, diffs = defaultdict(list), defaultdict(list)
            for k, v in best.items():
                clustered[k[0]].append(v)
                diffs[k[0]].append(v - np.mean(ctrl[k]))
            a = np.array(list(best.values()))
            b = np.array([v for vv in ctrl.values() for v in vv])
            pooled_var = ((len(a)-1)*np.var(a, ddof=1)+(len(b)-1)*np.var(b, ddof=1))/(len(a)+len(b)-2)
            d = (np.mean(a)-np.mean(b))/np.sqrt(pooled_var)
            cv = []
            for (task, seed), values in sweep.items():
                if seed == 0 or (task, 0) not in sweep:
                    continue
                base = sweep[(task, 0)]
                layer = max(sorted(base), key=base.get)
                if layer in values:
                    cv.append(values[layer])
            m, lo, hi = interval(clustered)
            summaries[model, method] = (m, lo, hi, np.mean(cv), np.mean(b), d)
            pairs_summary[model, method] = interval(diffs)
    for r in csv.DictReader((ROOT / 'results/summary.csv').open()):
        exp = summaries[r['model'], r['method']]
        np.testing.assert_allclose([float(r[k]) for k in ('mean', 'ci_lo', 'ci_hi', 'control_mean', 'cohens_d')],
                                   [exp[i] for i in (0, 1, 2, 4, 5)], atol=1e-12, rtol=0)
    for r in csv.DictReader((ROOT / 'results/paired_differences.csv').open()):
        np.testing.assert_allclose([float(r[k]) for k in ('diff_mean', 'diff_lo', 'diff_hi')],
                                   pairs_summary[r['model'], r['method']], atol=1e-12, rtol=0)
    table = (ROOT / 'paper/generated/main_results_table.tex').read_text().splitlines()
    numeric_rows = [numbers(line) for line in table if re.search(r' & (FV|TV) & ', line)]
    assert len(numeric_rows) == len(summaries)
    for got, expected in zip(numeric_rows, summaries.values()):
        rounded = [float(f'{x:.2f}') for x in expected]
        assert got == rounded, (got, rounded)

    paired_tex = (ROOT / 'paper/generated/paired_checks.tex').read_text()
    paired_rows = paired_tex.split('; ')
    assert len(paired_rows) == len(pairs_summary)
    for got, expected in zip(paired_rows, pairs_summary.values()):
        # Strip the model label before extracting numeric estimates.
        got = got.split('$', 1)[1]
        assert numbers(got) == [float(f'{x:.2f}') for x in expected]
    main_table = (ROOT / 'paper/generated/tv_control_table.tex').read_text().splitlines()
    variant_names = ['real, replace', 'label-shuffled', 'cross-task donor', 'template swap', 'real, additive']
    variant_methods = ['tv', 'tv_control_shuffled_theta', 'tv_control_cross_task', 'tv_control_template', 'tv_additive']
    extra_models = ['gpt-j-6b', 'llama-3.1-8b', 'llama-3.1-70b']
    for name, method in zip(variant_names, variant_methods):
        got = numbers(next(line for line in main_table if line.startswith(name + ' &')))
        expected = []
        for model in extra_models:
            source = rows if method in ('tv', 'tv_control_shuffled_theta') else json.loads(
                (ROOT / f'results/grid_{model}_tvextra.json').read_text())
            by_cell = defaultdict(list)
            for r in source:
                if r['model'] == model and r['method'] == method:
                    by_cell[r['task'], r['seed']].append(r['recovery_ratio'])
            expected.append(round(float(np.mean([max(v) for v in by_cell.values()])), 2))
        assert got == expected, (name, got, expected)

    # Explicit independent checks for the interpretation driving this revision.
    gptj = json.loads((ROOT / 'results/grid_gpt-j-6b_tvextra.json').read_text())
    for task, expected in [('capitalize', .97), ('capitalize_first_letter', .87)]:
        values = [r['recovery_ratio'] for r in gptj if r['task'] == task and r['method'] == 'tv_control_template']
        assert round(float(np.mean(values)), 2) == expected
    for r in json.loads((ROOT / 'results/grid_gemma-2-9b-it_crosstask.json').read_text()):
        if r['method'] == 'fv_xtask':
            assert round(r['recovery_ratio'], 2) <= .08
    for task in ('country-capital', 'country-currency'):
        assert len(json.loads((ROOT / f'data/tasks/{task}.json').read_text())) == 197
    print(f'PASS: {len(paths)} main-grid files, {len(rows)} rows, {len(summaries)} estimates, paired intervals,')
    print('      displayed main table, template-rescue values, Gemma cross-task bound, and task sizes.')


if __name__ == '__main__':
    main()
