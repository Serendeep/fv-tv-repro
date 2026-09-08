#!/usr/bin/env python
"""Generate the paper's numerical table bodies from the main-grid results.

Run after analyze.py and panel_revision_stats.py. --check reports stale files.
"""
import argparse
import csv
import io
import json
import re
from collections import defaultdict
from contextlib import redirect_stdout

from analyze import ROOT, best_layer_recovery, control_recovery, load_rows
from per_task_tv_table import main as per_task_table

MODELS = [("gpt-j-6b", "GPT-J"), ("llama-3.1-8b", "L3.1-8B"),
          ("gemma-2-9b-it", "Gemma-2"), ("llama-3.1-70b", "L3.1-70B")]


def artifacts():
    grid = load_rows()
    summary = {(r["model"], r["method"]): r for r in
               csv.DictReader((ROOT / "results/summary.csv").open())}
    sweep = defaultdict(dict)
    for r in grid:
        if r["method"] in ("fv", "tv") and r.get("recovery_ratio") is not None:
            sweep[(r["model"], r["task"], r["seed"], r["method"])][r["layer"]] = r["recovery_ratio"]
    lines = []
    for model, label in MODELS:
        for method in ("fv", "tv"):
            if (model, method) not in summary:
                continue
            r = summary[(model, method)]
            cv = []
            for task in sorted({k[1] for k in sweep if k[0] == model and k[3] == method}):
                s0 = sweep.get((model, task, 0, method))
                if not s0:
                    continue
                layer = max(sorted(s0), key=s0.get)
                for seed in (1, 2):
                    sw = sweep.get((model, task, seed, method), {})
                    if layer in sw:
                        cv.append(sw[layer])
            ctrl = float(r['control_mean'])
            # Avoid negative zero in the displayed control column.
            control = f"{ctrl:.2f}" if abs(ctrl) >= .005 else "0.00"
            fields = [label if method == 'fv' or model == 'llama-3.1-70b' else '', method.upper(),
                      f"{float(r['mean']):.2f} [{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}]",
                      f"{sum(cv)/len(cv):.2f}", control, f"{float(r['cohens_d']):.2f}"]
            lines.append(' & '.join(fields) + r' \\')
    out = {"main_results_table.tex": (r"\begin{tabular}{llcccc}" + "\n" +
           r"\toprule \textbf{Model} & \textbf{M.} & \textbf{Recovery [CI]} & \textbf{CV} & \textbf{Ctrl.} & $d$ \\" + "\n" +
           r"\midrule" + "\n" + '\n'.join(lines) + "\n" + r"\bottomrule \end{tabular}" + "\n")}

    paired = []
    labels = dict(MODELS)
    for r in csv.DictReader((ROOT / "results/paired_differences.csv").open()):
        paired.append(f"{labels[r['model']]} {r['method'].upper()} "
                      f"${float(r['diff_mean']):.2f}$ $[{float(r['diff_lo']):.2f},{float(r['diff_hi']):.2f}]$")
    out['paired_checks.tex'] = '; '.join(paired) + '.\n'

    variants = [('real, replace', None), ('label-shuffled', 'tv_control_shuffled_theta'),
                ('cross-task donor', 'tv_control_cross_task'), ('template swap', 'tv_control_template'),
                ('real, additive', 'tv_additive')]
    controls = defaultdict(list)
    extra_models = [m for m, _ in MODELS if m != 'gemma-2-9b-it']
    for model in extra_models:
        extra = json.loads((ROOT / f'results/grid_{model}_tvextra.json').read_text())
        for name, method in variants:
            if method is None:
                pairs = best_layer_recovery(grid, model, 'tv')
            elif method == 'tv_control_shuffled_theta':
                pairs = {(t, s): v for (t, s, _), v in control_recovery(grid, model, (method,)).items()}
            else:
                pairs = {}
                for r in extra:
                    if r['method'] == method:
                        key = (r['task'], r['seed'])
                        pairs[key] = max(pairs.get(key, float('-inf')), r['recovery_ratio'])
            controls[name].append(f"{sum(pairs.values())/len(pairs):.2f}")
    out['tv_control_table.tex'] = '\n'.join(name + ' & ' + ' & '.join(controls[name]) + r' \\'
                                          for name, _ in variants) + '\n'
    out['tv_control_table.tex'] = (r"\begin{tabular}{lccc}" + "\n" +
        r"\toprule \textbf{$\theta$ variant} & \textbf{GPT-J} & \textbf{L3.1-8B} & \textbf{L3.1-70B} \\" + "\n" +
        r"\midrule" + "\n" + out['tv_control_table.tex'] +
        r"\midrule cells / tasks & 27 / 14 & 24 / 8 & 48 / 16 \\" + "\n" +
        r"\bottomrule \end{tabular}" + "\n")
    with redirect_stdout(io.StringIO()) as capture:
        per_task_table()
    out['per_task_table.tex'] = capture.getvalue()
    paired_rows = re.findall(r'([\w.-]+) (FV|TV) \$([^$]+)\$ \$([^$]+)\$', out['paired_checks.tex'])
    assert len(paired_rows) == 7
    out['paired_grid_table.tex'] = '\n'.join([
        r'\begin{tabular}{llrr}', r'\toprule',
        r'Model & Method & Difference & 95\% CI \\', r'\midrule',
        *[f'{model} & {method} & ${value}$ & ${ci}$' + r' \\'
          for model, method, value, ci in paired_rows],
        r'\bottomrule', r'\end{tabular}', ''])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'paper/generated'
    stale = []
    for name, content in artifacts().items():
        path = directory / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            directory.mkdir(exist_ok=True)
            path.write_text(content)
            print(f'wrote {path.relative_to(ROOT)}')
    if stale:
        raise SystemExit('Stale paper tables: ' + ', '.join(stale))
    if args.check:
        print('Paper table sources match generated values.')


if __name__ == '__main__':
    main()
