#!/usr/bin/env python
"""Print the per-task TV recovery table (Appendix) as LaTeX from results/grid_*.json.

Usage: .venv/bin/python scripts/per_task_tv_table.py
"""
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import best_layer_recovery, control_recovery, load_rows

MODELS = [("gpt-j-6b", "GPT-J"), ("llama-3.1-8b", "L3.1-8B"),
          ("gemma-2-9b-it", "Gemma-2"), ("llama-3.1-70b", "L3.1-70B")]
TASKS = [
    ("linguistic", ["antonym", "synonym", "present-past", "singular-plural"]),
    ("knowledge", ["country-capital", "country-currency", "person-occupation", "park-country"]),
    ("translation", ["english-french", "english-spanish", "english-german"]),
    ("algorithmic", ["capitalize", "capitalize_first_letter", "lowercase_first_letter", "next_item", "prev_item"]),
]


def main():
    rows = load_rows()
    cells = {}
    for model, _ in MODELS:
        tv = defaultdict(list)
        for (task, seed), rr in best_layer_recovery(rows, model, "tv").items():
            tv[task].append(rr)
        sh = defaultdict(list)
        for (task, seed, _m), rr in control_recovery(rows, model, ("tv_control_shuffled_theta",)).items():
            sh[task].append(rr)
        for task in tv:
            cells[(model, task)] = (mean(tv[task]), mean(sh[task]) if sh[task] else None, len(tv[task]))

    print(r"\begin{tabular}{l" + "cc" * len(MODELS) + "}")
    print(r"\toprule")
    print(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{name}}}}}" for _, name in MODELS) + r" \\")
    print(r"\textbf{Task}" + " & TV & shuf." * len(MODELS) + r" \\")
    print(r"\midrule")
    for cat, tasks in TASKS:
        print(rf"\multicolumn{{{1 + 2 * len(MODELS)}}}{{l}}{{\emph{{{cat}}}}} \\")
        for task in tasks:
            out = [task.replace("_", "-")]
            for model, _ in MODELS:
                c = cells.get((model, task))
                if c is None:
                    out += ["--", "--"]
                else:
                    star = r"$^{\dagger}$" if c[2] == 1 else ""
                    out += [f"{c[0]:.2f}{star}", f"{c[1]:.2f}" if c[1] is not None else "--"]
            print(" & ".join(out) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
