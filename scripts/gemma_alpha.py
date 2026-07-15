#!/usr/bin/env python
"""Gemma-2 FV injection-strength ablation: does scaling the FV by alpha
rescue it? Tests the residual-norm explanation for the FV failure
(Gemma-2 multiplies embeddings by sqrt(d_model), so residual norms run
large relative to the models FVs were developed on).

Usage: .venv/bin/python scripts/gemma_alpha.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nnsight import LanguageModel

from fvtv import eval_icl, fv, stats, tasks

ALPHAS = (1.0, 2.0, 4.0, 8.0)
TASKS = ("antonym", "country-capital")
SEED = 0
N_EVAL, N_EX, N_TRIALS, STRIDE = 15, 16, 5, 4

model = LanguageModel("google/gemma-2-9b-it")
fv.verify_arch(model, remote=True)
cfg = fv.arch_config(model)

out = ROOT / "results" / "gemma_alpha.json"
out_rows = json.load(open(out)) if out.exists() else []  # resume
done = {(r["task"], r["alpha"]) for r in out_rows}

for task_name in TASKS:
    if all((task_name, a) in done for a in ALPHAS):
        continue
    pairs = tasks.load_task(task_name)
    acc_icl = eval_icl.icl_ceiling(model, pairs, SEED, n_eval=N_EVAL, remote=True)
    acc_zero = eval_icl.zero_shot_floor(model, pairs, SEED, n_eval=N_EVAL, remote=True)
    _, eval_pairs = tasks.split_pairs(pairs, SEED, n_eval=N_EVAL)
    prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in eval_pairs]
    targets = [tasks.target_first_token_id(model.tokenizer, e["output"]) for e in eval_pairs]

    mean_act = fv.compute_mean_head_activations(model, pairs, SEED, n_ex=N_EX, remote=True)
    aie = fv.compute_aie(model, pairs, mean_act, SEED, n_trials=N_TRIALS, remote=True, max_batch_rows=16)
    heads = fv.top_k_heads(aie, k=10)
    vec = fv.compute_fv(model, mean_act, heads,
                        out_proj_params=fv.grab_out_proj_params(model, cfg, layers={h[0] for h in heads}, remote=True))

    for alpha in ALPHAS:
        if (task_name, alpha) in done:
            continue
        best = -1.0
        for L in range(0, cfg["n_layers"], STRIDE):
            preds = fv.inject_fv(model, prompts, alpha * vec, L, remote=True)
            best = max(best, eval_icl.accuracy(preds, targets))
        rr = stats.recovery_ratio(best, acc_zero, acc_icl)
        out_rows.append({"task": task_name, "alpha": alpha, "best_acc": best,
                         "acc_icl": acc_icl, "acc_zero": acc_zero, "recovery_ratio": rr})
        json.dump(out_rows, open(out, "w"), indent=2)  # checkpoint per alpha
        print(f"{task_name} alpha={alpha}: best_acc={best:.3f} rr={rr:.3f}")

print(f"wrote {out} ({len(out_rows)} rows)")
