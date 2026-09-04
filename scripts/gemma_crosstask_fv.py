#!/usr/bin/env python
"""Camera-ready: Todd et al.-style cross-task head selection on Gemma-2-9b-it.

The main grid selects top-10 heads per task. Todd et al. average AIE across
tasks, pick one head set per model, and scale k with model size. This script
runs that protocol on the five Gemma-2 tasks (seed 0): per-task mean head
activations + AIE maps (cached), one shared head set per k in {10, 20, 40},
an injection sweep per task, and a norm-matched random-vector control at the
best layer. ICL ceiling and zero-shot floor are reused from the grid rows.

Usage: uv run python scripts/gemma_crosstask_fv.py
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import load_rows
from run_grid import git_sha, load_model
from fvtv import controls, eval_icl, fv, stats, tasks

MODEL_KEY, HF_ID = "gemma-2-9b-it", "google/gemma-2-9b-it"
TASKS = ["antonym", "capitalize", "country-capital", "next_item", "synonym"]
SEED, N_EVAL, N_EX, N_TRIALS, MAX_ROWS, STRIDE = 0, 25, 32, 5, 16, 2
KS = (10, 20, 40)
OUT = ROOT / "results" / f"grid_{MODEL_KEY}_crosstask.json"


def floors():
    out = {}
    for r in load_rows():
        if r["model"] == MODEL_KEY and r["seed"] == SEED and r["method"] in ("icl_ceiling", "zero_shot_floor"):
            out.setdefault(r["task"], {})[r["method"]] = r["accuracy"]
    return out


def main():
    sha = git_sha()
    base = floors()
    model = load_model(HF_ID, remote=True)
    fv.verify_arch(model, remote=True)
    cfg = fv.arch_config(model)

    mean_act, aie = {}, {}
    for task in TASKS:
        pairs = tasks.load_task(task)
        ma_path = ROOT / "results" / f"gemma_xtask_meanact_{task}_s{SEED}.pt"
        aie_path = ROOT / "results" / f"gemma_xtask_aie_{task}_s{SEED}.pt"
        t0 = time.time()
        if ma_path.exists():
            mean_act[task] = torch.load(ma_path)
        else:
            mean_act[task] = fv.compute_mean_head_activations(model, pairs, SEED, n_ex=N_EX, remote=True)
            torch.save(mean_act[task], ma_path)
        if aie_path.exists():
            aie[task] = torch.load(aie_path)
        else:
            aie[task] = fv.compute_aie(model, pairs, mean_act[task], SEED, n_trials=N_TRIALS, remote=True, max_batch_rows=MAX_ROWS)
            torch.save(aie[task], aie_path)
        print(f"[xtask] {task}: AIE top head {aie[task].max().item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    aie_avg = torch.stack([aie[t] for t in TASKS]).mean(0)
    all_rows = json.load(open(OUT)) if OUT.exists() else []
    done = {(r["task"], r["k"]) for r in all_rows}

    for k in KS:
        heads = fv.top_k_heads(aie_avg, k=k)
        params = fv.grab_out_proj_params(model, cfg, layers={h[0] for h in heads}, remote=True)
        print(f"[xtask] k={k} shared heads (layer, head): {[(h[0], h[1]) for h in heads]}", flush=True)
        for task in TASKS:
            if (task, k) in done:
                continue
            t0 = time.time()
            pairs = tasks.load_task(task)
            _, eval_pairs = tasks.split_pairs(pairs, SEED, n_eval=N_EVAL)
            prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in eval_pairs]
            targets = [tasks.target_first_token_id(model.tokenizer, e["output"]) for e in eval_pairs]
            acc_icl, acc_zero = base[task]["icl_ceiling"], base[task]["zero_shot_floor"]
            vec = fv.compute_fv(model, mean_act[task], heads, out_proj_params=params)

            def row(method, layer, acc):
                return {"model": MODEL_KEY, "task": task, "seed": SEED, "method": method, "layer": layer, "k": k,
                        "n_eval": N_EVAL, "git_sha": sha, "accuracy": acc,
                        "recovery_ratio": stats.recovery_ratio(acc, acc_zero, acc_icl)}

            rows, best_L, best_acc = [], None, -1.0
            for L in range(0, cfg["n_layers"], STRIDE):
                acc = eval_icl.accuracy(fv.inject_fv(model, prompts, vec, L, remote=True), targets)
                rows.append(row("fv_xtask", L, acc))
                if acc > best_acc:
                    best_L, best_acc = L, acc
            rand = controls.random_vector(vec, seed=SEED)
            acc_r = eval_icl.accuracy(fv.inject_fv(model, prompts, rand, best_L, remote=True), targets)
            rows.append(row("fv_xtask_control_random_vector", best_L, acc_r))
            all_rows.extend(rows)
            json.dump(all_rows, open(OUT, "w"), indent=2)
            print(f"  k={k} {task}: best recovery {max(r['recovery_ratio'] for r in rows[:-1]):.2f} at layer {best_L}, ||fv||={vec.norm():.1f}, control {rows[-1]['recovery_ratio']:.2f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
