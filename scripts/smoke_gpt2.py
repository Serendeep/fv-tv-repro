#!/usr/bin/env python
"""End-to-end smoke test on GPT-2-small: 4 tasks x 1 seed x 20 eval items, all
methods + controls, full layer sweeps. Writes results/smoke_gpt2.json and
prints a summary table. Not a reported result (pipeline validation only --
GPT-2-small is far weaker than the paper's target models).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from nnsight import LanguageModel

from fvtv import controls, eval_icl, fv, stats, tasks, tv

TASKS = ["singular-plural", "present-past", "capitalize", "person-occupation"]
SEED = 0
N_EVAL = 20
N_SHOT = 10
K_HEADS = 10
N_EX_MEAN_ACT = 16
N_AIE_TRIALS = 8


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def load_model():
    for device in ("mps", "cpu"):
        try:
            model = LanguageModel("openai-community/gpt2", device_map=device)
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.tokenizer.padding_side = "left"
            print(f"[smoke] loaded gpt2 on {device}")
            return model
        except Exception as e:
            print(f"[smoke] device {device} failed: {e}", file=sys.stderr)
    raise RuntimeError("could not load model on mps or cpu")


def row(model_name, task, seed, method, layer, k, n_eval, sha, **extra):
    r = {"model": model_name, "task": task, "seed": seed, "method": method,
         "layer": layer, "k": k, "n_eval": n_eval, "git_sha": sha}
    r.update(extra)
    return r


def main():
    t_start = time.time()
    model = load_model()
    sha = git_sha()
    model_name = "gpt2"
    rows = []

    for task_name in TASKS:
        t_task = time.time()
        print(f"\n=== task: {task_name} ===")
        task_pairs = tasks.load_task(task_name)

        acc_icl = eval_icl.icl_ceiling(model, task_pairs, SEED, n_eval=N_EVAL, n_shot=N_SHOT)
        acc_zero = eval_icl.zero_shot_floor(model, task_pairs, SEED, n_eval=N_EVAL)
        print(f"  icl_ceiling={acc_icl:.3f}  zero_shot_floor={acc_zero:.3f}")
        rows.append(row(model_name, task_name, SEED, "icl_ceiling", None, None, N_EVAL, sha, accuracy=acc_icl))
        rows.append(row(model_name, task_name, SEED, "zero_shot_floor", None, None, N_EVAL, sha, accuracy=acc_zero))

        # ---- FV ----
        mean_act = fv.compute_mean_head_activations(model, task_pairs, SEED, n_ex=N_EX_MEAN_ACT, n_shot=N_SHOT)
        aie = fv.compute_aie(model, task_pairs, mean_act, SEED, n_trials=N_AIE_TRIALS, n_shot=N_SHOT)
        top_heads = fv.top_k_heads(aie, k=K_HEADS)
        out_proj_params = fv.grab_out_proj_params(model, fv.arch_config(model))
        fv_vector = fv.compute_fv(model, mean_act, top_heads, out_proj_params=out_proj_params)

        _, zero_eval_pairs = tasks.split_pairs(task_pairs, SEED, n_eval=N_EVAL)
        zero_prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in zero_eval_pairs]
        targets = [tasks.target_first_token_id(model.tokenizer, e["output"]) for e in zero_eval_pairs]

        cfg = fv.arch_config(model)
        best_fv_layer, best_fv_acc = None, -1.0
        for L in range(cfg["n_layers"]):
            preds = fv.inject_fv(model, zero_prompts, fv_vector, L)
            acc = eval_icl.accuracy(preds, targets)
            rr = stats.recovery_ratio(acc, acc_zero, acc_icl)
            rows.append(row(model_name, task_name, SEED, "fv", L, K_HEADS, N_EVAL, sha, accuracy=acc, recovery_ratio=rr))
            if acc > best_fv_acc:
                best_fv_layer, best_fv_acc = L, acc
        best_fv_rr = stats.recovery_ratio(best_fv_acc, acc_zero, acc_icl)
        print(f"  FV best layer={best_fv_layer} acc={best_fv_acc:.3f} recovery={best_fv_rr:.3f}")

        # FV controls, evaluated at the best FV layer
        rand_vec = controls.random_vector(fv_vector, seed=SEED)
        preds = fv.inject_fv(model, zero_prompts, rand_vec, best_fv_layer)
        acc_rand = eval_icl.accuracy(preds, targets)
        rr_rand = stats.recovery_ratio(acc_rand, acc_zero, acc_icl)
        rows.append(row(model_name, task_name, SEED, "fv_control_random_vector", best_fv_layer, K_HEADS, N_EVAL, sha, accuracy=acc_rand, recovery_ratio=rr_rand))

        rand_fv, rand_heads = controls.random_k_heads_fv(model, mean_act, K_HEADS, seed=SEED, out_proj_params=out_proj_params)
        preds = fv.inject_fv(model, zero_prompts, rand_fv, best_fv_layer)
        acc_randk = eval_icl.accuracy(preds, targets)
        rr_randk = stats.recovery_ratio(acc_randk, acc_zero, acc_icl)
        rows.append(row(model_name, task_name, SEED, "fv_control_random_k_heads", best_fv_layer, K_HEADS, N_EVAL, sha, accuracy=acc_randk, recovery_ratio=rr_randk))
        print(f"  FV controls: random_vector acc={acc_rand:.3f} (rr={rr_rand:.3f})  random_k_heads acc={acc_randk:.3f} (rr={rr_randk:.3f})")

        # ---- TV ----
        thetas, demos, dummy_query, train_pool = tv.layer_sweep_setup(model, task_pairs, SEED, n_shot=N_SHOT)
        best_tv_layer, best_tv_acc = None, -1.0
        for L in range(cfg["n_layers"]):
            preds = tv.patch_theta(model, zero_prompts, thetas[L], L)
            acc = eval_icl.accuracy(preds, targets)
            rr = stats.recovery_ratio(acc, acc_zero, acc_icl)
            rows.append(row(model_name, task_name, SEED, "tv", L, None, N_EVAL, sha, accuracy=acc, recovery_ratio=rr))
            if acc > best_tv_acc:
                best_tv_layer, best_tv_acc = L, acc
        best_tv_rr = stats.recovery_ratio(best_tv_acc, acc_zero, acc_icl)
        print(f"  TV best layer={best_tv_layer} acc={best_tv_acc:.3f} recovery={best_tv_rr:.3f}")

        # TV control: theta from shuffled-label demos, evaluated at the best TV layer
        shuf_thetas, _, _ = controls.shuffled_label_theta_setup(model, task_pairs, SEED, n_shot=N_SHOT)
        preds = tv.patch_theta(model, zero_prompts, shuf_thetas[best_tv_layer], best_tv_layer)
        acc_shuf = eval_icl.accuracy(preds, targets)
        rr_shuf = stats.recovery_ratio(acc_shuf, acc_zero, acc_icl)
        rows.append(row(model_name, task_name, SEED, "tv_control_shuffled_theta", best_tv_layer, None, N_EVAL, sha, accuracy=acc_shuf, recovery_ratio=rr_shuf))
        print(f"  TV control: shuffled_theta acc={acc_shuf:.3f} (rr={rr_shuf:.3f})")

        print(f"  task time: {time.time() - t_task:.1f}s")

    Path(ROOT / "results").mkdir(exist_ok=True)
    out_path = ROOT / "results" / "smoke_gpt2.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {len(rows)} rows to {out_path}")
    print(f"total time: {time.time() - t_start:.1f}s")

    print_summary(rows)


def print_summary(rows):
    print("\n=== summary (best layer per method) ===")
    header = f"{'task':<18}{'icl':>7}{'zero':>7}{'fv_best':>9}{'fv_rr':>8}{'rand_v':>8}{'rand_k':>8}{'tv_best':>9}{'tv_rr':>8}{'tv_shuf':>9}"
    print(header)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], {})[r["method"] + (f"@{r['layer']}" if r["method"] in ("fv", "tv") else "")] = r

    for task_name in TASKS:
        icl = next(r["accuracy"] for r in rows if r["task"] == task_name and r["method"] == "icl_ceiling")
        zero = next(r["accuracy"] for r in rows if r["task"] == task_name and r["method"] == "zero_shot_floor")
        fv_rows = [r for r in rows if r["task"] == task_name and r["method"] == "fv"]
        tv_rows = [r for r in rows if r["task"] == task_name and r["method"] == "tv"]
        fv_best = max(fv_rows, key=lambda r: r["accuracy"])
        tv_best = max(tv_rows, key=lambda r: r["accuracy"])
        rand_v = next(r for r in rows if r["task"] == task_name and r["method"] == "fv_control_random_vector")
        rand_k = next(r for r in rows if r["task"] == task_name and r["method"] == "fv_control_random_k_heads")
        tv_shuf = next(r for r in rows if r["task"] == task_name and r["method"] == "tv_control_shuffled_theta")
        print(f"{task_name:<18}{icl:>7.3f}{zero:>7.3f}{fv_best['accuracy']:>9.3f}{fv_best['recovery_ratio']:>8.3f}"
              f"{rand_v['accuracy']:>8.3f}{rand_k['accuracy']:>8.3f}{tv_best['accuracy']:>9.3f}{tv_best['recovery_ratio']:>8.3f}{tv_shuf['accuracy']:>9.3f}")


if __name__ == "__main__":
    main()
