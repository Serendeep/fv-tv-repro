#!/usr/bin/env python
"""Config-driven full grid: FV + TV + controls, all tasks x all seeds, for one
model at a time. Local (GPT-2) by default; --remote switches to an NDIF-hosted
model id with remote=True on every trace call.

Examples:
  .venv/bin/python scripts/run_grid.py --model gpt2 --tasks all --seeds 0,1,2,3,4
  .venv/bin/python scripts/run_grid.py --model llama-3.1-8b --remote --tasks antonym,country-capital
  .venv/bin/python scripts/run_grid.py --model llama-3.1-70b --remote --aie-layer-start 30 --aie-layer-end 55

NOTE: this script does not gate on HF_TOKEN presence -- it reads it from the
environment if set (never hardcode) and passes it through for gated models
(Llama). gpt-j-6b and gemma-2-9b-it are public and need no token.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nnsight import LanguageModel

from fvtv import controls, eval_icl, fv, stats, tasks, tv

# model key -> (hf id, default n_layers unused here (read from config at load
# time), role). GPT-OSS-120B removed (not currently hosted).
MODEL_REGISTRY = {
    "gpt2":               {"hf_id": "openai-community/gpt2",         "role": "pipeline-dev / smoke test"},
    "gpt-j-6b":           {"hf_id": "EleutherAI/gpt-j-6b",            "role": "exact-replication arm"},
    "llama-3.1-8b":       {"hf_id": "meta-llama/Llama-3.1-8B",        "role": "modern-base generalization"},
    "gemma-2-9b-it":      {"hf_id": "google/gemma-2-9b-it",           "role": "family + instruct axis"},
    "llama-3.1-70b":      {"hf_id": "meta-llama/Llama-3.1-70B",       "role": "scale axis"},
    "llama-3.3-70b-it":   {"hf_id": "meta-llama/Llama-3.3-70B-Instruct", "role": "optional extra"},
}


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY), help="registry key, see MODEL_REGISTRY")
    p.add_argument("--remote", action="store_true", help="run via NDIF (nnsight remote=True) instead of loading weights locally")
    p.add_argument("--tasks", default="all", help="comma-separated task names, or 'all' for every data/tasks/*.json")
    p.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated seeds (spec default: 5 seeds)")
    p.add_argument("--n-eval", type=int, default=50, help="eval items per task/seed (spec default 50; fallback 25 if NDIF is slow)")
    p.add_argument("--n-shot", type=int, default=10)
    p.add_argument("--k", type=int, default=10, help="top-k heads for FV")
    p.add_argument("--n-ex", type=int, default=32, help="ICL prompts averaged for mean head activations")
    p.add_argument("--n-aie-trials", type=int, default=25, help="shuffled-label trials for AIE")
    p.add_argument("--aie-layer-start", type=int, default=None, help="restrict AIE head sweep to [start, end) layers (compute-budget fallback for 70B+)")
    p.add_argument("--aie-layer-end", type=int, default=None)
    p.add_argument("--aie-max-rows", type=int, default=64, help="AIE batch-row cap; lower for big-vocab/tight-memory deployments (gemma-2: 16)")
    p.add_argument("--skip-fv", action="store_true", help="TV-only (70B: AIE dominates cost)")
    p.add_argument("--sweep-stride", type=int, default=1, help="inject/patch every Nth layer in FV/TV sweeps")
    p.add_argument("--fv-layers", default=None, help="comma-separated layers to sweep for FV injection (default: all layers)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=None, help="output json path (default results/grid_{model}.json)")
    return p.parse_args()


def load_model(hf_id, remote):
    hf_token = os.environ.get("HF_TOKEN")
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    if remote:
        # NDIF holds the weights; don't materialize them locally.
        model = LanguageModel(hf_id, **kwargs)
    else:
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass
        model = LanguageModel(hf_id, device_map=device, **kwargs)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"
    return model


def row(model_key, task, seed, method, layer, k, n_eval, sha, **extra):
    r = {"model": model_key, "task": task, "seed": seed, "method": method,
         "layer": layer, "k": k, "n_eval": n_eval, "git_sha": sha}
    r.update(extra)
    return r


def run_one(model, model_key, task_name, seed, args, sha, remote):
    rows = []
    task_pairs = tasks.load_task(task_name)
    cfg = fv.arch_config(model)

    acc_icl = eval_icl.icl_ceiling(model, task_pairs, seed, n_eval=args.n_eval, n_shot=args.n_shot, batch_size=args.batch_size, remote=remote)
    acc_zero = eval_icl.zero_shot_floor(model, task_pairs, seed, n_eval=args.n_eval, batch_size=args.batch_size, remote=remote)
    rows.append(row(model_key, task_name, seed, "icl_ceiling", None, None, args.n_eval, sha, accuracy=acc_icl))
    rows.append(row(model_key, task_name, seed, "zero_shot_floor", None, None, args.n_eval, sha, accuracy=acc_zero))

    _, zero_eval_pairs = tasks.split_pairs(task_pairs, seed, n_eval=args.n_eval)
    zero_prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in zero_eval_pairs]
    targets = [tasks.target_first_token_id(model.tokenizer, e["output"]) for e in zero_eval_pairs]

    # ---- FV ----
    if args.skip_fv:
        rows.extend(run_tv(model, model_key, task_name, seed, args, sha, remote, task_pairs, cfg, zero_prompts, targets, acc_zero, acc_icl))
        return rows
    mean_act = fv.compute_mean_head_activations(model, task_pairs, seed, n_ex=args.n_ex, n_shot=args.n_shot, remote=remote)
    aie_full = fv.compute_aie(model, task_pairs, mean_act, seed, n_trials=args.n_aie_trials, n_shot=args.n_shot, remote=remote, max_batch_rows=args.aie_max_rows)
    aie = aie_full
    if args.aie_layer_start is not None or args.aie_layer_end is not None:
        lo = args.aie_layer_start or 0
        hi = args.aie_layer_end or cfg["n_layers"]
        mask = aie_full.clone()
        mask[:lo] = float("-inf")
        mask[hi:] = float("-inf")
        aie = mask
    top_heads = fv.top_k_heads(aie, k=args.k)
    rand_heads = controls.pick_random_heads(cfg, args.k, seed=seed)
    # fetch only layers FV needs; cached across tasks/seeds
    needed_layers = {h[0] for h in top_heads} | {h[0] for h in rand_heads}
    out_proj_params = fv.grab_out_proj_params(model, cfg, layers=needed_layers, remote=remote)
    fv_vector = fv.compute_fv(model, mean_act, top_heads, out_proj_params=out_proj_params)

    fv_layers = [int(x) for x in args.fv_layers.split(",")] if args.fv_layers else list(range(0, cfg["n_layers"], args.sweep_stride))
    best_fv_layer, best_fv_acc = None, -1.0
    for L in fv_layers:
        preds = fv.inject_fv(model, zero_prompts, fv_vector, L, batch_size=args.batch_size, remote=remote)
        acc = eval_icl.accuracy(preds, targets)
        rr = stats.recovery_ratio(acc, acc_zero, acc_icl)
        rows.append(row(model_key, task_name, seed, "fv", L, args.k, args.n_eval, sha, accuracy=acc, recovery_ratio=rr))
        if acc > best_fv_acc:
            best_fv_layer, best_fv_acc = L, acc

    rand_vec = controls.random_vector(fv_vector, seed=seed)
    preds = fv.inject_fv(model, zero_prompts, rand_vec, best_fv_layer, batch_size=args.batch_size, remote=remote)
    acc_rand = eval_icl.accuracy(preds, targets)
    rows.append(row(model_key, task_name, seed, "fv_control_random_vector", best_fv_layer, args.k, args.n_eval, sha,
                     accuracy=acc_rand, recovery_ratio=stats.recovery_ratio(acc_rand, acc_zero, acc_icl)))

    rand_fv = fv.compute_fv(model, mean_act, rand_heads, out_proj_params=out_proj_params)
    preds = fv.inject_fv(model, zero_prompts, rand_fv, best_fv_layer, batch_size=args.batch_size, remote=remote)
    acc_randk = eval_icl.accuracy(preds, targets)
    rows.append(row(model_key, task_name, seed, "fv_control_random_k_heads", best_fv_layer, args.k, args.n_eval, sha,
                     accuracy=acc_randk, recovery_ratio=stats.recovery_ratio(acc_randk, acc_zero, acc_icl)))

    rows.extend(run_tv(model, model_key, task_name, seed, args, sha, remote, task_pairs, cfg, zero_prompts, targets, acc_zero, acc_icl))
    return rows


def run_tv(model, model_key, task_name, seed, args, sha, remote, task_pairs, cfg, zero_prompts, targets, acc_zero, acc_icl):
    rows = []
    thetas, demos, dummy_query, train_pool = tv.layer_sweep_setup(model, task_pairs, seed, n_shot=args.n_shot, remote=remote)
    best_tv_layer, best_tv_acc = None, -1.0
    for L in range(0, cfg["n_layers"], args.sweep_stride):
        preds = tv.patch_theta(model, zero_prompts, thetas[L], L, batch_size=args.batch_size, remote=remote)
        acc = eval_icl.accuracy(preds, targets)
        rr = stats.recovery_ratio(acc, acc_zero, acc_icl)
        rows.append(row(model_key, task_name, seed, "tv", L, None, args.n_eval, sha, accuracy=acc, recovery_ratio=rr))
        if acc > best_tv_acc:
            best_tv_layer, best_tv_acc = L, acc

    shuf_thetas, _, _ = controls.shuffled_label_theta_setup(model, task_pairs, seed, n_shot=args.n_shot, remote=remote)
    preds = tv.patch_theta(model, zero_prompts, shuf_thetas[best_tv_layer], best_tv_layer, batch_size=args.batch_size, remote=remote)
    acc_shuf = eval_icl.accuracy(preds, targets)
    rows.append(row(model_key, task_name, seed, "tv_control_shuffled_theta", best_tv_layer, None, args.n_eval, sha,
                     accuracy=acc_shuf, recovery_ratio=stats.recovery_ratio(acc_shuf, acc_zero, acc_icl)))
    return rows


def main():
    args = parse_args()
    sha = git_sha()
    entry = MODEL_REGISTRY[args.model]
    task_names = tasks.list_tasks() if args.tasks == "all" else args.tasks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[run_grid] model={args.model} ({entry['hf_id']}, {entry['role']}) remote={args.remote}")
    print(f"[run_grid] tasks={task_names} seeds={seeds} n_eval={args.n_eval}")
    model = load_model(entry["hf_id"], args.remote)
    fv.verify_arch(model, remote=args.remote)
    print("[run_grid] arch verified")

    out_path = Path(args.out) if args.out else ROOT / "results" / f"grid_{args.model}.json"
    out_path.parent.mkdir(exist_ok=True)
    # resume: skip (task, seed) pairs already in the output file
    all_rows = json.load(open(out_path)) if out_path.exists() else []
    done = {(r["task"], r["seed"]) for r in all_rows}

    t0 = time.time()
    for task_name in task_names:
        for seed in seeds:
            if (task_name, seed) in done:
                print(f"  {task_name} seed={seed}: already done, skipping")
                continue
            t1 = time.time()
            rows = run_one(model, args.model, task_name, seed, args, sha, remote=args.remote)
            all_rows.extend(rows)
            with open(out_path, "w") as f:  # write after every pair
                json.dump(all_rows, f, indent=2)
            print(f"  {task_name} seed={seed}: {len(rows)} rows in {time.time()-t1:.1f}s")

    print(f"[run_grid] {len(all_rows)} rows in {out_path} after {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
