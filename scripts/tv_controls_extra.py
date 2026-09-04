#!/usr/bin/env python
"""Task-vector controls beyond label shuffling, run on an arm's existing
(task, seed) cells. Each cell's best TV layer, ICL ceiling, and zero-shot floor are read
from results/grid_*.json, so only the new conditions cost remote jobs:

  tv_control_cross_task  theta from a same-category donor task's demos,
                         patched at this task's best layer (format, no rule)
  tv_control_template    theta from this task's own demos in an arrow
                         template, patched into the Q:/A: zero-shot prompt
                         (rule, other format)
  tv_additive            theta added to the residual instead of replacing
                         it, swept over the arm's layers

Usage: uv run python scripts/tv_controls_extra.py --model gpt-j-6b --remote
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import load_rows
from run_grid import MODEL_REGISTRY, git_sha, load_model
from fvtv import eval_icl, fv, stats, tasks, tv

DONOR = {
    "antonym": "synonym", "synonym": "antonym",
    "present-past": "singular-plural", "singular-plural": "present-past",
    "country-capital": "country-currency", "country-currency": "country-capital",
    "person-occupation": "park-country", "park-country": "person-occupation",
    "english-french": "english-german", "english-german": "english-french", "english-spanish": "english-french",
    "capitalize": "capitalize_first_letter", "capitalize_first_letter": "capitalize",
    "lowercase_first_letter": "capitalize_first_letter",
    "next_item": "prev_item", "prev_item": "next_item",
}


def arrow_prompt(demos, query):
    return "".join(f"{d['input']} -> {d['output']}\n" for d in demos) + f"{query} ->"


def cells_for(rows, model_key):
    """{(task, seed): dict(best_layer, layers, n_eval, acc_icl, acc_zero)}"""
    out = {}
    for r in rows:
        if r["model"] != model_key:
            continue
        c = out.setdefault((r["task"], r["seed"]), {"tv": {}})
        if r["method"] == "tv" and r.get("recovery_ratio") is not None:
            c["tv"][r["layer"]] = r["accuracy"]
            c["n_eval"] = r["n_eval"]
        elif r["method"] == "icl_ceiling":
            c["acc_icl"] = r["accuracy"]
        elif r["method"] == "zero_shot_floor":
            c["acc_zero"] = r["accuracy"]
    keep = {}
    for key, c in out.items():
        if not c["tv"] or "acc_icl" not in c or "acc_zero" not in c:
            continue
        layers = sorted(c["tv"])
        best = max(layers, key=lambda L: (c["tv"][L], -L))
        keep[key] = dict(best_layer=best, layers=layers, n_eval=c["n_eval"], acc_icl=c["acc_icl"], acc_zero=c["acc_zero"])
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--remote", action="store_true")
    p.add_argument("--tasks", default="all")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    sha = git_sha()
    cells = cells_for(load_rows(), args.model)
    if args.tasks != "all":
        want = set(args.tasks.split(","))
        cells = {k: v for k, v in cells.items() if k[0] in want}
    out_path = ROOT / "results" / f"grid_{args.model}_tvextra.json"
    all_rows = json.load(open(out_path)) if out_path.exists() else []
    done = {(r["task"], r["seed"]) for r in all_rows}
    print(f"[tvextra] model={args.model} cells={len(cells)} done={len(done)}")

    model = load_model(MODEL_REGISTRY[args.model]["hf_id"], args.remote)
    fv.verify_arch(model, remote=args.remote)

    def row(task, seed, method, layer, n_eval, acc, c, **extra):
        r = {"model": args.model, "task": task, "seed": seed, "method": method, "layer": layer, "k": None,
             "n_eval": n_eval, "git_sha": sha, "accuracy": acc,
             "recovery_ratio": stats.recovery_ratio(acc, c["acc_zero"], c["acc_icl"])}
        r.update(extra)
        return r

    for (task, seed), c in sorted(cells.items()):
        if (task, seed) in done:
            continue
        t0 = time.time()
        pairs = tasks.load_task(task)
        _, eval_pairs = tasks.split_pairs(pairs, seed, n_eval=c["n_eval"])
        zero_prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in eval_pairs]
        targets = [tasks.target_first_token_id(model.tokenizer, e["output"]) for e in eval_pairs]
        L = c["best_layer"]
        rows = []

        donor = DONOR[task]
        donor_thetas, _, _, _ = tv.layer_sweep_setup(model, tasks.load_task(donor), seed, remote=args.remote)
        preds = tv.patch_theta(model, zero_prompts, donor_thetas[L], L, batch_size=args.batch_size, remote=args.remote)
        rows.append(row(task, seed, "tv_control_cross_task", L, c["n_eval"], eval_icl.accuracy(preds, targets), c, donor=donor))

        thetas, demos, dummy_query, _ = tv.layer_sweep_setup(model, pairs, seed, remote=args.remote)
        tmpl_thetas = tv.extract_theta_all_layers(model, demos, dummy_query, remote=args.remote, prompt=arrow_prompt(demos, dummy_query))
        preds = tv.patch_theta(model, zero_prompts, tmpl_thetas[L], L, batch_size=args.batch_size, remote=args.remote)
        rows.append(row(task, seed, "tv_control_template", L, c["n_eval"], eval_icl.accuracy(preds, targets), c))

        for Ls in c["layers"]:
            preds = tv.patch_theta(model, zero_prompts, thetas[Ls], Ls, batch_size=args.batch_size, remote=args.remote, additive=True)
            rows.append(row(task, seed, "tv_additive", Ls, c["n_eval"], eval_icl.accuracy(preds, targets), c))

        all_rows.extend(rows)
        with open(out_path, "w") as f:
            json.dump(all_rows, f, indent=2)
        best_add = max(r["recovery_ratio"] for r in rows if r["method"] == "tv_additive")
        print(f"  {task} s{seed}: cross={rows[0]['recovery_ratio']:.2f} template={rows[1]['recovery_ratio']:.2f} additive_best={best_add:.2f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
