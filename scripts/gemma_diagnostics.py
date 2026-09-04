#!/usr/bin/env python
"""Review-panel diagnostics for the Gemma-2 FV null (antonym, seed 0):
D1 norm ratio: ||FV|| vs residual ||h|| at swept layers.
D2 positive control: residual-scale random vector added at the same site;
   accuracy must degrade if the additive write lands.
D3 corrected injection: add FV at o_proj.output (pre post-attention norm),
   the basis the vector was actually built in.
D4 prediction flips: clean vs injected argmax ids, all three conditions.

Checkpoints per stage into results/gemma_diagnostics.json.

Usage: uv run python scripts/gemma_diagnostics.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nnsight import LanguageModel

from fvtv import eval_icl, fv, stats, tasks
from fvtv.eval_icl import encode_batch, with_retries

TASK = sys.argv[1] if len(sys.argv) > 1 else "antonym"
SEED, N_EVAL = 0, 15
OUT = ROOT / "results" / f"gemma_diagnostics_{TASK}.json"
state = json.load(open(OUT)) if OUT.exists() else {}


def save():
    json.dump(state, open(OUT, "w"), indent=2)


model = LanguageModel("google/gemma-2-9b-it")
fv.verify_arch(model, remote=True)
cfg = fv.arch_config(model)
blocks = fv._blocks(model, cfg)
pairs = tasks.load_task(TASK)
_, eval_pairs = tasks.split_pairs(pairs, SEED, n_eval=N_EVAL)
prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in eval_pairs]
targets = [tasks.target_first_token_id(model.tokenizer, e["output"]) for e in eval_pairs]

if "baseline" not in state:
    state["baseline"] = {
        "acc_icl": eval_icl.icl_ceiling(model, pairs, SEED, n_eval=N_EVAL, remote=True),
        "acc_zero": eval_icl.zero_shot_floor(model, pairs, SEED, n_eval=N_EVAL, remote=True),
        "clean_preds": eval_icl.predict_top1(model, prompts, remote=True),
    }
    save()

if "fv_norms" not in state:
    mean_act = fv.compute_mean_head_activations(model, pairs, SEED, n_ex=16, remote=True)
    aie = fv.compute_aie(model, pairs, mean_act, SEED, n_trials=5, remote=True, max_batch_rows=16)
    heads = fv.top_k_heads(aie, k=10)
    vec = fv.compute_fv(model, mean_act, heads,
                        out_proj_params=fv.grab_out_proj_params(model, cfg, layers={h[0] for h in heads}, remote=True))
    torch.save(vec, ROOT / "results" / f"gemma_fv_{TASK}_s0.pt")
    torch.save(aie, ROOT / "results" / f"gemma_aie_{TASK}_s0.pt")
    state["fv_norms"] = {"fv_norm": float(vec.norm()), "top_heads": [[h[0], h[1], h[2]] for h in heads],
                        "aie_top10_mean": float(sum(h[2] for h in heads) / 10),
                        "aie_flat_std": float(aie.std())}
    save()
else:
    vec = torch.load(ROOT / "results" / f"gemma_fv_{TASK}_s0.pt")

SWEEP = [8, 16, 24, 32]
if "residual_norms" not in state:
    enc = encode_batch(model.tokenizer, prompts[:4])
    blks = [blocks[L] for L in SWEEP]

    def _run():
        with torch.no_grad(), model.trace(enc, remote=True):
            ns = torch.stack([b.output[:, -1, :].float().norm(dim=-1).mean().cpu() for b in blks]).save()
        return ns

    ns = with_retries(_run)
    state["residual_norms"] = {str(L): float(n) for L, n in zip(SWEEP, ns)}
    save()

if "positive_control" not in state:
    res = {}
    for L in SWEEP:
        h_norm = state["residual_norms"][str(L)]
        rand = torch.randn(cfg["hidden"])
        rand = rand / rand.norm() * h_norm  # residual-scale, not FV-scale
        preds = fv.inject_fv(model, prompts, rand, L, remote=True)
        flips = sum(int(p != c) for p, c in zip(preds, state["baseline"]["clean_preds"]))
        res[str(L)] = {"acc": eval_icl.accuracy(preds, targets), "pred_flips": flips}
    state["positive_control"] = res
    save()

if "corrected_injection" not in state:
    res = {}
    for L in SWEEP:
        proj = fv._out_proj(blocks[L], cfg)
        vloc = vec

        def _run():
            with torch.no_grad(), model.trace(encode_batch(model.tokenizer, prompts), remote=True):
                proj.output[:, -1, :] += vloc.to(proj.output)
                logits = model.lm_head.output[:, -1, :].save()
            return logits

        logits = with_retries(_run)
        preds = logits.argmax(dim=-1).tolist()
        flips = sum(int(p != c) for p, c in zip(preds, state["baseline"]["clean_preds"]))
        res[str(L)] = {"acc": eval_icl.accuracy(preds, targets), "pred_flips": flips}
    state["corrected_injection"] = res
    save()

if "block_injection_flips" not in state:
    res = {}
    for L in SWEEP:
        preds = fv.inject_fv(model, prompts, vec, L, remote=True)
        flips = sum(int(p != c) for p, c in zip(preds, state["baseline"]["clean_preds"]))
        res[str(L)] = {"acc": eval_icl.accuracy(preds, targets), "pred_flips": flips}
    state["block_injection_flips"] = res
    save()

print(json.dumps(state, indent=2))
