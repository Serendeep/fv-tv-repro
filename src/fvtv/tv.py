"""Task Vectors (Hendel et al., EMNLP 2023 Findings), reproduced via nnsight.

theta = residual-stream hidden state at the last token of a 10-shot demo
prompt built with a dummy query (a query-independence probe -- theta should
not depend on which query follows the demos), at layer L. Patching theta into
a zero-shot run at the same layer/position recovers ICL behavior without
running the demos at inference time. Layer sweep over all layers.
"""
import numpy as np
import torch

from . import tasks
from .eval_icl import encode_batch, with_retries
from .fv import arch_config, _blocks, block_hidden


def pick_dummy_query(train_pool, demos, seed):
    """A held-out input from train_pool, excluded from the demo set, used as
    the query in theta's extraction prompt. A real in-domain word keeps the
    query-independence probe meaningful; a nonsense placeholder would push
    the model off-distribution for unrelated reasons."""
    demo_inputs = {d["input"] for d in demos}
    candidates = [p for p in train_pool if p["input"] not in demo_inputs]
    pool = candidates if candidates else train_pool
    rng = np.random.default_rng(seed)
    return pool[int(rng.integers(0, len(pool)))]["input"]


def extract_theta_all_layers(model, demos, dummy_query, remote=False, prompt=None):
    """{layer: (hidden,)} residual-stream vector at each layer's output, last
    token, from a single forward pass of a 10-shot prompt ending in the dummy
    query (cheaper than one trace per layer). `prompt` overrides the Q:/A:
    prompt built from demos (template-swap control)."""
    cfg = arch_config(model)
    blocks = _blocks(model, cfg)
    tokenizer = model.tokenizer
    prompt = prompt or tasks.build_icl_prompt(demos, dummy_query)
    enc = encode_batch(tokenizer, [prompt])
    blks, is_tuple = [blocks[L] for L in range(cfg["n_layers"])], cfg["block_out_tuple"]

    def _run():  # remote traces only return simple local saves, so stack once
        with torch.no_grad(), model.trace(enc, remote=remote):
            thetas = torch.stack([  # .cpu(): layers shard across GPUs on 70B
                (b.output[0] if is_tuple else b.output)[0, -1, :].cpu() for b in blks
            ], dim=0).save()
        return thetas

    thetas = with_retries(_run)
    return {L: thetas[L].float().cpu() for L in range(cfg["n_layers"])}


def extract_theta(model, demos, dummy_query, layer, remote=False):
    return extract_theta_all_layers(model, demos, dummy_query, remote=remote)[layer]


def patch_theta(model, prompts, theta, layer, batch_size=16, remote=False, additive=False):
    """Replace (default) or add (`additive=True`) theta at `layer`, last
    token, on (typically zero-shot) prompts. Returns argmax token id/prompt."""
    cfg = arch_config(model)
    blocks = _blocks(model, cfg)
    tokenizer = model.tokenizer
    preds = []
    blk, is_tuple = blocks[layer], cfg["block_out_tuple"]
    for i in range(0, len(prompts), batch_size):
        enc = encode_batch(tokenizer, prompts[i:i + batch_size])

        def _run():
            with torch.no_grad(), model.trace(enc, remote=remote):
                h = blk.output[0] if is_tuple else blk.output
                if additive:
                    h[:, -1, :] = h[:, -1, :] + theta.to(h)
                else:
                    h[:, -1, :] = theta.to(h)
                logits_last = model.lm_head.output[:, -1, :].save()
            return logits_last

        preds.extend(with_retries(_run).argmax(dim=-1).tolist())
    return preds


def layer_sweep_setup(model, task_pairs, seed, n_shot=10, remote=False):
    """One fixed 10-shot demo set + dummy query for this (task, seed), and
    theta extracted at every layer from it in a single trace."""
    train_pool, _ = tasks.split_pairs(task_pairs, seed, n_eval=50)
    demos = tasks.sample_demos(train_pool, seed, n_shot=n_shot)
    dummy_query = pick_dummy_query(train_pool, demos, seed)
    thetas = extract_theta_all_layers(model, demos, dummy_query, remote=remote)
    return thetas, demos, dummy_query, train_pool
