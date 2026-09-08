"""Function Vectors (Todd et al., ICLR 2024), reproduced against the original
repo's math (ericwtodd/function_vectors: src/utils/extract_utils.py,
src/compute_indirect_effect.py) but expressed via nnsight instead of baukit.

Pipeline:
  (a) compute_mean_head_activations -- per-(layer,head) task-conditioned mean
      of the pre-out_proj attention output, at the last token, over N_ex
      10-shot prompts.
  (b) compute_aie -- Average Indirect Effect: patch each head's task-mean into
      shuffled-label runs at the last token, score recovery of the correct
      answer's probability.
  (c) compute_fv -- FV = sum of the top-k heads' means, each projected into the
      residual stream via that head's own attention out_proj.
  (d) inject_fv -- add the FV additively to a zero-shot run's residual stream
      at a chosen layer, last token.
"""
import numpy as np
import torch

from . import tasks
from .eval_icl import encode_batch, with_retries

# Module-tree adapter keyed off HF config.model_type. block_out_tuple:
# whether block.output is (hidden, ...) or a bare tensor; differs per family
# even on the same transformers version, so verify_arch checks it at startup.
_FAMILIES = {
    "gpt2":   {"gpt2_style": True,  "attn_attr": "attn",      "out_proj_attr": "c_proj",   "is_conv1d": True,  "block_out_tuple": False},
    "gptj":   {"gpt2_style": True,  "attn_attr": "attn",      "out_proj_attr": "out_proj", "is_conv1d": False, "block_out_tuple": True},
    "llama":  {"gpt2_style": False, "attn_attr": "self_attn", "out_proj_attr": "o_proj",   "is_conv1d": False, "block_out_tuple": False},
    "gemma2": {"gpt2_style": False, "attn_attr": "self_attn", "out_proj_attr": "o_proj",   "is_conv1d": False, "block_out_tuple": False},
}
_FALLBACK_FAMILY = "llama"  # most modern HF causal LMs (incl. MoE variants) use this tree shape


def arch_config(model):
    """Architecture-adaptive geometry, keyed off config.model_type."""
    cfg = model.config
    family = getattr(cfg, "model_type", None)
    if family not in _FAMILIES:
        family = _FALLBACK_FAMILY
    spec = _FAMILIES[family]
    gpt2_style = spec["gpt2_style"]
    n_layers = cfg.n_layer if gpt2_style else cfg.num_hidden_layers
    n_heads = cfg.n_head if gpt2_style else cfg.num_attention_heads
    hidden = cfg.n_embd if gpt2_style else cfg.hidden_size
    # gemma-2 sets head_dim explicitly (16*256=4096, hidden=3584)
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    return {
        "family": family,
        "gpt2_style": gpt2_style,
        "attn_attr": spec["attn_attr"],
        "out_proj_attr": spec["out_proj_attr"],
        "is_conv1d": spec["is_conv1d"],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden": hidden,
        "head_dim": head_dim,
        "attn_in": n_heads * head_dim,  # out_proj input width; == hidden except Gemma-2
        "block_out_tuple": spec["block_out_tuple"],
    }


def block_hidden(block_envoy, cfg):
    """Residual-stream output, tuple-unwrapped per family."""
    out = block_envoy.output
    return out[0] if cfg["block_out_tuple"] else out


def verify_arch(model, remote=False):
    """One trace to assert the family's block_out_tuple flag matches reality;
    a wrong flag mis-slices silently."""
    cfg = arch_config(model)
    blocks = _blocks(model, cfg)
    with torch.no_grad(), model.trace(" ", remote=remote):
        out = blocks[0].output.save()
    got = isinstance(out, tuple)
    if got != cfg["block_out_tuple"]:
        raise RuntimeError(f"{cfg['family']}: block_out_tuple={cfg['block_out_tuple']} but model returns {type(out).__name__}")
    hidden = (out[0] if got else out).shape[-1]
    if hidden != cfg["hidden"]:
        raise RuntimeError(f"{cfg['family']}: hidden {hidden} != config {cfg['hidden']}")


def _blocks(model, cfg):
    return model.transformer.h if cfg["gpt2_style"] else model.model.layers


def _out_proj(block, cfg):
    attn = getattr(block, cfg["attn_attr"])
    return getattr(attn, cfg["out_proj_attr"])


_PARAM_CACHE = {}  # (model_id, layer) -> (weight, is_conv1d)


def grab_out_proj_params(model, cfg, layers=None, remote=False):
    """out_proj weights for `layers` (default all), cached per (model, layer).
    Remote weight downloads are expensive; fetch only what FV needs."""
    model_id = getattr(model.config, "_name_or_path", id(model))
    layers = list(range(cfg["n_layers"])) if layers is None else sorted(set(layers))
    missing = [L for L in layers if (model_id, L) not in _PARAM_CACHE]
    if missing:
        blocks = _blocks(model, cfg)
        projs = [_out_proj(blocks[L], cfg) for L in missing]

        def _run():  # remote traces only return simple local saves, so stack once
            with torch.no_grad(), model.trace(" ", remote=remote):
                ws = torch.stack([p.weight.cpu() for p in projs]).save()
            return ws

        ws = with_retries(_run)
        for L, w in zip(missing, ws.cpu().unbind(0)):
            _PARAM_CACHE[(model_id, L)] = (w, cfg["is_conv1d"])
    return {L: _PARAM_CACHE[(model_id, L)] for L in layers}


def _project_head(x_head, weight, is_conv1d):
    """One head's residual-stream contribution via out_proj. Bias excluded:
    it is a layer-level constant, so summing k heads would count it k times."""
    x = x_head.reshape(1, 1, -1).to(weight.device).to(weight.dtype)
    if is_conv1d:  # Conv1D: y = x @ W, W: (in, out)
        y = x @ weight
    else:  # nn.Linear: y = x @ W.T, W: (out, in)
        y = x @ weight.T
    return y.reshape(-1).float().cpu()


def compute_mean_head_activations(model, task_pairs, seed, n_ex=32, n_shot=10, remote=False, batch_size=8):
    """(n_layers, n_heads, head_dim) mean of the pre-out_proj per-head output at
    the last token, over n_ex freshly-sampled (demos, query) 10-shot ICL
    prompts, batched into a single trace."""
    cfg = arch_config(model)
    tokenizer = model.tokenizer
    train_pool, _ = tasks.split_pairs(task_pairs, seed, n_eval=50)
    rng = np.random.default_rng(seed)
    prompts = []
    for _ in range(n_ex):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        demos = tasks.sample_demos(train_pool, trial_seed, n_shot=n_shot)
        query = train_pool[int(rng.integers(0, len(train_pool)))]["input"]
        prompts.append(tasks.build_icl_prompt(demos, query))

    blocks = _blocks(model, cfg)
    # hoisted: the remote compiler can't resolve helper calls inside trace bodies
    projs = [_out_proj(blocks[L], cfg) for L in range(cfg["n_layers"])]
    nh, hd = cfg["n_heads"], cfg["head_dim"]

    total = torch.zeros(cfg["n_layers"], nh, hd)
    for i in range(0, len(prompts), batch_size):  # chunked: full n_ex batch OOMs tight deployments
        chunk = prompts[i:i + batch_size]
        enc = encode_batch(tokenizer, chunk)

        def _run():
            with torch.no_grad(), model.trace(enc, remote=remote):
                acts = torch.stack([  # .cpu(): layers shard across GPUs on 70B
                    p.input[:, -1, :].reshape(-1, nh, hd).mean(dim=0).cpu() for p in projs
                ], dim=0).save()
            return acts

        total += with_retries(_run).float().cpu() * len(chunk)
    return total / len(prompts)


def compute_aie(model, task_pairs, mean_activations, seed, n_trials=25, n_shot=10, remote=False, layer_chunk=None, max_batch_rows=64):
    """(n_layers, n_heads) Average Indirect Effect: for n_trials shuffled-label
    prompts, patch each head's task-mean activation into that head's own
    pre-out_proj slice at the last token and measure the resulting change in
    the correct answer's softmax probability, vs. the unpatched (clean,
    shuffled-label) run. (layer, head) pairs are folded into the batch dim,
    one prompt replica per pair and `layer_chunk` layers per trace, so a trial
    costs ceil(n_layers/layer_chunk) traces, not n_layers*n_heads."""
    cfg = arch_config(model)
    tokenizer = model.tokenizer
    n_heads, n_layers = cfg["n_heads"], cfg["n_layers"]
    if layer_chunk is None:
        layer_chunk = max(1, max_batch_rows // n_heads)
    train_pool, eval_pairs = tasks.split_pairs(task_pairs, seed, n_eval=50)
    blocks = _blocks(model, cfg)
    rng = np.random.default_rng(seed + 999)
    aie = torch.zeros(n_trials, n_layers, n_heads)
    chunks = [list(range(s, min(s + layer_chunk, n_layers))) for s in range(0, n_layers, layer_chunk)]

    for t in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        demos = tasks.sample_demos(train_pool, trial_seed, n_shot=n_shot)
        query_pair = eval_pairs[int(rng.integers(0, len(eval_pairs)))]
        prompt = tasks.build_icl_prompt(demos, query_pair["input"], shuffle_labels=True, shuffle_seed=trial_seed)
        target_id = tasks.target_first_token_id(tokenizer, query_pair["output"])

        enc1 = encode_batch(tokenizer, [prompt])

        def _clean():
            with torch.no_grad(), model.trace(enc1, remote=remote):
                clean_logits = model.lm_head.output[:, -1, :].save()
            return clean_logits

        clean_prob = torch.softmax(with_retries(_clean)[0], dim=-1)[target_id].item()

        pending = list(chunks)
        while pending:
            chunk = pending.pop(0)
            enc_batch = encode_batch(tokenizer, [prompt] * (n_heads * len(chunk)))
            projs = [_out_proj(blocks[L], cfg) for L in chunk]
            hd = cfg["head_dim"]

            def _patched():
                with torch.no_grad(), model.trace(enc_batch, remote=remote):
                    for ci, L in enumerate(chunk):  # row ci*n_heads+h patches (L, h)
                        c_in_heads = projs[ci].input[:, -1, :].reshape(-1, n_heads, hd)
                        for h in range(n_heads):
                            c_in_heads[ci * n_heads + h, h, :] = mean_activations[L, h].to(c_in_heads)
                    logits_last = model.lm_head.output[:, -1, :].save()
                return logits_last

            try:
                logits_last = _patched()
            except Exception as e:
                if "OutOfMemory" in str(e):
                    if len(chunk) > 1:  # halve on server OOM
                        mid = len(chunk) // 2
                        pending[:0] = [chunk[:mid], chunk[mid:]]
                        continue
                    logits_last = with_retries(_patched)  # minimal chunk: wait out pressure
                else:
                    logits_last = with_retries(_patched)  # non-OOM transient
            probs = torch.softmax(logits_last, dim=-1)[:, target_id]
            for ci, L in enumerate(chunk):
                aie[t, L, :] = probs[ci * n_heads:(ci + 1) * n_heads] - clean_prob

    return aie.mean(dim=0)


def top_k_heads(aie, k=10):
    """List of (layer, head, score) for the k heads with highest AIE."""
    flat = aie.reshape(-1)
    vals, inds = torch.topk(flat, k=min(k, flat.numel()), largest=True)
    n_heads = aie.shape[1]
    return [(int(i) // n_heads, int(i) % n_heads, float(v)) for i, v in zip(inds, vals)]


def compute_fv(model, mean_activations, heads, out_proj_params=None):
    """FV = sum over `heads` (list of (L,H) or (L,H,score)) of that head's
    task-mean activation, projected into the residual stream via its own
    attention out_proj. Mirrors Todd's compute_function_vector."""
    cfg = arch_config(model)
    if out_proj_params is None:
        out_proj_params = grab_out_proj_params(model, cfg, layers={h[0] for h in heads})
    fv = torch.zeros(cfg["hidden"])
    for h in heads:
        L, H = h[0], h[1]
        weight, is_conv1d = out_proj_params[L]
        x = torch.zeros(cfg["attn_in"], device=mean_activations.device)
        x[H * cfg["head_dim"]:(H + 1) * cfg["head_dim"]] = mean_activations[L, H]
        fv += _project_head(x, weight, is_conv1d)
    return fv


def inject_fv(model, prompts, fv_vector, layer, batch_size=16, remote=False):
    """Additive injection of fv_vector into the residual stream at `layer`,
    last token, on (typically zero-shot) prompts. Returns argmax token id per
    prompt."""
    cfg = arch_config(model)
    tokenizer = model.tokenizer
    blocks = _blocks(model, cfg)
    preds = []
    blk, is_tuple = blocks[layer], cfg["block_out_tuple"]
    for i in range(0, len(prompts), batch_size):
        enc = encode_batch(tokenizer, prompts[i:i + batch_size])

        def _run():
            with torch.no_grad(), model.trace(enc, remote=remote):
                h = blk.output[0] if is_tuple else blk.output
                h[:, -1, :] += fv_vector.to(h)
                logits_last = model.lm_head.output[:, -1, :].save()
            return logits_last

        preds.extend(with_retries(_run).argmax(dim=-1).tolist())
    return preds
