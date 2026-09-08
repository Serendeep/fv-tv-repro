"""ICL ceiling / zero-shot floor (top-1 first-token accuracy), plus shared
batched-forward-pass utilities reused by fv.py, tv.py, controls.py.
"""
import time

import torch

from . import tasks

_TRANSIENT = ("outofmemory", "timeout", "timed out", "handshake", "connection",
              "disconnect", "submitting request", "try again", "queue", "503", "502")


def with_retries(fn, attempts=40, wait=90, max_wait=300):
    """Retry a trace closure on transient remote failures. Patience over
    process death: a killed run redoes its whole (task, seed) pair, waiting
    does not."""
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:
            s = (type(e).__name__ + " " + str(e)).lower()
            if a == attempts - 1 or not any(k in s for k in _TRANSIENT):
                raise
            time.sleep(min(wait * (1.3 ** a), max_wait))


def ensure_left_padding(tokenizer):
    """Left-pad so the last real token of every sequence in a batch sits at
    position -1, regardless of prompt length -- required for ragged-length ICL
    prompts sharing one batched trace."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def encode_batch(tokenizer, prompts):
    ensure_left_padding(tokenizer)
    return dict(tokenizer(prompts, return_tensors="pt", padding=True))


def predict_top1(model, prompts, batch_size=16, remote=False):
    """argmax token id at the last prompt position, one per prompt."""
    tokenizer = model.tokenizer
    preds = []
    for i in range(0, len(prompts), batch_size):
        enc = encode_batch(tokenizer, prompts[i:i + batch_size])

        def _run():
            with torch.no_grad(), model.trace(enc, remote=remote):
                logits_last = model.lm_head.output[:, -1, :].save()
            return logits_last

        preds.extend(with_retries(_run).argmax(dim=-1).tolist())
    return preds


def accuracy(pred_ids, target_ids) -> float:
    assert len(pred_ids) == len(target_ids)
    if not pred_ids:
        return float("nan")
    correct = sum(int(p == t) for p, t in zip(pred_ids, target_ids))
    return correct / len(pred_ids)


def icl_ceiling(model, task_pairs, seed, n_eval=50, n_shot=10, batch_size=16, remote=False):
    tokenizer = model.tokenizer
    train_pool, eval_pairs = tasks.split_pairs(task_pairs, seed, n_eval=n_eval)
    demos = tasks.sample_demos(train_pool, seed, n_shot=n_shot)
    prompts = [tasks.build_icl_prompt(demos, e["input"]) for e in eval_pairs]
    targets = [tasks.target_first_token_id(tokenizer, e["output"]) for e in eval_pairs]
    preds = predict_top1(model, prompts, batch_size=batch_size, remote=remote)
    return accuracy(preds, targets)


def zero_shot_floor(model, task_pairs, seed, n_eval=50, batch_size=16, remote=False):
    tokenizer = model.tokenizer
    _, eval_pairs = tasks.split_pairs(task_pairs, seed, n_eval=n_eval)
    prompts = [tasks.build_zeroshot_prompt(e["input"]) for e in eval_pairs]
    targets = [tasks.target_first_token_id(tokenizer, e["output"]) for e in eval_pairs]
    preds = predict_top1(model, prompts, batch_size=batch_size, remote=remote)
    return accuracy(preds, targets)
