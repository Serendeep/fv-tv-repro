"""Controls shared by the FV and TV experiments:
  (a) norm-matched random Gaussian vector -- FV injection control.
  (b) random-k-heads FV -- same construction as fv.compute_fv, but heads picked
      uniformly at random instead of by AIE rank.
  (c) shuffled-label theta -- same extraction as tv.layer_sweep_setup, but the
      demo prompt's labels are permuted (task-ablated) before extraction.
"""
import numpy as np
import torch

from . import tasks
from .fv import arch_config, compute_fv
from .tv import extract_theta_all_layers, pick_dummy_query


def random_vector(reference_vector, seed):
    """A random Gaussian vector with the same L2 norm as reference_vector,
    for use as the FV-injection null control."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(reference_vector.shape[-1])
    v = v / np.linalg.norm(v) * reference_vector.norm().item()
    return torch.tensor(v, dtype=reference_vector.dtype)


def pick_random_heads(cfg, k, seed):
    """k uniformly-random (layer, head) pairs, without replacement."""
    rng = np.random.default_rng(seed)
    all_heads = [(L, H) for L in range(cfg["n_layers"]) for H in range(cfg["n_heads"])]
    idx = rng.choice(len(all_heads), size=min(k, len(all_heads)), replace=False)
    return [all_heads[i] for i in idx]


def random_k_heads_fv(model, mean_activations, k, seed, out_proj_params=None):
    """FV built from k uniformly-random (layer, head) pairs instead of the
    top-k by AIE -- isolates "does head *selection* matter" from "does summing
    projected head means and adding them work at all"."""
    heads = pick_random_heads(arch_config(model), k, seed)
    fv = compute_fv(model, mean_activations, heads, out_proj_params=out_proj_params)
    return fv, heads


def shuffled_label_theta_setup(model, task_pairs, seed, n_shot=10, remote=False):
    """Same as tv.layer_sweep_setup, but the 10-shot demo prompt has its
    labels shuffled before theta extraction (task-ablated control: theta
    should carry no usable task signal)."""
    train_pool, _ = tasks.split_pairs(task_pairs, seed, n_eval=50)
    demos = tasks.sample_demos(train_pool, seed, n_shot=n_shot)
    dummy_query = pick_dummy_query(train_pool, demos, seed)

    # extract_theta_all_layers builds its prompt from plain demos; shuffle the
    # labels here to get the task-ablated variant.
    rng = np.random.default_rng(seed)
    outputs = [d["output"] for d in demos]
    perm = rng.permutation(len(outputs))
    shuffled_demos = [{"input": d["input"], "output": outputs[p]} for d, p in zip(demos, perm)]

    thetas = extract_theta_all_layers(model, shuffled_demos, dummy_query, remote=remote)
    return thetas, shuffled_demos, dummy_query
