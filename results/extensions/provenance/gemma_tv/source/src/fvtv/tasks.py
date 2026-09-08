"""Task loading, seeded demo/eval sampling, and ICL prompt construction.

Prompt format (fixed for this study): 'Q: {x}\\nA: {y}\\n\\n' per shot,
ending in an unterminated 'Q: {query}\\nA:' (no trailing space -- the tokenizer's
leading-space-attaches-to-next-token convention supplies the space, so the
target is the first token of ' {answer}').
"""
import json
from pathlib import Path

import numpy as np

TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tasks"


def load_task(name: str) -> list[dict]:
    """Load a task's word pairs from data/tasks/{name}.json."""
    with open(TASKS_DIR / f"{name}.json") as f:
        return json.load(f)


def list_tasks() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.json"))


def split_pairs(pairs: list[dict], seed: int, n_eval: int = 50) -> tuple[list[dict], list[dict]]:
    """Seeded, non-overlapping split into (train_pool, eval_pairs).

    train_pool is everything not in the eval split; demos for a given trial are
    drawn from train_pool by sample_demos(). n_eval is capped so at least 10
    items remain in the train pool (need >=10 for a full 10-shot prompt).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_eval = min(n_eval, max(len(pairs) - 10, 1))
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
    train_pool = [pairs[i] for i in train_idx]
    eval_pairs = [pairs[i] for i in eval_idx]
    return train_pool, eval_pairs


def sample_demos(train_pool: list[dict], seed: int, n_shot: int = 10) -> list[dict]:
    """Seeded, without-replacement sample of n_shot demo pairs from train_pool."""
    rng = np.random.default_rng(seed)
    n_shot = min(n_shot, len(train_pool))
    idx = rng.choice(len(train_pool), size=n_shot, replace=False)
    return [train_pool[i] for i in idx]


def build_icl_prompt(demos: list[dict], query_input: str, shuffle_labels: bool = False, shuffle_seed: int | None = None) -> str:
    """Build a k-shot prompt. If shuffle_labels, demo outputs are permuted among
    themselves (the AIE / TV-control 'task-ablated' condition) while inputs stay put."""
    if shuffle_labels:
        rng = np.random.default_rng(shuffle_seed)
        outputs = [d["output"] for d in demos]
        perm = rng.permutation(len(outputs))
        demos = [{"input": d["input"], "output": outputs[p]} for d, p in zip(demos, perm)]
    body = "".join(f"Q: {d['input']}\nA: {d['output']}\n\n" for d in demos)
    return body + f"Q: {query_input}\nA:"


def build_zeroshot_prompt(query_input: str) -> str:
    return f"Q: {query_input}\nA:"


def target_first_token_id(tokenizer, answer: str) -> int:
    """First token id of ' {answer}' -- handles GPT-2/Llama-style BPE prefix-space
    tokenization, where a token immediately following whitespace is a distinct
    vocab entry from the same word at string-start."""
    ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    return ids[0]
