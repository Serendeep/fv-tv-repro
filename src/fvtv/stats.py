"""Recovery ratio, bootstrap CIs, effect sizes, verdict summaries."""
import numpy as np


def recovery_ratio(acc_method, acc_zeroshot, acc_icl):
    """(acc_method - acc_zeroshot) / (acc_icl - acc_zeroshot). Undefined (nan)
    when the ICL ceiling equals the zero-shot floor (no headroom to recover)."""
    denom = acc_icl - acc_zeroshot
    if abs(denom) < 1e-9:
        return float("nan")
    return (acc_method - acc_zeroshot) / denom


def cluster_bootstrap_ci(pairs_values, n_boot=10000, ci=0.95, seed=0):
    """Mean + percentile bootstrap CI resampling clusters (tasks), then pooling
    their values. pairs_values: dict cluster_id -> list of values."""
    clusters = [v for v in pairs_values.values() if v]
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(clusters)
    flat_mean = float(np.mean([x for c in clusters for x in c]))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, n, size=n)
        vals = [x for j in picks for x in clusters[j]]
        boot[i] = np.mean(vals)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return flat_mean, float(lo), float(hi)


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=0):
    """Mean + percentile bootstrap CI over `values` (e.g. per-task or
    per-task-per-seed recovery ratios). Returns (mean, lo, hi)."""
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = values.size
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def cohens_d(a, b):
    """Cohen's d for two independent samples (a = method, b = matched control),
    pooled standard deviation."""
    a = np.asarray([v for v in a if not np.isnan(v)], dtype=float)
    b = np.asarray([v for v in b if not np.isnan(v)], dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std < 1e-9:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def summarize(rows, group_cols, value_col):
    """rows: list[dict]. Groups by group_cols, returns list[dict] with
    mean/lo/hi bootstrap CI of value_col per group. No pandas dependency
    (keeps stats.py usable standalone); scripts can wrap the output in a
    DataFrame themselves."""
    groups = {}
    for r in rows:
        key = tuple(r[c] for c in group_cols)
        groups.setdefault(key, []).append(r[value_col])
    out = []
    for key, values in groups.items():
        mean, lo, hi = bootstrap_ci(values)
        entry = dict(zip(group_cols, key))
        entry.update({f"{value_col}_mean": mean, f"{value_col}_lo": lo, f"{value_col}_hi": hi, "n": len(values)})
        out.append(entry)
    return out
