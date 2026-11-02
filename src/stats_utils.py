"""Small statistics helpers shared by the analysis scripts."""
import numpy as np
from scipy.stats import pointbiserialr, spearmanr


def _stat(kind):
    if kind == "spearman":
        return lambda a, b: spearmanr(a, b)[0]
    if kind == "pointbiserial":
        return lambda a, b: pointbiserialr(a, b)[0]
    raise ValueError(kind)


def bootstrap_ci(x, y, kind="spearman", n_boot=2000, alpha=0.05, seed=0):
    """Point estimate and percentile bootstrap CI for a correlation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    fn, rng, n = _stat(kind), np.random.default_rng(seed), len(x)
    if np.std(x) == 0 or np.std(y) == 0:      # constant input: correlation undefined
        return float("nan"), float("nan"), float("nan")
    est = fn(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        boots.append(fn(xb, yb))
    if not boots:
        return float(est), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)
