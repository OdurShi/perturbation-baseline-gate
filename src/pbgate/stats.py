"""Bootstrap intervals for means over (seed, perturbation) records.

Records are crossed: the same perturbation is held out under several seeds and one
seed holds out many perturbations. Both factors are resampled independently with
replacement, and a record receives the product of its seed weight and its perturbation
weight. With a single seed this reduces to the ordinary bootstrap over perturbations.
"""

from __future__ import annotations

import numpy as np


def _codes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes, int(codes.max()) + 1


def crossed_bootstrap_means(
    values: np.ndarray,
    seeds: np.ndarray,
    perturbations: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
    chunk: int = 1000,
) -> np.ndarray:
    """Bootstrap distribution of the weighted mean of ``values`` (n_boot or n_boot x k).

    ``values`` has shape (N,) or (N, k); for (N, k) the same resampling weights are
    used for every column, so differences between columns are paired.
    """
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    s_codes, n_s = _codes(seeds)
    p_codes, n_p = _codes(perturbations)
    out = np.empty((n_boot, values.shape[1]))
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        w_s = rng.multinomial(n_s, np.full(n_s, 1.0 / n_s), size=size)
        w_p = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=size)
        weights = (w_s[:, s_codes] * w_p[:, p_codes]).astype(float)  # (size, N)
        total = weights.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + size] = (weights @ values) / total
    return out[:, 0] if squeeze else out


def mean_ci(
    values: np.ndarray,
    seeds: np.ndarray,
    perturbations: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile bootstrap interval (crossed resampling)."""
    boot = crossed_bootstrap_means(values, seeds, perturbations, n_boot=n_boot, rng=rng)
    low, high = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(values)), float(low), float(high)
