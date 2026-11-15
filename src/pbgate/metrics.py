"""Per-perturbation metrics, all based on the change from the control mean."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data import PseudobulkData

# metric name -> True when larger values are better
HIGHER_IS_BETTER = {
    "pearson_all": True,
    "pearson_de20": True,
    "mse_de20": False,
    "l2_top1000": False,
}
METRICS = tuple(HIGHER_IS_BETTER)
N_DE = 20
N_TOP_EXPRESSED = 1000
# The perturbed genes themselves are left out of every gene set: under CRISPRa their own
# up-regulation is the intervention, not a response, and it is trivially additive.
# ``exclude_targets=False`` restores the conventional definition (sensitivity analysis).
EXCLUDE_TARGETS = True


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either vector is constant (correlation undefined)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def target_mask(condition: str, data: PseudobulkData) -> np.ndarray:
    """Boolean mask over genes: True for the genes perturbed in ``condition``."""
    mask = np.zeros(len(data.genes), dtype=bool)
    gene_index = data.gene_index
    for gene in condition.split("+"):
        if gene in gene_index:
            mask[gene_index[gene]] = True
    return mask


def gene_sets(
    condition: str, data: PseudobulkData, exclude_targets: bool | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gene index sets of one perturbation: (all genes, top DE genes, top expressed genes).

    With ``exclude_targets`` the perturbed genes themselves are removed from all three
    sets before the sets are truncated to their nominal size (20 and 1,000).
    """
    exclude = EXCLUDE_TARGETS if exclude_targets is None else exclude_targets
    row = data.index[condition]
    order = data.expression_order
    if not exclude:
        return np.arange(len(data.genes)), data.de_genes[row][:N_DE], order[:N_TOP_EXPRESSED]
    mask = target_mask(condition, data)
    de = data.de_genes[row]
    return np.flatnonzero(~mask), de[~mask[de]][:N_DE], order[~mask[order]][:N_TOP_EXPRESSED]


def score_profile(
    pred: np.ndarray,
    obs: np.ndarray,
    control: np.ndarray,
    all_genes: np.ndarray,
    de_genes: np.ndarray,
    top_expressed: np.ndarray,
) -> dict[str, float]:
    """Metrics for one perturbation.

    pred, obs, control : (G,) predicted profile, observed profile, control mean
    all_genes          : gene indices used for the correlation over all genes
    de_genes           : indices of the perturbation's differentially expressed genes
    top_expressed      : indices of the genes with the highest control expression
    """
    d_pred = pred - control
    d_obs = obs - control
    return {
        "pearson_all": pearson(d_pred[all_genes], d_obs[all_genes]),
        "pearson_de20": pearson(d_pred[de_genes], d_obs[de_genes]),
        "mse_de20": float(np.mean((d_pred[de_genes] - d_obs[de_genes]) ** 2)),
        "l2_top1000": float(np.sqrt(np.sum((d_pred[top_expressed] - d_obs[top_expressed]) ** 2))),
    }
