"""Analyses built on the pseudobulk file and on baseline records.

``additivity``   how much of the double-perturbation effects the additive model explains
``ranking``      how the order of the baselines differs between metrics
``pairwise``     paired comparison of every baseline with every other baseline
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .data import PseudobulkData, genes_of
from .metrics import HIGHER_IS_BETTER, METRICS, N_DE, pearson, target_mask
from .splits import CATEGORIES
from .stats import crossed_bootstrap_means

GENE_SETS = ("all", "top1000", "de20")


# --------------------------------------------------------------------- additivity
def additivity(
    data: PseudobulkData, exclude_targets: bool = True, n_boot: int = 10_000, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Compare every double perturbation with the sum of its two single effects.

    Uses all double perturbations whose two singles are measured (no train/test split:
    the additive model has no fitted parameter). Sums of squares are taken over genes;
    with ``exclude_targets`` the two perturbed genes themselves are left out.

    Per double and gene set:
      ss_total  sum over genes of (double effect)^2
      ss_resid  sum over genes of (double effect - sum of single effects)^2
      r2        1 - ss_resid / ss_total  (share of the effect's sum of squares explained)
    For the gene sets ``all`` and ``top1000`` the expected contribution of sampling
    noise (squared standard errors of the pseudobulk means) is reported as well:
      noise_total  expected noise in ss_total  (double + control)
      noise_resid  expected noise in ss_resid  (double + both singles + control)
    """
    index = data.index
    top = data.top_expressed()
    control = data.control_mean
    rows = []
    for cond in data.doubles:
        a, b = genes_of(cond)
        if a not in index or b not in index:
            continue
        i, ia, ib = index[cond], index[a], index[b]
        d_ab = data.mean[i] - control
        d_add = (data.mean[ia] - control) + (data.mean[ib] - control)
        keep = ~target_mask(cond, data) if exclude_targets else np.ones(len(control), dtype=bool)
        de = [g for g in data.de_genes[i] if keep[g]][:N_DE]
        sets = {"all": np.flatnonzero(keep), "top1000": top[keep[top]], "de20": np.array(de)}
        row = {"condition": cond, "n_cells": int(data.n_cells[i])}
        for name, genes in sets.items():
            obs, add = d_ab[genes], d_add[genes]
            row[f"ss_total_{name}"] = float(np.sum(obs**2))
            row[f"ss_resid_{name}"] = float(np.sum((obs - add) ** 2))
            row[f"r2_{name}"] = 1.0 - row[f"ss_resid_{name}"] / row[f"ss_total_{name}"]
            row[f"pearson_{name}"] = pearson(add, obs)
            row[f"slope_{name}"] = float(np.dot(add, obs) / np.dot(add, add))
        for name in ("all", "top1000"):
            key = f"sem2_{name}"
            if key in data.extra:
                ctrl = float(data.meta[f"control_{key}"])
                noise = data.extra[key]
                row[f"noise_total_{name}"] = float(noise[i] + ctrl)
                row[f"noise_resid_{name}"] = float(noise[i] + noise[ia] + noise[ib] + ctrl)
        rows.append(row)
    table = pd.DataFrame(rows)

    # effects of all doubles on all kept genes, for the reference "mean double effect"
    effects = data.mean[data.rows(table["condition"].tolist())] - control
    mean_effect = effects.mean(axis=0)
    centred_total = {}
    for name in ("all", "top1000"):
        per_double = []
        for cond, effect in zip(table["condition"], effects, strict=True):
            keep = ~target_mask(cond, data) if exclude_targets else np.ones(len(control), dtype=bool)
            genes = np.flatnonzero(keep) if name == "all" else top[keep[top]]
            per_double.append(float(np.sum((effect[genes] - mean_effect[genes]) ** 2)))
        centred_total[name] = np.array(per_double)
        table[f"ss_centred_{name}"] = centred_total[name]

    rng = np.random.default_rng(seed)
    n = len(table)
    draws = rng.integers(0, n, size=(n_boot, n))

    def pooled(numerator: np.ndarray, denominator: np.ndarray) -> dict:
        value = 1.0 - numerator.sum() / denominator.sum()
        boot = 1.0 - numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        return {"value": float(value), "ci_low": float(low), "ci_high": float(high)}

    summary: dict = {"n_doubles": n, "exclude_targets": exclude_targets, "n_boot": n_boot, "seed": seed}
    for name in GENE_SETS:
        total, resid = table[f"ss_total_{name}"].to_numpy(), table[f"ss_resid_{name}"].to_numpy()
        r2 = table[f"r2_{name}"]
        entry = {
            "pooled_r2": pooled(resid, total),
            "per_double_r2_median": float(r2.median()),
            "per_double_r2_q25": float(r2.quantile(0.25)),
            "per_double_r2_q75": float(r2.quantile(0.75)),
            "per_double_r2_min": float(r2.min()),
            "share_of_doubles_r2_below_0.5": float((r2 < 0.5).mean()),
            "per_double_pearson_median": float(table[f"pearson_{name}"].median()),
            "per_double_slope_median": float(table[f"slope_{name}"].median()),
        }
        if name in centred_total:
            entry["pooled_r2_vs_mean_double"] = pooled(resid, centred_total[name])
        if f"noise_total_{name}" in table:
            n_total = table[f"noise_total_{name}"].to_numpy()
            n_resid = table[f"noise_resid_{name}"].to_numpy()
            entry["noise_share_of_total"] = float(n_total.sum() / total.sum())
            entry["noise_share_of_residual"] = float(n_resid.sum() / resid.sum())
            entry["pooled_r2_noise_corrected"] = pooled(resid - n_resid, total - n_total)
            # the control cancels in the spread of doubles around their mean effect
            n_double = n_total - float(data.meta[f"control_sem2_{name}"])
            entry["pooled_r2_vs_mean_double_noise_corrected"] = pooled(
                resid - n_resid, centred_total[name] - n_double
            )
        summary[name] = entry
    return table, summary


# ------------------------------------------------------------------------ ranking
def _wide(records: pd.DataFrame, category: str, predictors: list[str]) -> pd.DataFrame:
    subset = records[(records["category"] == category) & records["predictor"].isin(predictors)]
    return subset.pivot(index=["seed", "condition"], columns="predictor", values=list(METRICS))


def ranking(
    records: pd.DataFrame,
    predictors: list[str],
    metric_a: str = "pearson_all",
    metric_b: str = "pearson_de20",
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank ``predictors`` under every metric and compare two of the rankings.

    Returns (rank table, comparison table). Rank 1 is best. The comparison reports
    Kendall's tau between the orders under ``metric_a`` and ``metric_b`` and, over
    bootstrap resamples of seeds and perturbations, how often the best baseline and the
    complete order agree.
    """
    rng = np.random.default_rng(seed)
    rank_rows, compare_rows = [], []
    for category in CATEGORIES:
        wide = _wide(records, category, predictors)
        if wide.empty:
            continue
        seeds = wide.index.get_level_values("seed").to_numpy()
        names = wide.index.get_level_values("condition").to_numpy()
        oriented = {}
        for metric in METRICS:
            sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
            values = wide[metric][predictors].to_numpy()
            means = values.mean(axis=0)
            order = np.argsort(np.argsort(-sign * means)) + 1
            oriented[metric] = sign * values
            for j, predictor in enumerate(predictors):
                rank_rows.append(
                    {
                        "category": category,
                        "metric": metric,
                        "predictor": predictor,
                        "mean": float(means[j]),
                        "rank": int(order[j]),
                    }
                )
        stacked = np.column_stack([oriented[metric_a], oriented[metric_b]])
        boot = crossed_bootstrap_means(stacked, seeds, names, n_boot=n_boot, rng=rng)
        k = len(predictors)
        boot_a, boot_b = boot[:, :k], boot[:, k:]
        order_a, order_b = np.argsort(-boot_a, axis=1), np.argsort(-boot_b, axis=1)
        taus = np.array([kendalltau(x, y).statistic for x, y in zip(boot_a, boot_b, strict=True)])
        point_a, point_b = oriented[metric_a].mean(axis=0), oriented[metric_b].mean(axis=0)
        compare_rows.append(
            {
                "category": category,
                "metric_a": metric_a,
                "metric_b": metric_b,
                "best_a": predictors[int(np.argmax(point_a))],
                "best_b": predictors[int(np.argmax(point_b))],
                "order_a": " > ".join(predictors[j] for j in np.argsort(-point_a)),
                "order_b": " > ".join(predictors[j] for j in np.argsort(-point_b)),
                "kendall_tau": float(kendalltau(point_a, point_b).statistic),
                "boot_tau_mean": float(np.nanmean(taus)),
                "boot_p_same_best": float(np.mean(order_a[:, 0] == order_b[:, 0])),
                "boot_p_same_order": float(np.mean(np.all(order_a == order_b, axis=1))),
                "n_perturbations": len(np.unique(names)),
            }
        )
    return pd.DataFrame(rank_rows), pd.DataFrame(compare_rows)


# ----------------------------------------------------------------------- pairwise
def pairwise(
    records: pd.DataFrame, predictors: list[str], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """Paired advantage of predictor ``a`` over ``b`` (positive: ``a`` is better)."""
    rng = np.random.default_rng(seed)
    rows = []
    for category in CATEGORIES:
        wide = _wide(records, category, predictors)
        if wide.empty:
            continue
        seeds = wide.index.get_level_values("seed").to_numpy()
        names = wide.index.get_level_values("condition").to_numpy()
        pairs = list(combinations(predictors, 2))
        for metric in METRICS:
            sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
            diffs = np.column_stack(
                [sign * (wide[(metric, a)].to_numpy() - wide[(metric, b)].to_numpy()) for a, b in pairs]
            )
            boot = crossed_bootstrap_means(diffs, seeds, names, n_boot=n_boot, rng=rng)
            low, high = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
            for j, (a, b) in enumerate(pairs):
                verdict = "a" if low[j] > 0 else "b" if high[j] < 0 else "tie"
                rows.append(
                    {
                        "category": category,
                        "metric": metric,
                        "a": a,
                        "b": b,
                        "advantage_a_over_b": float(diffs[:, j].mean()),
                        "ci_low": float(low[j]),
                        "ci_high": float(high[j]),
                        "better": {"a": a, "b": b, "tie": "neither"}[verdict],
                    }
                )
    return pd.DataFrame(rows)
