"""Fit baselines on a split, score held-out perturbations, summarise over seeds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import PseudobulkData, genes_of
from .metrics import METRICS, gene_sets, score_profile
from .predictors import Predictor, default_predictors
from .splits import CATEGORIES, Split, make_split, validate_split
from .stats import mean_ci

RECORD_KEYS = ["seed", "condition", "category", "n_singles_in_train", "predictor"]


def score_predictions(
    pred: np.ndarray, conditions: Sequence[str], data: PseudobulkData, exclude_targets: bool | None = None
) -> pd.DataFrame:
    """Per-perturbation metrics of predicted profiles ``pred`` (len(conditions) x G)."""
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (len(conditions), len(data.genes)):
        raise ValueError(
            f"predictions have shape {pred.shape}, expected {(len(conditions), len(data.genes))}"
        )
    if not np.all(np.isfinite(pred)):
        raise ValueError("predictions contain NaN or infinite values")
    rows = data.rows(conditions)
    scores = [
        score_profile(pred[i], data.mean[row], data.control_mean, *gene_sets(cond, data, exclude_targets))
        for i, (cond, row) in enumerate(zip(conditions, rows, strict=True))
    ]
    table = pd.DataFrame(scores, columns=list(METRICS))
    table.insert(0, "condition", list(conditions))
    return table


def fit_predict(predictor: Predictor, data: PseudobulkData, split: Split) -> np.ndarray:
    """Fit on the training view of ``data`` and predict the held-out perturbations."""
    train = data.train_view(split.train)
    return predictor.fit(train).predict(list(split.test))


def run_split(
    data: PseudobulkData,
    split: Split,
    predictors: Iterable[Predictor] | None = None,
    exclude_targets: bool | None = None,
) -> pd.DataFrame:
    """Records (one row per held-out perturbation and predictor) for one split."""
    validate_split(split, data.conditions.tolist())
    singles = split.train_singles
    frames = []
    for predictor in predictors if predictors is not None else default_predictors():
        table = score_predictions(fit_predict(predictor, data, split), split.test, data, exclude_targets)
        table["predictor"] = predictor.name
        frames.append(table)
    records = pd.concat(frames, ignore_index=True)
    records["seed"] = split.seed
    records["category"] = records["condition"].map(split.category)
    records["n_singles_in_train"] = [sum(g in singles for g in genes_of(c)) for c in records["condition"]]
    return records[RECORD_KEYS + list(METRICS)]


def run_baselines(
    data: PseudobulkData,
    seeds: Sequence[int],
    unseen_gene_frac: float = 0.25,
    seen2_test_frac: float = 0.25,
    predictors: Sequence[Predictor] | None = None,
    exclude_targets: bool | None = None,
) -> tuple[pd.DataFrame, list[Split]]:
    """Run the baselines (default: :func:`default_predictors`) on one split per seed."""
    splits = [make_split(data.conditions.tolist(), s, unseen_gene_frac, seen2_test_frac) for s in seeds]
    records = pd.concat(
        [run_split(data, split, predictors, exclude_targets) for split in splits], ignore_index=True
    )
    return records, splits


def summarise(
    records: pd.DataFrame, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """Mean and bootstrap interval per split category, predictor and metric.

    ``mean`` is the mean over all (seed, perturbation) records of the category;
    the interval resamples seeds and perturbations (see :mod:`pbgate.stats`).
    ``seed_mean_sd`` is the standard deviation across seeds of the per-seed mean.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for category in CATEGORIES:
        for predictor, group in records[records["category"] == category].groupby("predictor", sort=False):
            for metric in METRICS:
                mean, low, high = mean_ci(
                    group[metric].to_numpy(),
                    group["seed"].to_numpy(),
                    group["condition"].to_numpy(),
                    n_boot=n_boot,
                    alpha=alpha,
                    rng=rng,
                )
                per_seed = group.groupby("seed")[metric].mean()
                rows.append(
                    {
                        "category": category,
                        "predictor": predictor,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "seed_mean_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                        "n_records": len(group),
                        "n_perturbations": group["condition"].nunique(),
                        "n_seeds": group["seed"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def split_table(splits: Sequence[Split]) -> pd.DataFrame:
    """Number of training and held-out perturbations per seed and category."""
    rows = []
    for split in splits:
        row = {
            "seed": split.seed,
            "rejected_draws": split.draw,
            "n_unseen_genes": len(split.unseen_genes),
            "n_train": len(split.train),
            "n_train_single": len(split.train_singles),
            "n_train_double": len(split.train) - len(split.train_singles),
            "n_test": len(split.test),
        }
        for category in CATEGORIES:
            row[f"n_{category}"] = len(split.test_in(category))
        rows.append(row)
    return pd.DataFrame(rows)
