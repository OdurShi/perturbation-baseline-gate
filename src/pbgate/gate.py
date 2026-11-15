"""The baseline gate: does a model beat every simple baseline on held-out perturbations?

For each metric the model is compared with each baseline on the same held-out
perturbations of the same split (paired). The *advantage* is oriented so that positive
means the model is better (model - baseline for correlations, baseline - model for
errors). A comparison is won when the bootstrap interval of the mean advantage lies
entirely above zero, more precisely above ``MIN_ADVANTAGE``, a margin far below any
difference of interest that keeps rounding in prediction files from deciding a
comparison. A metric passes in a split category when the model wins against every
baseline; the gate passes when every metric passes in every category for which
predictions were supplied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PseudobulkData, condition_name
from .evaluate import fit_predict, score_predictions
from .leakage import check_predictions, check_split
from .metrics import HIGHER_IS_BETTER, METRICS
from .predictors import Predictor, default_predictors
from .splits import CATEGORIES, Split
from .stats import crossed_bootstrap_means

CONTROL_NAMES = {"ctrl", "control"}
MIN_ADVANTAGE = 1e-4  # in units of the metric


def normalise_condition(name: str) -> str:
    """``"B+A"``, ``"A_B"`` free forms such as ``"A+ctrl"`` -> canonical ``"A+B"`` / ``"A"``."""
    genes = [g for g in str(name).strip().split("+") if g and g.lower() not in CONTROL_NAMES]
    return condition_name(genes)


def read_predictions(path: str | Path) -> pd.DataFrame:
    """Read predicted profiles: rows are perturbations, columns are gene symbols.

    ``.csv``/``.csv.gz``: first column holds the perturbation names, header the genes.
    ``.npz``: arrays ``conditions`` (n,), ``genes`` (G,), ``pred`` (n, G).
    Values are mean log-normalised expression (not changes from control).
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            table = pd.DataFrame(
                npz["pred"], index=npz["conditions"].astype(str), columns=npz["genes"].astype(str)
            )
    else:
        table = pd.read_csv(path, index_col=0)
    table.index = [normalise_condition(c) for c in table.index]
    if table.index.has_duplicates:
        raise ValueError("the prediction file lists a perturbation more than once")
    return table.astype(float)


def write_predictions(
    path: str | Path, pred: np.ndarray, conditions: Sequence[str], genes: Sequence[str]
) -> None:
    path = Path(path)
    if path.suffix == ".npz":
        np.savez_compressed(
            path,
            pred=np.asarray(pred, dtype=np.float32),
            conditions=np.array(conditions),
            genes=np.array(genes),
        )
    else:
        pd.DataFrame(pred, index=pd.Index(conditions, name="condition"), columns=list(genes)).to_csv(
            path, float_format="%.5f"
        )


@dataclass
class GateResult:
    comparisons: pd.DataFrame  # one row per category, metric and baseline
    verdict: pd.DataFrame  # one row per category and metric
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        rows = self.verdict[self.verdict["category"] != "all"]
        return bool(len(rows) and rows["passes"].all())

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "verdict": self.verdict.to_dict(orient="records"),
            "comparisons": self.comparisons.to_dict(orient="records"),
            "warnings": self.warnings,
        }


def _align(pred: pd.DataFrame, data: PseudobulkData, split: Split) -> tuple[np.ndarray, list[str]]:
    train, test = set(split.train), set(split.test)
    in_train = [c for c in pred.index if c in train]
    if in_train:
        raise ValueError(f"predictions supplied for training perturbations: {in_train[:5]}")
    unknown = [c for c in pred.index if c not in test]
    if unknown:
        raise ValueError(f"predictions supplied for perturbations that are not held out: {unknown[:5]}")
    missing = [g for g in data.genes.tolist() if g not in pred.columns]
    if missing:
        raise ValueError(
            f"predictions lack {len(missing)} of {len(data.genes)} genes (first: {missing[:5]}); "
            "a prediction is needed for every gene of the pseudobulk file"
        )
    conditions = [c for c in split.test if c in set(pred.index)]
    if not conditions:
        raise ValueError("no prediction matches a held-out perturbation of the split")
    return pred.loc[conditions, data.genes.tolist()].to_numpy(dtype=float), conditions


def run_gate(
    entries: Sequence[tuple[pd.DataFrame, Split]],
    data: PseudobulkData,
    predictors: Sequence[Predictor] | None = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    model_name: str = "model",
    min_advantage: float = MIN_ADVANTAGE,
) -> GateResult:
    """Compare model predictions with the baselines fitted on the same splits.

    entries : (predictions, split) pairs. ``predictions`` is a table as returned by
              :func:`read_predictions`; it may cover a subset of ``split.test``.
    """
    frames, warnings = [], []
    for entry_id, (pred, split) in enumerate(entries):
        check_split(split, data)
        values, conditions = _align(pred, data, split)
        check_predictions(values, data.profiles(conditions), conditions)
        table = score_predictions(values, conditions, data)
        table["predictor"] = model_name
        tables = [table]
        for baseline in predictors if predictors is not None else default_predictors():
            full = fit_predict(baseline, data, split)
            keep = [split.test.index(c) for c in conditions]
            base = score_predictions(full[keep], conditions, data)
            base["predictor"] = baseline.name
            tables.append(base)
        frame = pd.concat(tables, ignore_index=True)
        frame["entry"] = entry_id
        frame["category"] = frame["condition"].map(split.category)
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    baselines = [p for p in records["predictor"].unique() if p != model_name]

    rng = np.random.default_rng(seed)
    comparisons, verdicts = [], []
    present = [c for c in CATEGORIES if (records["category"] == c).any()]
    for category in [*present, "all"]:
        subset = records if category == "all" else records[records["category"] == category]
        wide = subset.pivot(index=["entry", "condition"], columns="predictor", values=list(METRICS))
        entry_ids = wide.index.get_level_values("entry").to_numpy()
        names = wide.index.get_level_values("condition").to_numpy()
        n_pert = len(np.unique(names))
        if n_pert < 5:
            warnings.append(f"{category}: only {n_pert} held-out perturbations; intervals are unreliable")
        floor = _noise_floor(data, names)
        for metric in METRICS:
            sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
            model = wide[(metric, model_name)].to_numpy()
            advantage = np.column_stack([sign * (model - wide[(metric, b)].to_numpy()) for b in baselines])
            boot = crossed_bootstrap_means(advantage, entry_ids, names, n_boot=n_boot, rng=rng)
            low, high = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
            wins = []
            for j, baseline in enumerate(baselines):
                beats = bool(low[j] > min_advantage)
                wins.append(beats)
                comparisons.append(
                    {
                        "category": category,
                        "metric": metric,
                        "baseline": baseline,
                        "n_perturbations": n_pert,
                        "model_mean": float(model.mean()),
                        "baseline_mean": float(wide[(metric, baseline)].mean()),
                        "advantage": float(advantage[:, j].mean()),
                        "ci_low": float(low[j]),
                        "ci_high": float(high[j]),
                        "beats": beats,
                    }
                )
            base_means = np.array([wide[(metric, b)].mean() for b in baselines])
            strongest = int(np.argmax(sign * base_means))
            verdicts.append(
                {
                    "category": category,
                    "metric": metric,
                    "n_perturbations": n_pert,
                    "model_mean": float(model.mean()),
                    "strongest_baseline": baselines[strongest],
                    "strongest_baseline_mean": float(base_means[strongest]),
                    "min_ci_low": float(np.min(low)),
                    "baselines_beaten": int(np.sum(wins)),
                    "n_baselines": len(baselines),
                    "passes": bool(all(wins)),
                }
            )
            if metric == "l2_top1000" and floor is not None and np.mean(model**2) < floor:
                warnings.append(
                    f"{category}: mean squared L2 error {np.mean(model**2):.3f} is below the sampling-noise "
                    f"floor of the observed profiles ({floor:.3f}); a predictor that has not seen the "
                    "held-out cells cannot be expected to achieve this, check for leakage"
                )
    return GateResult(pd.DataFrame(comparisons), pd.DataFrame(verdicts), warnings)


def _noise_floor(data: PseudobulkData, conditions: np.ndarray) -> float | None:
    """Mean expected squared L2 error (1,000 most expressed genes) of a perfect prediction.

    The observed pseudobulk of a perturbation is a mean over a finite number of cells.
    Its squared standard errors, summed over the genes, are the expected squared error
    of the true mean profile, and a lower bound in expectation for any prediction that
    is independent of the held-out cells.
    """
    if "sem2_top1000" not in data.extra:
        return None
    return float(data.extra["sem2_top1000"][data.rows(conditions.tolist())].mean())
