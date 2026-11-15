from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pbgate.analysis import additivity, pairwise, ranking
from pbgate.evaluate import run_split, summarise
from pbgate.metrics import METRICS
from pbgate.splits import make_split
from pbgate.stats import crossed_bootstrap_means, mean_ci

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def test_bootstrap_is_reproducible_and_covers_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 1.0, size=200)
    seeds = np.repeat(np.arange(4), 50)
    names = np.tile(np.arange(50), 4)
    first = mean_ci(values, seeds, names, n_boot=2000, rng=np.random.default_rng(1))
    second = mean_ci(values, seeds, names, n_boot=2000, rng=np.random.default_rng(1))
    assert first == second
    assert first[1] < first[0] < first[2]
    zero = crossed_bootstrap_means(np.zeros(200), seeds, names, n_boot=200)
    assert np.all(zero == 0)


def test_paired_columns_share_resampling_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    names = np.arange(60)
    boot = crossed_bootstrap_means(np.column_stack([x, x + 1.0]), np.zeros(60), names, n_boot=500)
    assert np.allclose(boot[:, 1] - boot[:, 0], 1.0)


def test_additivity_is_one_on_exactly_additive_data(toy):
    table, summary = additivity(toy, n_boot=200)
    assert np.allclose(table["r2_all"], 1.0) and np.allclose(table["slope_all"], 1.0)
    assert np.isclose(summary["all"]["pooled_r2"]["value"], 1.0)


def test_shipped_additivity_summary_matches_the_code(data):
    shipped = json.loads((RESULTS / "additivity_summary.json").read_text())
    _, summary = additivity(data, n_boot=50)
    for gene_set in ("all", "top1000", "de20"):
        assert np.isclose(summary[gene_set]["pooled_r2"]["value"], shipped[gene_set]["pooled_r2"]["value"])
    assert summary["n_doubles"] == shipped["n_doubles"] == 131


def test_shipped_records_match_the_code_for_one_seed(data):
    shipped = pd.read_csv(RESULTS / "baseline_records.csv")
    seed = 7
    fresh = run_split(data, make_split(data.conditions.tolist(), seed))
    old = shipped[shipped["seed"] == seed].reset_index(drop=True)
    assert old[["condition", "category", "predictor"]].equals(fresh[["condition", "category", "predictor"]])
    assert np.allclose(old[list(METRICS)], fresh[list(METRICS)], rtol=1e-4, atol=1e-6)


def test_summary_ranking_and_pairwise_shapes(data):
    records = pd.concat([run_split(data, make_split(data.conditions.tolist(), s)) for s in (0, 1)])
    summary = summarise(records, n_boot=200)
    assert len(summary) == 4 * 5 * len(METRICS)
    assert (summary["ci_low"] <= summary["mean"]).all() and (summary["mean"] <= summary["ci_high"]).all()
    predictors = ["train_mean", "additive", "linear", "knn"]
    ranks, comparison = ranking(records, predictors, n_boot=200)
    assert sorted(ranks[(ranks.category == "combo_seen2") & (ranks.metric == "pearson_all")]["rank"]) == [
        1,
        2,
        3,
        4,
    ]
    assert comparison["kendall_tau"].between(-1, 1).all()
    pairs = pairwise(records, predictors, n_boot=200)
    row = pairs[
        (pairs.category == "combo_seen2")
        & (pairs.metric == "mse_de20")
        & (pairs.a == "train_mean")
        & (pairs.b == "additive")
    ].iloc[0]
    assert row["better"] == "additive"
