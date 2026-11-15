from __future__ import annotations

import numpy as np

from pbgate.evaluate import score_predictions
from pbgate.metrics import N_DE, N_TOP_EXPRESSED, gene_sets, pearson, score_profile, target_mask


def test_pearson_basics():
    x = np.array([0.0, 1.0, 2.0, 4.0])
    assert pearson(x, 3 * x + 1) == 1.0
    assert np.isclose(pearson(x, -x), -1.0)
    assert pearson(x, np.ones(4)) == 0.0  # undefined correlation is reported as 0


def test_gene_sets_exclude_the_perturbed_genes(data):
    cond = data.doubles[0]
    mask = target_mask(cond, data)
    assert mask.sum() == 2
    all_genes, de, top = gene_sets(cond, data)
    assert len(all_genes) == len(data.genes) - 2
    assert len(de) == N_DE and len(top) == N_TOP_EXPRESSED
    assert not mask[all_genes].any() and not mask[de].any() and not mask[top].any()
    # the conventional definition keeps them and uses the stored ranking as is
    all_conv, de_conv, top_conv = gene_sets(cond, data, exclude_targets=False)
    assert len(all_conv) == len(data.genes)
    assert np.array_equal(de_conv, data.de_genes[data.index[cond]][:N_DE])
    assert np.array_equal(top_conv, data.top_expressed())


def test_top_expressed_is_defined_by_control(data):
    top = data.top_expressed()
    assert data.control_mean[top].min() >= np.delete(data.control_mean, top).max()


def test_score_profile_known_values():
    control = np.zeros(6)
    obs = np.array([1.0, -1.0, 2.0, 0.0, 0.0, 0.0])
    pred = np.array([2.0, -2.0, 4.0, 0.0, 0.0, 1.0])
    idx = np.arange(6)
    scores = score_profile(pred, obs, control, idx, np.array([0, 1, 2]), np.array([0, 5]))
    assert np.isclose(scores["pearson_de20"], 1.0)
    assert np.isclose(scores["mse_de20"], (1 + 1 + 4) / 3)
    assert np.isclose(scores["l2_top1000"], np.sqrt(1 + 1))
    assert 0 < scores["pearson_all"] < 1


def test_perfect_and_no_change_predictions(data, split):
    conditions = list(split.test[:5])
    perfect = score_predictions(data.profiles(conditions), conditions, data)
    assert np.allclose(perfect["pearson_all"], 1) and np.allclose(perfect["pearson_de20"], 1)
    assert np.allclose(perfect["mse_de20"], 0) and np.allclose(perfect["l2_top1000"], 0)
    control = np.tile(data.control_mean, (5, 1))
    unchanged = score_predictions(control, conditions, data)
    assert np.allclose(unchanged["pearson_all"], 0)
    assert (unchanged["mse_de20"] > 0).all()
