from __future__ import annotations

import numpy as np
import pytest

from pbgate.evaluate import fit_predict
from pbgate.predictors import (
    Additive,
    LinearLowRank,
    NearestPerturbations,
    NoChange,
    TrainMean,
    default_predictors,
    gene_embedding,
)
from pbgate.splits import split_from_lists


@pytest.fixture()
def toy_split(toy):
    held_single, seen = "H", list("ABCDEFG")
    train = [*seen, "A+B", "C+D", "E+F"]
    test = [held_single, "A+C", "A+H", "B+G"]
    return split_from_lists(train, test)


def test_no_change_and_train_mean(toy, toy_split):
    train = toy.train_view(toy_split.train)
    assert np.allclose(NoChange().fit(train).predict(["A+C"])[0], toy.control_mean)
    assert np.allclose(TrainMean().fit(train).predict(["H"])[0], toy.profiles(toy_split.train).mean(axis=0))


def test_additive_is_exact_on_additive_data_and_falls_back(toy, toy_split):
    model = Additive().fit(toy.train_view(toy_split.train))
    pred = dict(zip(toy_split.test, model.predict(toy_split.test), strict=True))
    assert np.allclose(pred["A+C"], toy.profiles(["A+C"])[0])
    assert np.allclose(pred["B+G"], toy.profiles(["B+G"])[0])
    control = toy.control_mean
    mean_effect = (toy.profiles(list("ABCDEFG")) - control).mean(axis=0)
    assert np.allclose(pred["H"], control + mean_effect)
    assert np.allclose(pred["A+H"], toy.profiles(["A"])[0] + mean_effect)
    assert model.n_singles_available("A+H") == 1


def test_linear_offset(toy, toy_split):
    train = toy.train_view(toy_split.train)
    heavy = LinearLowRank(ridge=1e12).fit(train)  # the low-rank term vanishes
    n = np.array([1, 2])
    expected = toy.control_mean + n[:, None] * heavy.shift_
    assert np.allclose(heavy.predict(["H", "A+C"]), expected, atol=1e-6)
    # singles only: the offset is the mean training profile, as in the published model
    singles = toy.train_view(list("ABCDEFG"))
    only_singles = LinearLowRank(ridge=1e12).fit(singles)
    assert np.allclose(only_singles.predict(["H"])[0], singles.mean.mean(axis=0), atol=1e-6)
    legacy = LinearLowRank(ridge=1e12, per_gene_shift=False).fit(train)
    assert np.allclose(legacy.predict(["A+C"])[0], train.mean.mean(axis=0), atol=1e-6)


def test_gene_embedding_uses_training_rows_only(toy, toy_split):
    train = toy.train_view(toy_split.train)
    emb = gene_embedding(train, 5)
    assert emb.shape == (len(toy.genes), 5)
    scrambled = toy.with_scrambled(toy_split.test, np.random.default_rng(0)).train_view(toy_split.train)
    assert np.allclose(emb, gene_embedding(scrambled, 5))


def test_knn_returns_training_profiles(toy, toy_split):
    train = toy.train_view(toy_split.train)
    pred = NearestPerturbations(k=1).fit(train).predict(["H", "A+C"])
    for row in pred:
        assert np.isclose(np.abs(train.mean - row).max(axis=1).min(), 0.0)


def test_default_predictors_on_real_data(data, split):
    for predictor in default_predictors():
        pred = fit_predict(predictor, data, split)
        assert pred.shape == (len(split.test), len(data.genes))
        assert np.all(np.isfinite(pred))
