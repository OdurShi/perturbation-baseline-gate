from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pbgate.leakage import LeakageError, as_predict_fn, check_predictions, check_predictor, check_split
from pbgate.predictors import default_predictors


def test_baselines_do_not_depend_on_held_out_profiles(data, split):
    for predictor in default_predictors():
        check_predictor(as_predict_fn(predictor), data, split)


def test_oracle_that_returns_test_profiles_is_caught(data, split):
    def peeking(dataset, current_split):
        return dataset.profiles(current_split.test)

    with pytest.raises(LeakageError, match="reads test data"):
        check_predictor(peeking, data, split)


def test_mean_that_includes_test_profiles_is_caught(data, split):
    """A subtle leak: the 'training mean' is averaged over all perturbations."""

    def leaky_mean(dataset, current_split):
        return np.tile(dataset.mean.mean(axis=0), (len(current_split.test), 1))

    with pytest.raises(LeakageError):
        check_predictor(leaky_mean, data, split)


def test_peeking_at_differential_expression_ranks_is_caught(data, split):
    """A predictor that uses the held-out DE gene lists (derived from held-out cells)."""

    def uses_de(dataset, current_split):
        out = np.tile(dataset.control_mean, (len(current_split.test), 1))
        for i, row in enumerate(dataset.rows(current_split.test)):
            out[i, dataset.de_genes[row][:20]] += 1.0
        return out

    with pytest.raises(LeakageError):
        check_predictor(uses_de, data, split)


def test_copied_profiles_and_bad_splits_are_reported(data, split):
    conditions = list(split.test[:3])
    observed = data.profiles(conditions)
    with pytest.raises(LeakageError, match="identical"):
        check_predictions(observed.copy(), observed, conditions)
    check_predictions(observed + 0.01, observed, conditions)
    with pytest.raises(LeakageError, match="invalid split"):
        check_split(replace(split, train=(*split.train, split.test[0])), data)
