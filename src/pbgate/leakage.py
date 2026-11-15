"""Checks that held-out profiles cannot influence predictions.

Three checks are provided.

``check_split``        structural: train and test are disjoint, the control is not a
                       perturbation, no gene declared unseen occurs in training.
``check_predictor``    behavioural: a prediction function is run twice, on the data and
                       on a copy whose held-out profiles are scrambled. A function that
                       reads only the control and the training perturbations returns
                       identical predictions; any difference is reported as leakage.
``check_predictions``  for prediction files: a predicted profile that reproduces the
                       observed held-out profile to numerical precision is reported.

The second check can only catch a function that reads the data object it is given. It
cannot certify a prediction file produced elsewhere; for those, only the third check
and the split-half reference of the gate apply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .data import PseudobulkData
from .predictors import Predictor
from .splits import Split, SplitError, validate_split

PredictFn = Callable[[PseudobulkData, Split], np.ndarray]


class LeakageError(RuntimeError):
    """Raised when held-out information reaches a predictor or a prediction."""


def check_split(split: Split, data: PseudobulkData) -> None:
    try:
        validate_split(split, data.conditions.tolist())
    except SplitError as error:
        raise LeakageError(f"invalid split: {error}") from error


def as_predict_fn(predictor: Predictor) -> PredictFn:
    """Wrap a :class:`Predictor`; it is fitted on the training view only."""

    def predict(data: PseudobulkData, split: Split) -> np.ndarray:
        return predictor.fit(data.train_view(split.train)).predict(list(split.test))

    return predict


def check_predictor(
    predict_fn: PredictFn, data: PseudobulkData, split: Split, seed: int = 0, atol: float = 1e-9
) -> None:
    """Raise :class:`LeakageError` if predictions depend on the held-out profiles.

    ``predict_fn(data, split)`` must return predicted profiles for ``split.test`` in
    order, shape (len(split.test), n_genes).
    """
    check_split(split, data)
    reference = np.asarray(predict_fn(data, split), dtype=float)
    scrambled = data.with_scrambled(split.test, np.random.default_rng(seed))
    repeat = np.asarray(predict_fn(scrambled, split), dtype=float)
    if reference.shape != repeat.shape:
        raise LeakageError("the prediction shape changes when held-out profiles are scrambled")
    changed = np.flatnonzero(np.abs(reference - repeat).max(axis=1) > atol)
    if changed.size:
        names = [split.test[i] for i in changed[:5]]
        raise LeakageError(
            f"predictions for {changed.size} of {len(split.test)} held-out perturbations change when "
            f"the held-out profiles are scrambled (first: {names}); the predictor reads test data"
        )
    check_predictions(reference, data.profiles(split.test), split.test)


def check_predictions(
    pred: np.ndarray, observed: np.ndarray, conditions: Sequence[str], atol: float = 1e-6
) -> None:
    """Raise :class:`LeakageError` if a prediction reproduces its observed profile."""
    copied = np.flatnonzero(np.abs(np.asarray(pred) - np.asarray(observed)).max(axis=1) <= atol)
    if copied.size:
        names = [conditions[i] for i in copied[:5]]
        raise LeakageError(
            f"{copied.size} predicted profiles are identical to the observed held-out profiles "
            f"(first: {names})"
        )
