from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pbgate.cli import main
from pbgate.evaluate import fit_predict
from pbgate.gate import normalise_condition, read_predictions, run_gate, write_predictions
from pbgate.leakage import LeakageError
from pbgate.metrics import METRICS
from pbgate.predictors import Additive, NoChange, TrainMean


def as_table(pred, conditions, data) -> pd.DataFrame:
    return pd.DataFrame(pred, index=list(conditions), columns=data.genes.tolist())


def test_gate_rejects_no_change_compared_with_itself(data, split):
    pred = as_table(fit_predict(NoChange(), data, split), split.test, data)
    result = run_gate([(pred, split)], data, n_boot=500)
    assert not result.passed
    assert not result.verdict["passes"].any()
    own = result.comparisons[result.comparisons["baseline"] == "no_change"]
    assert (own["advantage"] == 0).all() and (own["ci_low"] == 0).all() and (own["ci_high"] == 0).all()
    assert not own["beats"].any()


@pytest.mark.parametrize("baseline", [TrainMean, Additive])
def test_gate_rejects_a_baseline_submitted_as_a_model(data, split, baseline):
    pred = as_table(fit_predict(baseline(), data, split), split.test, data)
    result = run_gate([(pred, split)], data, n_boot=500)
    assert not result.passed
    own = result.comparisons[result.comparisons["baseline"] == baseline.name]
    assert np.allclose(own["advantage"], 0) and not own["beats"].any()


def test_gate_passes_a_model_that_is_close_to_the_truth_and_flags_the_noise_floor(data, split):
    rng = np.random.default_rng(0)
    truth = data.profiles(split.test)
    pred = as_table(truth + rng.normal(0, 0.005, size=truth.shape), split.test, data)
    result = run_gate([(pred, split)], data, n_boot=500)
    assert result.passed
    assert set(result.verdict["metric"]) == set(METRICS)
    assert any("noise" in warning for warning in result.warnings)


def test_gate_refuses_copied_truth_training_rows_and_missing_genes(data, split):
    truth = as_table(data.profiles(split.test), split.test, data)
    with pytest.raises(LeakageError):
        run_gate([(truth, split)], data, n_boot=100)
    train_rows = as_table(data.profiles(split.train[:3]), split.train[:3], data)
    with pytest.raises(ValueError, match="training"):
        run_gate([(train_rows, split)], data, n_boot=100)
    pred = as_table(fit_predict(TrainMean(), data, split), split.test, data)
    with pytest.raises(ValueError, match="lack"):
        run_gate([(pred.iloc[:, :100], split)], data, n_boot=100)


def test_gate_accepts_a_subset_and_several_splits(data, split):
    from pbgate.splits import make_split

    other = make_split(data.conditions.tolist(), 1)
    entries = []
    for current in (split, other):
        singles = current.test_in("unseen_single")
        pred = fit_predict(TrainMean(), data, current)
        rows = [current.test.index(c) for c in singles]
        entries.append((as_table(pred[rows], singles, data), current))
    result = run_gate(entries, data, n_boot=300)
    assert set(result.verdict["category"]) == {"unseen_single", "all"}
    assert result.verdict["n_perturbations"].max() <= len(data.singles)
    assert not result.passed


def test_prediction_files_round_trip(data, split, tmp_path):
    conditions = list(split.test[:4])
    pred = data.profiles(conditions) + 0.1
    for name in ("pred.csv", "pred.npz"):
        write_predictions(tmp_path / name, pred, conditions, data.genes.tolist())
        table = read_predictions(tmp_path / name)
        assert table.index.tolist() == conditions
        assert np.allclose(table.to_numpy(), pred, atol=1e-4)
    assert normalise_condition("KLF1+ctrl") == "KLF1"
    assert normalise_condition("ctrl+KLF1") == "KLF1"
    assert normalise_condition("MAP2K6+KLF1") == "KLF1+MAP2K6"


@pytest.mark.parametrize("suffix", ["npz", "csv"])
def test_rounding_in_prediction_files_does_not_decide_the_gate(data, split, tmp_path, suffix):
    """A baseline written to a file (32-bit floats or five decimals) must not beat itself."""
    path = tmp_path / f"additive.{suffix}"
    write_predictions(path, fit_predict(Additive(), data, split), split.test, data.genes.tolist())
    result = run_gate([(read_predictions(path), split)], data, n_boot=500)
    own = result.comparisons[result.comparisons["baseline"] == "additive"]
    assert not own["beats"].any()
    assert not result.verdict["passes"].any()


def test_cli_end_to_end(tmp_path, capsys):
    split_path, pred_path, out_path = tmp_path / "split.json", tmp_path / "pred.npz", tmp_path / "gate.json"
    assert main(["export-split", "--seed", "0", "--out", str(split_path)]) == 0
    assert (
        main(["predict", "--predictor", "no_change", "--split", str(split_path), "--out", str(pred_path)])
        == 0
    )
    code = main(
        [
            "gate",
            "--pred",
            str(pred_path),
            "--split",
            str(split_path),
            "--n-boot",
            "300",
            "--out",
            str(out_path),
        ]
    )
    assert code == 1  # the no-change baseline does not beat itself
    assert "gate: FAIL" in capsys.readouterr().out
    report = json.loads(out_path.read_text())
    assert report["passed"] is False
    assert main(["info"]) == 0
