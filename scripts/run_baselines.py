#!/usr/bin/env python
"""Run the baselines on random splits and write records, summary and pairwise tests.

Outputs (in --out, default results/):
  splits.csv              number of training and held-out perturbations per seed
  baseline_records.csv    one row per seed, held-out perturbation and baseline
  baseline_summary.csv    mean and 95% interval per split category, baseline and metric
  baseline_pairwise.csv   paired comparison of every baseline with every other one
  splithalf_summary.csv   split-half replicate reference on the same held-out records
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pbgate.analysis import pairwise
from pbgate.cli import parse_seeds
from pbgate.data import load_pseudobulk
from pbgate.evaluate import run_baselines, split_table, summarise
from pbgate.metrics import METRICS
from pbgate.predictors import BASELINE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    start = time.time()
    data = load_pseudobulk(args.data)
    records, splits = run_baselines(data, parse_seeds(args.seeds))
    summary = summarise(records, n_boot=args.n_boot)
    pairs = pairwise(records, list(BASELINE_NAMES), n_boot=args.n_boot)

    # split-half replicate reference, attached to the same (seed, perturbation) records
    reference = records[records["predictor"] == BASELINE_NAMES[0]].copy()
    rows = data.rows(reference["condition"].tolist())
    for metric in METRICS:
        reference[metric] = data.extra[f"splithalf_{metric}"][rows]
    reference["predictor"] = "splithalf_replicate"
    reference_summary = summarise(reference, n_boot=args.n_boot)

    args.out.mkdir(parents=True, exist_ok=True)
    split_table(splits).to_csv(args.out / "splits.csv", index=False)
    records.to_csv(args.out / "baseline_records.csv", index=False, float_format="%.6g")
    summary.to_csv(args.out / "baseline_summary.csv", index=False, float_format="%.6g")
    pairs.to_csv(args.out / "baseline_pairwise.csv", index=False, float_format="%.6g")
    reference_summary.to_csv(args.out / "splithalf_summary.csv", index=False, float_format="%.6g")

    table = summary.pivot(index=["category", "predictor"], columns="metric", values="mean")[list(METRICS)]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table)
    print(f"{len(records)} records, {len(splits)} seeds, {time.time() - start:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
