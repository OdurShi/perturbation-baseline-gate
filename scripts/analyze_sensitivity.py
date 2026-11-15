#!/usr/bin/env python
"""Sensitivity of the baseline results to definitions and fixed settings.

Each variant changes one thing relative to the primary analysis and reports the mean of
every metric per split category (same seeds, no intervals):

  primary                 the settings of run_baselines.py
  targets_included        metric gene sets keep the perturbed genes (conventional definition)
  first_pass              the first-pass definitions: targets included, linear model without
                          the per-gene shift, no minimum number of perturbations per category
  linear_no_gene_shift    linear model with the mean training profile as offset
  linear_rank<K>          linear model with K principal components (primary: 10)
  linear_ridge<L>         linear model with ridge penalty L (primary: 0.1)
  knn_k<k>                nearest-perturbation model with k neighbours (primary: 5)

Also writes the additivity summary with the perturbed genes included. With
--all-cells-data (a pseudobulk file built by preprocess_pseudobulk.py --keep-all-cells,
which needs the raw data) the primary analysis is repeated without the cell filter.

Outputs (in --out, default results/): sensitivity.csv, sensitivity_additivity.json and,
with --all-cells-data, sensitivity_all_cells.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pbgate.analysis import GENE_SETS, additivity
from pbgate.cli import parse_seeds
from pbgate.data import load_pseudobulk
from pbgate.evaluate import run_split
from pbgate.metrics import METRICS
from pbgate.predictors import Additive, LinearLowRank, NearestPerturbations, NoChange, TrainMean
from pbgate.splits import make_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def mean_table(data, seeds, predictors=None, exclude_targets=None, min_per_category=5) -> pd.DataFrame:
    splits = [make_split(data.conditions.tolist(), s, min_per_category=min_per_category) for s in seeds]
    records = pd.concat([run_split(data, split, predictors, exclude_targets) for split in splits])
    table = records.groupby(["category", "predictor"])[list(METRICS)].mean().reset_index()
    return table.melt(id_vars=["category", "predictor"], var_name="metric", value_name="mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None)
    parser.add_argument("--all-cells-data", default=None)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    data = load_pseudobulk(args.data)
    seeds = parse_seeds(args.seeds)

    first_pass = [
        NoChange(),
        TrainMean(),
        Additive(),
        LinearLowRank(per_gene_shift=False),
        NearestPerturbations(),
    ]
    variants = {
        "primary": mean_table(data, seeds),
        "targets_included": mean_table(data, seeds, exclude_targets=False),
        "first_pass": mean_table(data, seeds, first_pass, exclude_targets=False, min_per_category=0),
        "linear_no_gene_shift": mean_table(data, seeds, [LinearLowRank(per_gene_shift=False)]),
    }
    for rank in (2, 5, 20, 40):
        variants[f"linear_rank{rank}"] = mean_table(data, seeds, [LinearLowRank(rank=rank)])
    for ridge in (10.0, 1000.0, 100000.0):
        variants[f"linear_ridge{ridge:g}"] = mean_table(data, seeds, [LinearLowRank(ridge=ridge)])
    for k in (1, 3, 10, 20):
        variants[f"knn_k{k}"] = mean_table(data, seeds, [NearestPerturbations(k=k)])
    args.out.mkdir(parents=True, exist_ok=True)
    columns = ["variant", "category", "predictor", "metric", "mean"]
    if args.all_cells_data:
        both = {
            "primary": variants["primary"],
            "all_cells": mean_table(load_pseudobulk(args.all_cells_data), seeds),
        }
        table = pd.concat([t.assign(variant=name) for name, t in both.items()], ignore_index=True)
        table[columns].to_csv(args.out / "sensitivity_all_cells.csv", index=False, float_format="%.6g")

    table = pd.concat([t.assign(variant=name) for name, t in variants.items()], ignore_index=True)
    table = table[columns]
    table.to_csv(args.out / "sensitivity.csv", index=False, float_format="%.6g")

    _, with_targets = additivity(data, exclude_targets=False)
    (args.out / "sensitivity_additivity.json").write_text(json.dumps(with_targets, indent=1) + "\n")

    wide = table[table["metric"] == "pearson_all"].pivot(
        index=["variant", "predictor"], columns="category", values="mean"
    )
    with pd.option_context(
        "display.width", 200, "display.float_format", "{:.3f}".format, "display.max_rows", 200
    ):
        print(wide)
    for name in GENE_SETS:
        value = with_targets[name]["pooled_r2"]["value"]
        print(f"additivity with perturbed genes included, {name}: pooled R2 {value:.3f}")


if __name__ == "__main__":
    main()
