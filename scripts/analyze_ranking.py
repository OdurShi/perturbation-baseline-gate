#!/usr/bin/env python
"""How does the order of the baselines change between metrics?

Reads results/baseline_records.csv (written by run_baselines.py).
Outputs (in --out, default results/):
  ranking_ranks.csv        mean and rank of each baseline per split category and metric
  ranking_comparison.csv   all-gene Pearson order versus the order under each other metric
  de_gene_frequency.csv    how many perturbations share each gene in their top-20 DE set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pbgate.analysis import ranking
from pbgate.data import load_pseudobulk
from pbgate.metrics import gene_sets

REPO_ROOT = Path(__file__).resolve().parents[1]
RANKED = ["train_mean", "additive", "linear", "knn"]  # no_change has no defined correlation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None)
    parser.add_argument("--records", type=Path, default=REPO_ROOT / "results" / "baseline_records.csv")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.records.parent

    records = pd.read_csv(args.records)
    comparisons = []
    ranks = None
    for other in ("pearson_de20", "mse_de20", "l2_top1000"):
        ranks, comparison = ranking(records, RANKED, "pearson_all", other, n_boot=args.n_boot)
        comparisons.append(comparison)
    comparison = pd.concat(comparisons, ignore_index=True)
    # overlap of the top-20 DE sets: explains why a constant prediction correlates on them
    data = load_pseudobulk(args.data)
    members = [gene_sets(cond, data)[1] for cond in data.conditions.tolist()]
    counts = np.bincount(np.concatenate(members), minlength=len(data.genes))
    used = np.flatnonzero(counts)
    frequency = pd.DataFrame({"gene": data.genes[used], "n_perturbations": counts[used]})
    frequency["share_of_perturbations"] = frequency["n_perturbations"] / len(members)
    frequency = frequency.sort_values(["n_perturbations", "gene"], ascending=[False, True])
    frequency.to_csv(out / "de_gene_frequency.csv", index=False, float_format="%.4g")
    print(
        f"{len(frequency)} distinct genes fill the {sum(map(len, members))} top-20 DE slots; most frequent:"
    )
    print(frequency.head(5).to_string(index=False))

    ranks.to_csv(out / "ranking_ranks.csv", index=False, float_format="%.6g")
    comparison.to_csv(out / "ranking_comparison.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(ranks.pivot(index=["category", "predictor"], columns="metric", values="rank"))
        print(
            comparison[
                [
                    "category",
                    "metric_b",
                    "order_a",
                    "order_b",
                    "kendall_tau",
                    "boot_p_same_best",
                    "boot_p_same_order",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
