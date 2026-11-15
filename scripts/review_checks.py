#!/usr/bin/env python
"""Numbers behind the review of the first pass (docs/notes.md, section 3).

Output (in --out, default results/): review_checks.json with
  cells            cells in the source file, kept cells, share removed by the cell filter
  control_noise    summed squared standard error of the control mean compared with the
                   size of the perturbation effects, and the correlation that the shared
                   control noise can induce between a mean-effect prediction and an
                   observed effect
  targets          how often the perturbed genes are among the 20 top-ranked DE genes
                   (conventional ranking) and their share of the squared effect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pbgate.data import genes_of, load_pseudobulk
from pbgate.metrics import N_DE, target_mask

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    data = load_pseudobulk(args.data)
    effects = data.mean - data.control_mean

    sum_sq = np.sum(effects**2, axis=1)
    mean_effect_sq = float(np.sum(effects.mean(axis=0) ** 2))
    control_sem2 = float(data.meta["control_sem2_all"])
    control_noise = {
        "control_sem2_all": control_sem2,
        "median_sum_sq_effect": float(np.median(sum_sq)),
        "min_sum_sq_effect": float(sum_sq.min()),
        "sum_sq_mean_effect": mean_effect_sq,
        "induced_correlation": control_sem2 / float(np.sqrt(mean_effect_sq * np.median(sum_sq))),
    }

    targets = {}
    for label, size in (("single", 1), ("double", 2)):
        in_top, share_all, share_de = [], [], []
        for i, cond in enumerate(data.conditions.tolist()):
            if len(genes_of(cond)) != size:
                continue
            mask = target_mask(cond, data)
            de = data.de_genes[i][:N_DE]
            in_top.append(int(mask[de].sum()))
            share_all.append(float(np.sum(effects[i][mask] ** 2) / sum_sq[i]))
            share_de.append(float(np.sum(effects[i][de][mask[de]] ** 2) / np.sum(effects[i][de] ** 2)))
        targets[label] = {
            "n_perturbations": len(in_top),
            "share_with_target_in_top20_de": float(np.mean(np.array(in_top) > 0)),
            "mean_targets_in_top20_de": float(np.mean(in_top)),
            "target_share_of_sum_sq_effect_all_genes_median": float(np.median(share_all)),
            "target_share_of_sum_sq_effect_all_genes_max": float(np.max(share_all)),
            "target_share_of_sum_sq_effect_top20_de_mean": float(np.mean(share_de)),
        }

    meta = data.meta
    result = {
        "cells": {
            "source": meta["raw_cells"],
            "kept": meta["cells_after_filter"],
            "share_removed": 1 - meta["cells_after_filter"] / meta["raw_cells"],
        },
        "control_noise": control_noise,
        "targets": targets,
        "on_target": {
            "singles_checked": meta["on_target_singles_checked"],
            "target_higher_than_control": meta["on_target_singles_up"],
            "target_in_top20_de": meta["on_target_singles_top20"],
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_checks.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
