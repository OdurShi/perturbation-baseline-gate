#!/usr/bin/env python
"""Demonstrate the gate: every baseline is treated as a candidate model.

Each baseline's held-out predictions (all seeds) are passed through the gate against
the remaining four baselines. Three synthetic candidates bracket the behaviour:

  observed + noise (sd 0.05)    the observed profiles plus independent Gaussian noise per
                                gene, a stand-in for a very accurate model whose error is
                                about as large as the sampling noise of the data
  observed + noise (sd 0.005)   an implausibly accurate candidate; it passes everywhere
                                and triggers the sampling-noise warning of the gate
  no_change                     tested against all five baselines including itself; it
                                must fail

Outputs (in --out, default results/):
  gate_demo.csv       verdict per candidate, split category and metric
  gate_demo.json      full comparison tables and warnings of the synthetic candidates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pbgate.cli import parse_seeds
from pbgate.data import load_pseudobulk
from pbgate.evaluate import fit_predict
from pbgate.gate import run_gate
from pbgate.predictors import default_predictors
from pbgate.splits import make_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    data = load_pseudobulk(args.data)
    genes = data.genes.tolist()
    splits = [make_split(data.conditions.tolist(), s) for s in parse_seeds(args.seeds)]

    def table(pred: np.ndarray, split) -> pd.DataFrame:
        return pd.DataFrame(pred, index=list(split.test), columns=genes)

    verdicts, full = [], {}
    for candidate in default_predictors():
        entries = [(table(fit_predict(candidate, data, s), s), s) for s in splits]
        others = [p for p in default_predictors() if p.name != candidate.name]
        result = run_gate(entries, data, predictors=others, n_boot=args.n_boot)
        verdicts.append(result.verdict.assign(candidate=candidate.name, against="the other four baselines"))

    rng = np.random.default_rng(0)
    for sd in (0.05, 0.005):
        noisy = [
            (table(data.profiles(s.test) + rng.normal(0, sd, (len(s.test), len(genes))), s), s)
            for s in splits
        ]
        result = run_gate(noisy, data, n_boot=args.n_boot)
        name = f"observed + noise (sd {sd:g})"
        verdicts.append(result.verdict.assign(candidate=name, against="all five baselines"))
        full[f"observed_plus_noise_sd{sd:g}"] = result.to_dict()

    unchanged = [(table(np.tile(data.control_mean, (len(s.test), 1)), s), s) for s in splits]
    result = run_gate(unchanged, data, n_boot=args.n_boot)
    verdicts.append(result.verdict.assign(candidate="no_change", against="all five baselines"))
    full["no_change_against_itself"] = result.to_dict()

    out = pd.concat(verdicts, ignore_index=True)
    columns = ["candidate", "against", *[c for c in out.columns if c not in ("candidate", "against")]]
    args.out.mkdir(parents=True, exist_ok=True)
    out[columns].to_csv(args.out / "gate_demo.csv", index=False, float_format="%.6g")
    (args.out / "gate_demo.json").write_text(json.dumps(full, indent=1) + "\n")
    passed = out[out["category"] != "all"].pivot_table(
        index="candidate", columns="category", values="passes", aggfunc="sum"
    )
    print("number of metrics (of 4) on which the candidate beats every baseline it is tested against")
    print(passed.astype(int))


if __name__ == "__main__":
    main()
