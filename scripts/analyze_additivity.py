#!/usr/bin/env python
"""How much of the double-perturbation effects does the additive model explain?

Outputs (in --out, default results/):
  additivity_per_double.csv   sums of squares, R2, correlation and slope per double
  additivity_summary.json     pooled R2 with bootstrap intervals over doubles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbgate.analysis import GENE_SETS, additivity
from pbgate.data import load_pseudobulk

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    data = load_pseudobulk(args.data)
    table, summary = additivity(data, n_boot=args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "additivity_per_double.csv", index=False, float_format="%.6g")
    (args.out / "additivity_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name in GENE_SETS:
        entry = summary[name]
        pooled = entry["pooled_r2"]
        line = f"{name:8s} pooled R2 {pooled['value']:.3f} [{pooled['ci_low']:.3f}, {pooled['ci_high']:.3f}]"
        line += f"  median per double {entry['per_double_r2_median']:.3f}"
        if "pooled_r2_noise_corrected" in entry:
            line += f"  noise-corrected {entry['pooled_r2_noise_corrected']['value']:.3f}"
        if "pooled_r2_vs_mean_double" in entry:
            line += f"  vs mean double {entry['pooled_r2_vs_mean_double']['value']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
