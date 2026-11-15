"""Command line interface: ``pbgate info | baselines | export-split | predict | gate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_pseudobulk
from .evaluate import fit_predict, run_baselines, split_table, summarise
from .gate import MIN_ADVANTAGE, read_predictions, run_gate, write_predictions
from .leakage import LeakageError
from .predictors import BASELINE_NAMES, default_predictors
from .splits import Split, make_split


def parse_seeds(text: str) -> list[int]:
    """``"0-19"`` or ``"0,3,7"`` or a mixture."""
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


def cmd_info(args: argparse.Namespace) -> int:
    data = load_pseudobulk(args.data)
    print(f"genes                 {len(data.genes)}")
    print(f"single perturbations  {len(data.singles)}")
    print(f"double perturbations  {len(data.doubles)}")
    print(f"control cells         {data.n_control_cells}")
    print(f"perturbed cells       {int(data.n_cells.sum())}")
    for key in ("dataset", "release", "record_url", "licence", "cell_filter", "normalisation", "gene_filter"):
        print(f"{key:<21} {data.meta.get(key)}")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    data = load_pseudobulk(args.data)
    records, splits = run_baselines(data, parse_seeds(args.seeds))
    summary = summarise(records, n_boot=args.n_boot)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        records.to_csv(out / "baseline_records.csv", index=False, float_format="%.6g")
        summary.to_csv(out / "baseline_summary.csv", index=False, float_format="%.6g")
        split_table(splits).to_csv(out / "splits.csv", index=False)
    table = summary.pivot(index=["category", "predictor"], columns="metric", values="mean")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.loc[:, ["pearson_all", "pearson_de20", "mse_de20", "l2_top1000"]])
    return 0


def cmd_export_split(args: argparse.Namespace) -> int:
    data = load_pseudobulk(args.data)
    split = make_split(data.conditions.tolist(), args.seed)
    split.to_json(args.out)
    counts = f"{len(split.train)} training and {len(split.test)} held-out perturbations"
    print(f"seed {args.seed}: {counts} -> {args.out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    data = load_pseudobulk(args.data)
    split = Split.from_json(args.split)
    predictor = {p.name: p for p in default_predictors()}[args.predictor]
    pred = fit_predict(predictor, data, split)
    write_predictions(args.out, pred, split.test, data.genes.tolist())
    print(f"{args.predictor}: {pred.shape[0]} predicted profiles -> {args.out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    if len(args.pred) != len(args.split):
        print("give one --split per --pred", file=sys.stderr)
        return 2
    data = load_pseudobulk(args.data)
    entries = [(read_predictions(p), Split.from_json(s)) for p, s in zip(args.pred, args.split, strict=True)]
    try:
        result = run_gate(
            entries,
            data,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
            min_advantage=args.min_advantage,
        )
    except LeakageError as error:
        print(f"LEAKAGE: {error}", file=sys.stderr)
        return 2
    verdict = result.verdict.copy()
    verdict["passes"] = np.where(verdict["passes"], "PASS", "fail")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(verdict.to_string(index=False))
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(f"\ngate: {'PASS' if result.passed else 'FAIL'}")
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=1) + "\n")
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pbgate", description=__doc__)
    parser.add_argument("--data", default=None, help="pseudobulk .npz (default: the file shipped in data/)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="describe the shipped pseudobulk file").set_defaults(func=cmd_info)

    p = sub.add_parser("baselines", help="run the baselines over several random splits")
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument(
        "--out", default=None, help="folder for baseline_records.csv, baseline_summary.csv, splits.csv"
    )
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("export-split", help="write the train/held-out lists of one seed as JSON")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_split)

    p = sub.add_parser("predict", help="write the predictions of one baseline in the gate's input format")
    p.add_argument("--predictor", choices=BASELINE_NAMES, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("gate", help="test a prediction file against every baseline")
    p.add_argument("--pred", action="append", required=True, help="predictions (.csv or .npz); repeatable")
    p.add_argument("--split", action="append", required=True, help="split JSON of each --pred; repeatable")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
    p.add_argument(
        "--min-advantage",
        type=float,
        default=MIN_ADVANTAGE,
        help="the interval of the advantage over a baseline must lie above this value (metric units)",
    )
    p.add_argument("--out", default=None, help="write the full result as JSON")
    p.set_defaults(func=cmd_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
