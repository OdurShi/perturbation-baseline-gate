#!/usr/bin/env python
"""Optional: train GEARS on one split of this repository and write gate-ready predictions.

This script is not part of the offline reproduction. It needs the raw data
(scripts/download_data.py) and a separate environment with ``cell-gears``, ``torch``,
``torch_geometric`` and ``scanpy``; GEARS itself downloads its gene-ontology files.
Tested with cell-gears 0.1.2, torch 2.15.0, torch_geometric 2.8.1, scanpy 1.11.5,
numpy 1.26.4 and scipy 1.11.4 on Python 3.11 (GEARS 0.1.2 fails with scipy 1.17).
Run it with ``PYTHONPATH=scripts`` so that the preprocessing module can be imported.

The cells, the normalisation and the genes are those of the shipped pseudobulk file,
so the predictions can be passed to ``pbgate gate`` unchanged. To keep CPU training
feasible the cells are subsampled (``--max-cells`` per perturbation, ``--max-control``
control cells) and the number of epochs is small; both are recorded in the output.
One tenth of the training perturbations is set aside for GEARS' model selection.

Example:
  python scripts/optional_gears.py --work-dir /tmp/gears --seed 0 --epochs 3 \\
      --out /tmp/gears/gears_seed0.npz
  pbgate export-split --seed 0 --out /tmp/gears/split0.json
  pbgate gate --pred /tmp/gears/gears_seed0.npz --split /tmp/gears/split0.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from pbgate.data import genes_of, load_pseudobulk
from pbgate.splits import make_split

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_NAME = "NormanWeissman2019_filtered.h5ad"


def gears_name(condition: str) -> str:
    genes = genes_of(condition)
    return f"{genes[0]}+ctrl" if len(genes) == 1 else "+".join(genes)


def build_adata(raw: Path, data, max_cells: int, max_control: int, seed: int):
    """Per-cell log-normalised matrix on the genes of the pseudobulk file."""
    import anndata as ad
    import pandas as pd

    from preprocess_pseudobulk import TARGET_ALIASES, read_h5ad  # same reader and aliases

    counts, obs, all_genes = read_h5ad(raw)
    keep = obs["good_coverage"].to_numpy(bool) & (obs["number_of_cells"].to_numpy() == 1)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    counts, obs, totals = counts[keep], obs[keep].reset_index(drop=True), totals[keep]

    def canonical(label: str) -> str:
        if label == "control":
            return "ctrl"
        return "+".join(sorted(TARGET_ALIASES.get(g, g) for g in label.split("_")))

    labels = np.array([canonical(v) for v in obs["perturbation"].to_numpy(str)])
    known = set(data.conditions.tolist()) | {"ctrl"}
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(set(labels.tolist()) & known):
        members = np.flatnonzero(labels == label)
        limit = max_control if label == "ctrl" else max_cells
        if len(members) > limit:
            members = rng.choice(members, size=limit, replace=False)
        chosen.append(np.sort(members))
    chosen = np.sort(np.concatenate(chosen))

    columns = np.array([np.flatnonzero(all_genes == g)[0] for g in data.genes])
    matrix = counts[chosen].tocsc()[:, columns].tocsr().astype(np.float32)
    matrix.data *= np.repeat((1e4 / totals[chosen]).astype(np.float32), np.diff(matrix.indptr))
    np.log1p(matrix.data, out=matrix.data)
    cell_obs = pd.DataFrame(
        {"condition": [c if c == "ctrl" else gears_name(c) for c in labels[chosen]], "cell_type": "K562"},
        index=[f"cell{i}" for i in chosen],
    )
    var = pd.DataFrame({"gene_name": data.genes}, index=data.genes)
    return ad.AnnData(X=sp.csr_matrix(matrix), obs=cell_obs, var=var)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_cache = Path(os.environ.get("PBGATE_CACHE", REPO_ROOT / "data" / "raw"))
    parser.add_argument("--raw", type=Path, default=default_cache / RAW_NAME)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=60)
    parser.add_argument("--max-control", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from gears import GEARS, PertData

    start = time.time()
    data = load_pseudobulk()
    split = make_split(data.conditions.tolist(), args.seed)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    name = f"norman_pbgate_c{args.max_cells}"
    pert_data = PertData(str(work))
    if not (work / name / "data_pyg" / "cell_graphs.pkl").exists():
        adata = build_adata(args.raw, data, args.max_cells, args.max_control, seed=0)
        print(f"built {adata.n_obs} cells x {adata.n_vars} genes in {time.time() - start:.0f} s")
        pert_data.new_data_process(dataset_name=name, adata=adata)
    pert_data.load(data_path=str(work / name))
    available = set(pert_data.adata.obs["condition"].unique())
    print(f"data ready after {time.time() - start:.0f} s; {len(available) - 1} perturbations kept by GEARS")
    if args.prepare_only:
        return

    rng = np.random.default_rng(args.seed)
    train = [gears_name(c) for c in split.train if gears_name(c) in available]
    test = [gears_name(c) for c in split.test if gears_name(c) in available]
    val = sorted(rng.choice(train, size=max(1, len(train) // 10), replace=False).tolist())
    train = [c for c in train if c not in set(val)]
    split_path = work / f"gears_split_seed{args.seed}.pkl"
    with split_path.open("wb") as handle:
        pickle.dump({"train": train, "val": val, "test": test}, handle)
    pert_data.prepare_split(split="custom", seed=args.seed, split_dict_path=str(split_path))
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=128)

    model = GEARS(pert_data, device=args.device)
    model.model_initialize(hidden_size=args.hidden_size)
    fit_start = time.time()
    model.train(epochs=args.epochs)
    fit_seconds = time.time() - fit_start

    held_out = [c for c in split.test if gears_name(c) in available]
    predictions = model.predict([list(genes_of(c)) for c in held_out])
    pred = np.stack([predictions["_".join(genes_of(c))] for c in held_out])
    out = args.out or work / f"gears_seed{args.seed}.npz"
    np.savez_compressed(out, pred=pred.astype(np.float32), conditions=np.array(held_out), genes=data.genes)
    info = {
        "seed": args.seed,
        "epochs": args.epochs,
        "max_cells_per_perturbation": args.max_cells,
        "max_control_cells": args.max_control,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "device": args.device,
        "n_train_perturbations": len(train),
        "n_val_perturbations": len(val),
        "n_predicted": len(held_out),
        "n_held_out_in_split": len(split.test),
        "fit_seconds": round(fit_seconds, 1),
        "total_seconds": round(time.time() - start, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
