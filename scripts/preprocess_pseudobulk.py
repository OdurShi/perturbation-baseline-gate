#!/usr/bin/env python
"""Build condition-level pseudobulk profiles from the raw Norman et al. 2019 counts.

Input : NormanWeissman2019_filtered.h5ad (see scripts/download_data.py)
Output: data/norman2019_pseudobulk.npz, data/norman2019_conditions.csv

Steps (every number that results is written to the ``meta`` field of the output):
 1. keep cells flagged ``good_coverage`` with ``number_of_cells == 1`` (one cell per
    droplet with a confidently assigned guide pair, as in the source metadata);
 2. keep perturbations with at least ``--min-cells`` cells; the control is the set of
    cells labelled ``control`` (non-targeting guide pairs);
 3. normalise each cell to 10,000 total counts (total over all genes), then log1p;
 4. keep genes detected in at least ``--min-detection`` of the kept cells, plus the
    perturbation target genes that are measured;
 5. average log-normalised expression per perturbation and for control cells;
 6. rank genes per perturbation by |Welch t| against control cells and store the top
    ``--n-de`` indices (the metrics use the first 20);
 7. split the cells of every perturbation into two random halves and score one half
    as a "prediction" of the other (split-half replicate reference);
 8. store the summed squared standard error of the mean per perturbation (sampling
    noise of the pseudobulk), over all genes and over the 1,000 most expressed genes
    (the perturbed genes themselves excluded, as in the metrics).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pbgate.data import PseudobulkData, condition_name, save_pseudobulk, sha256_of
from pbgate.metrics import EXCLUDE_TARGETS, METRICS, N_TOP_EXPRESSED, gene_sets, score_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_NAME = "NormanWeissman2019_filtered.h5ad"
SOURCE = {
    "dataset": "Norman et al. 2019, Perturb-seq, K562, CRISPRa",
    "release": "scPerturb, Zenodo record 13350497, version 1.4",
    "record_url": "https://zenodo.org/records/13350497",
    "file_url": f"https://zenodo.org/api/records/13350497/files/{RAW_NAME}/content",
    "licence": "CC BY 4.0",
}
# Perturbation names in the source that are outdated symbols of measured genes.
# Checked against HGNC (https://rest.genenames.org) and the Ensembl IDs in the file:
#   C19orf26 -> CBARP, C3orf72 -> FOXL2NB, KIAA1804 -> MAP3K21.
TARGET_ALIASES = {"C19orf26": "CBARP", "C3orf72": "FOXL2NB", "KIAA1804": "MAP3K21"}
# MAP3K21 (ENSG00000143674) is listed in the count matrix under the clone-based name
# RP5-862P8.2; the gene axis is relabelled so that the target can be found.
GENE_RENAMES = {"RP5-862P8.2": "MAP3K21"}


def _strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]])


def _column(group: h5py.Group, name: str) -> np.ndarray:
    item = group[name]
    if isinstance(item, h5py.Group):  # categorical
        return _strings(item["categories"])[item["codes"][:]]
    return item[:]


def read_h5ad(path: Path) -> tuple[sp.csr_matrix, pd.DataFrame, np.ndarray]:
    """Read counts (cells x genes, CSR), the needed obs columns, and gene symbols."""
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        shape = tuple(int(v) for v in x.attrs["shape"])
        kind = x.attrs["encoding-type"]
        parts = (x["data"][:], x["indices"][:], x["indptr"][:])
        matrix = (
            sp.csc_matrix(parts, shape=shape) if kind == "csc_matrix" else sp.csr_matrix(parts, shape=shape)
        )
        obs = pd.DataFrame(
            {
                k: _column(handle["obs"], k)
                for k in ("perturbation", "nperts", "good_coverage", "number_of_cells")
            }
        )
        genes = _strings(handle["var"]["_index"])
        ensembl = _strings(handle["var"]["ensemble_id"])
    if ensembl[genes == "RP5-862P8.2"].tolist() != ["ENSG00000143674"]:
        raise SystemExit("unexpected gene annotation: RP5-862P8.2 is not ENSG00000143674")
    genes = np.array([GENE_RENAMES.get(g, g) for g in genes])
    return matrix.tocsr(), obs, genes


def group_sums(indicator: sp.csr_matrix, matrix: sp.csr_matrix) -> np.ndarray:
    return np.asarray((indicator @ matrix).todense())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_cache = Path(os.environ.get("PBGATE_CACHE", REPO_ROOT / "data" / "raw"))
    parser.add_argument("--raw", type=Path, default=default_cache / RAW_NAME)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--min-detection", type=float, default=0.10)
    parser.add_argument("--n-de", type=int, default=50)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="seed of the split-half assignment")
    parser.add_argument(
        "--keep-all-cells", action="store_true", help="skip step 1 (sensitivity analysis only)"
    )
    args = parser.parse_args()

    start = time.time()
    counts, obs, all_genes = read_h5ad(args.raw)
    n_cells_raw, n_genes_raw = counts.shape
    print(f"read {n_cells_raw} cells x {n_genes_raw} genes in {time.time() - start:.0f} s")
    if np.any(counts.data % 1 != 0):
        raise SystemExit("the matrix does not hold integer counts")

    # 1. cells --------------------------------------------------------------------
    keep = np.ones(n_cells_raw, dtype=bool)
    if not args.keep_all_cells:
        keep = obs["good_coverage"].to_numpy(bool) & (obs["number_of_cells"].to_numpy() == 1)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    counts, obs, totals = counts[keep], obs[keep].reset_index(drop=True), totals[keep]

    # 2. conditions ---------------------------------------------------------------
    labels = obs["perturbation"].to_numpy(str)
    sizes = pd.Series(labels).value_counts()
    if "control" not in sizes:
        raise SystemExit("no control cells found")
    raw_names = sorted(n for n in sizes.index if n != "control" and sizes[n] >= args.min_cells)
    dropped = sorted(n for n in sizes.index if n != "control" and sizes[n] < args.min_cells)
    gene_names = set(all_genes.tolist())

    def canonical(raw: str) -> str:
        return condition_name([TARGET_ALIASES.get(g, g) for g in raw.split("_")])

    names = {raw: canonical(raw) for raw in raw_names}
    targets = sorted({g for name in names.values() for g in name.split("+")})
    missing_targets = [g for g in targets if g not in gene_names]

    # 3.-4. normalise, choose genes ----------------------------------------------
    detection = np.bincount(counts.indices, minlength=n_genes_raw) / counts.shape[0]
    gene_keep = detection >= args.min_detection
    n_by_detection = int(gene_keep.sum())
    gene_keep |= np.isin(all_genes, targets)
    n_targets_added = int(gene_keep.sum()) - n_by_detection
    lognorm = counts.tocsc()[:, gene_keep].tocsr().astype(np.float64)
    del counts
    lognorm.data *= np.repeat(args.target_sum / totals, np.diff(lognorm.indptr))
    np.log1p(lognorm.data, out=lognorm.data)
    genes = all_genes[gene_keep]
    squared = lognorm.copy()
    squared.data **= 2

    # 5. pseudobulk ---------------------------------------------------------------
    order = ["control", *sorted(raw_names, key=lambda r: names[r])]
    code = {name: i for i, name in enumerate(order)}
    rows = np.array([code.get(label, -1) for label in labels])
    cells = np.flatnonzero(rows >= 0)
    indicator = sp.csr_matrix(
        (np.ones(len(cells)), (rows[cells], cells)), shape=(len(order), lognorm.shape[0])
    )
    n = np.asarray(indicator.sum(axis=1)).ravel()
    s1 = group_sums(indicator, lognorm)
    s2 = group_sums(indicator, squared)
    mean = s1 / n[:, None]
    var = np.maximum(s2 - n[:, None] * mean**2, 0.0) / (n[:, None] - 1)
    sem2 = var / n[:, None]

    # 6. differentially expressed genes (Welch t against control) -------------------
    denom = np.sqrt(sem2[1:] + sem2[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(denom > 0, (mean[1:] - mean[0]) / denom, 0.0)
    de_genes = np.argsort(-np.abs(tstat), axis=1, kind="stable")[:, : args.n_de]

    # 7. split-half replicate -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    half = np.zeros(lognorm.shape[0], dtype=bool)
    for i in range(1, len(order)):
        members = np.flatnonzero(rows == i)
        half[rng.permutation(members)[: len(members) // 2]] = True
    cells_a = np.flatnonzero((rows >= 1) & half)
    ind_a = sp.csr_matrix((np.ones(len(cells_a)), (rows[cells_a], cells_a)), shape=indicator.shape)
    n_a = np.asarray(ind_a.sum(axis=1)).ravel()[1:]
    s_a = group_sums(ind_a, lognorm)[1:]
    mean_a = s_a / n_a[:, None]
    mean_b = (s1[1:] - s_a) / (n[1:] - n_a)[:, None]

    # round and assemble ----------------------------------------------------------
    def stored(values: np.ndarray) -> np.ndarray:
        return np.round(values, args.decimals).astype(np.float32).astype(np.float64)

    conditions = np.array([names[r] for r in order[1:]])
    data = PseudobulkData(
        genes=genes,
        conditions=conditions,
        mean=stored(mean[1:]),
        n_cells=n[1:].astype(int),
        control_mean=stored(mean[0]),
        n_control_cells=int(n[0]),
        de_genes=de_genes,
    )
    extra = {f"splithalf_{m}": np.empty(len(conditions)) for m in METRICS}
    extra["sem2_all"] = np.empty(len(conditions))
    extra["sem2_top1000"] = np.empty(len(conditions))
    for i, cond in enumerate(conditions):
        all_genes, de, top = gene_sets(cond, data)
        scores = score_profile(stored(mean_a[i]), stored(mean_b[i]), data.control_mean, all_genes, de, top)
        for metric, value in scores.items():
            extra[f"splithalf_{metric}"][i] = value
        # 8. sampling noise of the pseudobulk mean, summed over the metric gene sets
        extra["sem2_all"][i] = sem2[i + 1][all_genes].sum()
        extra["sem2_top1000"][i] = sem2[i + 1][top].sum()
    top = data.top_expressed(N_TOP_EXPRESSED)

    on_target = {}
    gene_pos = {g: i for i, g in enumerate(genes.tolist())}
    for i, cond in enumerate(conditions):
        if "+" not in cond and cond in gene_pos:
            j = gene_pos[cond]
            rank = int(np.sum(np.abs(tstat[i]) > abs(tstat[i, j]))) + 1
            on_target[cond] = {"delta": round(float(mean[i + 1, j] - mean[0, j]), 4), "de_rank": rank}

    meta = {
        **SOURCE,
        "raw_sha256": sha256_of(args.raw),
        "raw_size_bytes": args.raw.stat().st_size,
        "raw_cells": int(n_cells_raw),
        "raw_genes": int(n_genes_raw),
        "cell_filter": "all cells" if args.keep_all_cells else "good_coverage and number_of_cells == 1",
        "cells_after_filter": int(keep.sum()),
        "cells_in_kept_conditions": int(n.sum()),
        "control_cells": int(n[0]),
        "min_cells_per_perturbation": args.min_cells,
        "perturbations_dropped": dropped,
        "n_single": int(sum("+" not in c for c in conditions)),
        "n_double": int(sum("+" in c for c in conditions)),
        "normalisation": f"counts / cell total (all {n_genes_raw} genes) * {args.target_sum:g}, then log1p",
        "gene_filter": f"detected in >= {args.min_detection:g} of kept cells, plus measured target genes",
        "genes_by_detection": n_by_detection,
        "target_genes_added": n_targets_added,
        "n_genes": int(len(genes)),
        "target_aliases": TARGET_ALIASES,
        "gene_renames": GENE_RENAMES,
        "targets_not_measured": missing_targets,
        "de_method": "Welch t on log-normalised expression, perturbation versus control cells, ranked by |t|",
        "n_de_stored": args.n_de,
        "metric_gene_sets_exclude_targets": EXCLUDE_TARGETS,
        "decimals": args.decimals,
        "splithalf_seed": args.seed,
        "control_sem2_all": float(sem2[0].sum()),
        "control_sem2_top1000": float(sem2[0][top].sum()),
        "on_target_singles_up": int(sum(v["delta"] > 0 for v in on_target.values())),
        "on_target_singles_top20": int(sum(v["de_rank"] <= 20 for v in on_target.values())),
        "on_target_singles_checked": len(on_target),
    }
    data = replace(data, meta=meta, extra=extra)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "norman2019_pseudobulk.npz"
    save_pseudobulk(data, out)
    table = pd.DataFrame(
        {
            "condition": conditions,
            "n_genes_perturbed": [c.count("+") + 1 for c in conditions],
            "n_cells": data.n_cells,
            **{k: np.round(v, 5) for k, v in extra.items()},
        }
    )
    table.to_csv(args.out_dir / "norman2019_conditions.csv", index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), sha256 {sha256_of(out)}")
    for key in (
        "cells_after_filter",
        "control_cells",
        "n_single",
        "n_double",
        "n_genes",
        "genes_by_detection",
        "target_genes_added",
        "targets_not_measured",
        "perturbations_dropped",
        "on_target_singles_up",
        "on_target_singles_top20",
        "on_target_singles_checked",
    ):
        print(f"  {key}: {meta[key]}")
    print(
        f"  cells per perturbation: min {data.n_cells.min()}, median {int(np.median(data.n_cells))}, "
        f"max {data.n_cells.max()}"
    )
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
