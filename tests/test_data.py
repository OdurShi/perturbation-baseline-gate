from __future__ import annotations

from pathlib import Path

import numpy as np

from pbgate.data import (
    content_sha256,
    default_data_path,
    genes_of,
    load_pseudobulk,
    save_pseudobulk,
    sha256_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shipped_file_matches_recorded_checksums(data):
    recorded = dict(
        line.split()[::-1]
        for line in (REPO_ROOT / "data" / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    )
    assert content_sha256(data) == recorded["norman2019_pseudobulk.npz:content"]
    # the file hash also depends on the zlib build that wrote the archive
    assert sha256_of(default_data_path()) == recorded["norman2019_pseudobulk.npz"]


def test_shape_and_counts(data):
    assert data.mean.shape == (len(data.conditions), len(data.genes))
    assert len(data.singles) == 105
    assert len(data.doubles) == 131
    assert np.all(np.isfinite(data.mean)) and np.all(np.isfinite(data.control_mean))
    assert data.n_cells.min() >= data.meta["min_cells_per_perturbation"]
    assert data.n_control_cells == data.meta["control_cells"]
    assert "control" not in data.conditions.tolist()


def test_every_double_has_both_singles_and_measured_targets(data):
    singles = set(data.singles)
    gene_index = data.gene_index
    for cond in data.doubles:
        a, b = genes_of(cond)
        assert a < b, "gene names of a double are sorted"
        assert a in singles and b in singles
    for gene in singles:
        assert gene in gene_index, "every perturbed gene is on the gene axis"


def test_de_genes_are_valid_indices(data):
    assert data.de_genes.shape[1] >= 22  # 20 remain after removing up to two targets
    assert data.de_genes.min() >= 0 and data.de_genes.max() < len(data.genes)
    assert all(len(set(row)) == len(row) for row in data.de_genes.tolist())


def test_save_is_deterministic_and_round_trips(data, tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    save_pseudobulk(data, first)
    save_pseudobulk(data, second)
    assert sha256_of(first) == sha256_of(second)
    again = load_pseudobulk(first)
    assert content_sha256(again) == content_sha256(data)
    assert np.array_equal(again.mean, data.mean)
    assert again.meta == data.meta
