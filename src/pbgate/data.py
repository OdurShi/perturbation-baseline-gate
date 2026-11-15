"""Condition-level pseudobulk container and file I/O."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np

SEP = "+"
DATA_FILE = "norman2019_pseudobulk.npz"


def genes_of(condition: str) -> tuple[str, ...]:
    """Perturbed genes of a condition name such as ``"KLF1"`` or ``"CBL+CNN1"``."""
    return tuple(condition.split(SEP))


def condition_name(genes: Sequence[str]) -> str:
    """Canonical condition name: perturbed genes sorted and joined with ``+``."""
    return SEP.join(sorted(genes))


@dataclass(frozen=True)
class TrainData:
    """What a predictor is allowed to see: control and training perturbations."""

    genes: np.ndarray  # (G,) gene symbols
    control_mean: np.ndarray  # (G,) mean log-normalised expression of control cells
    conditions: tuple[str, ...]  # training perturbations
    mean: np.ndarray  # (n_train, G) mean log-normalised expression

    def effects(self) -> np.ndarray:
        """Change from control of every training perturbation, shape (n_train, G)."""
        return self.mean - self.control_mean


@dataclass(frozen=True)
class PseudobulkData:
    """Mean log-normalised expression per perturbation, plus control and metadata."""

    genes: np.ndarray  # (G,) gene symbols
    conditions: np.ndarray  # (C,) perturbation names (control excluded)
    mean: np.ndarray  # (C, G)
    n_cells: np.ndarray  # (C,)
    control_mean: np.ndarray  # (G,)
    n_control_cells: int
    de_genes: np.ndarray  # (C, n_de) gene indices ranked by decreasing |Welch t| versus control
    meta: dict = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # per-condition scalars, each (C,)

    def __post_init__(self) -> None:
        n_cond, n_genes = self.mean.shape
        if len(self.conditions) != n_cond or len(self.genes) != n_genes:
            raise ValueError("mean matrix does not match conditions and genes")
        if self.control_mean.shape != (n_genes,) or len(self.n_cells) != n_cond:
            raise ValueError("control_mean or n_cells has the wrong shape")
        if len(self.de_genes) != n_cond:
            raise ValueError("de_genes must have one row per condition")
        if len(set(self.conditions.tolist())) != n_cond:
            raise ValueError("condition names are not unique")

    # ------------------------------------------------------------------ lookups
    @cached_property
    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.conditions.tolist())}

    @cached_property
    def gene_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genes.tolist())}

    @property
    def singles(self) -> list[str]:
        return [c for c in self.conditions.tolist() if len(genes_of(c)) == 1]

    @property
    def doubles(self) -> list[str]:
        return [c for c in self.conditions.tolist() if len(genes_of(c)) == 2]

    def rows(self, conditions: Sequence[str]) -> np.ndarray:
        index = self.index
        return np.array([index[c] for c in conditions], dtype=int)

    def profiles(self, conditions: Sequence[str]) -> np.ndarray:
        return self.mean[self.rows(conditions)]

    @cached_property
    def expression_order(self) -> np.ndarray:
        """Gene indices sorted by decreasing mean expression in control cells."""
        return np.argsort(-self.control_mean, kind="stable")

    def top_expressed(self, n: int = 1000) -> np.ndarray:
        """Indices of the ``n`` genes with the highest mean expression in control cells."""
        if n > len(self.genes):
            raise ValueError(f"asked for {n} genes but the data has {len(self.genes)}")
        return self.expression_order[:n]

    def train_view(self, train_conditions: Sequence[str]) -> TrainData:
        """Copy of the control and training rows only (no held-out profile is included)."""
        rows = self.rows(train_conditions)
        return TrainData(
            genes=self.genes.copy(),
            control_mean=self.control_mean.copy(),
            conditions=tuple(train_conditions),
            mean=self.mean[rows].copy(),
        )

    def with_scrambled(self, conditions: Sequence[str], rng: np.random.Generator) -> PseudobulkData:
        """Copy in which the profiles of ``conditions`` are replaced by gene-shuffled versions.

        Used by the leakage check: a predictor that never reads these rows returns the
        same predictions for the original and for the scrambled data.
        """
        mean = self.mean.copy()
        de_genes = self.de_genes.copy()
        for row in self.rows(conditions):
            perm = rng.permutation(mean.shape[1])
            mean[row] = self.mean[row][perm] + rng.normal(0.0, 0.5, size=mean.shape[1])
            de_genes[row] = rng.permutation(mean.shape[1])[: de_genes.shape[1]]
        return replace(self, mean=mean, de_genes=de_genes)


# ---------------------------------------------------------------------- file I/O
def default_data_path() -> Path:
    """Shipped pseudobulk file: ``$PBGATE_DATA`` or ``<repository>/data/``."""
    env = os.environ.get("PBGATE_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / DATA_FILE


def _arrays(data: PseudobulkData) -> dict[str, np.ndarray]:
    arrays = {
        "genes": data.genes.astype(str),
        "conditions": data.conditions.astype(str),
        "mean": data.mean.astype(np.float32),
        "n_cells": data.n_cells.astype(np.int32),
        "control_mean": data.control_mean.astype(np.float32),
        "n_control_cells": np.array(data.n_control_cells, dtype=np.int32),
        "de_genes": data.de_genes.astype(np.int32),
        "meta_json": np.array(json.dumps(data.meta, sort_keys=True)),
    }
    for key in sorted(data.extra):
        arrays[f"extra_{key}"] = np.asarray(data.extra[key], dtype=np.float64)
    return arrays


def save_pseudobulk(data: PseudobulkData, path: str | Path) -> None:
    """Write a compressed ``.npz`` with fixed entry order and timestamps.

    The file is byte-identical for identical content when written with the same zlib
    build, so its SHA-256 can be compared across runs on one machine.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in _arrays(data).items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def content_sha256(data: PseudobulkData) -> str:
    """SHA-256 over the array contents (independent of compression and platform)."""
    digest = hashlib.sha256()
    for name, array in _arrays(data).items():
        digest.update(name.encode())
        if array.dtype.kind == "U":
            digest.update("\n".join(np.atleast_1d(array).tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes())
    return digest.hexdigest()


def load_pseudobulk(path: str | Path | None = None) -> PseudobulkData:
    path = Path(path) if path is not None else default_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"pseudobulk file not found: {path}. Run from a checkout of the repository "
            "(pip install -e .) or pass the path explicitly."
        )
    with np.load(path, allow_pickle=False) as npz:
        extra = {
            key.removeprefix("extra_"): npz[key].astype(float)
            for key in npz.files
            if key.startswith("extra_")
        }
        return PseudobulkData(
            genes=npz["genes"].astype(str),
            conditions=npz["conditions"].astype(str),
            mean=npz["mean"].astype(np.float64),
            n_cells=npz["n_cells"].astype(int),
            control_mean=npz["control_mean"].astype(np.float64),
            n_control_cells=int(npz["n_control_cells"]),
            de_genes=npz["de_genes"].astype(int),
            meta=json.loads(str(npz["meta_json"])),
            extra=extra,
        )


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()
