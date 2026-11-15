from __future__ import annotations

import numpy as np
import pytest

from pbgate.data import PseudobulkData, load_pseudobulk
from pbgate.splits import make_split


@pytest.fixture(scope="session")
def data() -> PseudobulkData:
    """The pseudobulk file shipped in data/ (no network access needed)."""
    return load_pseudobulk()


@pytest.fixture(scope="session")
def split(data):
    return make_split(data.conditions.tolist(), seed=0)


@pytest.fixture()
def toy() -> PseudobulkData:
    """Small synthetic data set: 8 genes A..H, all singles, some doubles, exactly additive."""
    rng = np.random.default_rng(1)
    targets = list("ABCDEFGH")
    genes = np.array(targets + [f"g{i}" for i in range(1200)])
    control = rng.uniform(0.0, 4.0, size=len(genes))
    effect = {t: rng.normal(0.0, 0.3, size=len(genes)) for t in targets}
    conditions, rows = [], []
    for t in targets:
        conditions.append(t)
        rows.append(control + effect[t])
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            conditions.append(f"{a}+{b}")
            rows.append(control + effect[a] + effect[b])
    mean = np.array(rows)
    de = np.argsort(-np.abs(mean - control), axis=1)[:, :50]
    return PseudobulkData(
        genes=genes,
        conditions=np.array(conditions),
        mean=mean,
        n_cells=np.full(len(conditions), 100),
        control_mean=control,
        n_control_cells=1000,
        de_genes=de,
    )
