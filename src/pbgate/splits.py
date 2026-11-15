"""Train/test splits over perturbations (GEARS "simulation" convention).

A random subset of the perturbed genes is declared *unseen*. Every perturbation that
involves an unseen gene is held out. Of the double perturbations whose two genes are
both seen, a random fraction is held out as well. Test perturbations are then labelled
by what the realised training set contains:

``unseen_single``  single perturbation of a gene that occurs in no training perturbation
``combo_seen0``    double perturbation, neither gene occurs in a training perturbation
``combo_seen1``    double perturbation, exactly one gene occurs in a training perturbation
``combo_seen2``    double perturbation, both genes occur in training perturbations
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import genes_of

CATEGORIES = ("unseen_single", "combo_seen0", "combo_seen1", "combo_seen2")


class SplitError(ValueError):
    """Raised when a split leaks held-out perturbations or genes into training."""


@dataclass(frozen=True)
class Split:
    seed: int
    train: tuple[str, ...]
    test: tuple[str, ...]
    category: dict[str, str]  # test perturbation -> one of CATEGORIES
    unseen_genes: tuple[str, ...]
    draw: int = 0  # number of rejected draws before this split was accepted

    def test_in(self, category: str) -> list[str]:
        return [c for c in self.test if self.category[c] == category]

    @property
    def train_genes(self) -> set[str]:
        return {g for c in self.train for g in genes_of(c)}

    @property
    def train_singles(self) -> set[str]:
        return {c for c in self.train if len(genes_of(c)) == 1}

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n")

    @staticmethod
    def from_json(path: str | Path) -> Split:
        raw = json.loads(Path(path).read_text())
        return Split(
            seed=int(raw.get("seed", -1)),
            train=tuple(raw["train"]),
            test=tuple(raw["test"]),
            category=dict(raw.get("category", {})),
            unseen_genes=tuple(raw.get("unseen_genes", ())),
            draw=int(raw.get("draw", 0)),
        )


def categorise(condition: str, train_genes: set[str]) -> str:
    """Label a held-out perturbation by how many of its genes occur in training."""
    genes = genes_of(condition)
    n_seen = sum(g in train_genes for g in genes)
    if len(genes) == 1:
        if n_seen:
            raise SplitError(f"single perturbation {condition} is held out but its gene is in training")
        return "unseen_single"
    if len(genes) == 2:
        return f"combo_seen{n_seen}"
    raise SplitError(f"only single and double perturbations are supported: {condition}")


def _draw_split(
    conditions: list[str], seed: int, draw: int, unseen_gene_frac: float, seen2_test_frac: float
) -> Split:
    rng = np.random.default_rng(seed if draw == 0 else [seed, draw])
    all_genes = sorted({g for c in conditions for g in genes_of(c)})
    n_unseen = int(round(unseen_gene_frac * len(all_genes)))
    unseen = set(rng.choice(all_genes, size=n_unseen, replace=False).tolist())

    train, test = [], []
    seen_doubles = []
    for cond in conditions:
        genes = genes_of(cond)
        if any(g in unseen for g in genes):
            test.append(cond)
        elif len(genes) == 1:
            train.append(cond)
        else:
            seen_doubles.append(cond)
    n_held = int(round(seen2_test_frac * len(seen_doubles)))
    held = set(rng.choice(seen_doubles, size=n_held, replace=False).tolist()) if n_held else set()
    for cond in seen_doubles:
        (test if cond in held else train).append(cond)

    train_genes = {g for c in train for g in genes_of(c)}
    category = {c: categorise(c, train_genes) for c in test}
    return Split(
        seed=int(seed),
        train=tuple(sorted(train)),
        test=tuple(sorted(test)),
        category=category,
        unseen_genes=tuple(sorted(unseen)),
        draw=draw,
    )


def make_split(
    conditions: Sequence[str],
    seed: int,
    unseen_gene_frac: float = 0.25,
    seen2_test_frac: float = 0.25,
    min_per_category: int = 5,
    max_draws: int = 1000,
) -> Split:
    """Draw one split. ``conditions`` are all perturbations (control excluded).

    A draw is rejected, and repeated with the next sub-seed, when a category that can
    occur in the data has fewer than ``min_per_category`` held-out perturbations.
    ``Split.draw`` records how many draws were rejected.
    """
    conditions = sorted(conditions)
    sizes = {len(genes_of(c)) for c in conditions}
    required = [c for c in CATEGORIES if (1 if c == "unseen_single" else 2) in sizes]
    for draw in range(max_draws):
        split = _draw_split(conditions, seed, draw, unseen_gene_frac, seen2_test_frac)
        if all(len(split.test_in(c)) >= min_per_category for c in required):
            validate_split(split, conditions)
            return split
    raise SplitError(
        f"no split with at least {min_per_category} perturbations per category in {max_draws} draws"
    )


def split_from_lists(train: Sequence[str], test: Sequence[str], seed: int = -1) -> Split:
    """Build a split from explicit lists; categories are derived from the training set."""
    train_genes = {g for c in train for g in genes_of(c)}
    category = {c: categorise(c, train_genes) for c in test}
    unseen = sorted({g for c in test for g in genes_of(c)} - train_genes)
    return Split(int(seed), tuple(train), tuple(test), category, tuple(unseen), 0)


def validate_split(split: Split, conditions: Sequence[str] | None = None) -> None:
    """Raise :class:`SplitError` if the split is inconsistent or leaks.

    Checks: train and test are disjoint and non-empty; no duplicates; every name is a
    known perturbation; the control is in neither list; every category label agrees
    with the realised training set (so no gene declared unseen occurs in training).
    """
    train, test = set(split.train), set(split.test)
    if not train or not test:
        raise SplitError("train and test must both be non-empty")
    if len(train) != len(split.train) or len(test) != len(split.test):
        raise SplitError("duplicate perturbation in train or test")
    overlap = train & test
    if overlap:
        raise SplitError(f"perturbations in both train and test: {sorted(overlap)[:5]}")
    if conditions is not None:
        unknown = (train | test) - set(conditions)
        if unknown:
            raise SplitError(f"unknown perturbations: {sorted(unknown)[:5]}")
    for cond in train | test:
        if cond.lower() in {"control", "ctrl"}:
            raise SplitError("the control condition must not be listed as a perturbation")
    train_genes = split.train_genes
    leaked = set(split.unseen_genes) & train_genes
    if leaked:
        raise SplitError(f"genes declared unseen occur in training: {sorted(leaked)[:5]}")
    for cond in split.test:
        expected = categorise(cond, train_genes)
        if split.category.get(cond) != expected:
            raise SplitError(
                f"{cond} is labelled {split.category.get(cond)} but the training set implies {expected}"
            )
