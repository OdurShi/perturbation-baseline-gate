from __future__ import annotations

from dataclasses import replace

import pytest

from pbgate.data import genes_of
from pbgate.splits import CATEGORIES, Split, SplitError, make_split, split_from_lists, validate_split


def test_split_is_deterministic_and_disjoint(data):
    conditions = data.conditions.tolist()
    a, b = make_split(conditions, 3), make_split(conditions[::-1], 3)
    assert a == b
    assert not set(a.train) & set(a.test)
    assert set(a.train) | set(a.test) == set(conditions)
    assert make_split(conditions, 4).test != a.test


@pytest.mark.parametrize("seed", range(20))
def test_no_unseen_gene_in_training_and_no_degenerate_category(data, seed):
    split = make_split(data.conditions.tolist(), seed)
    train_genes = split.train_genes
    assert not set(split.unseen_genes) & train_genes
    for cond in split.test:
        seen = sum(g in train_genes for g in genes_of(cond))
        expected = "unseen_single" if len(genes_of(cond)) == 1 else f"combo_seen{seen}"
        assert split.category[cond] == expected
    for category in CATEGORIES:
        assert len(split.test_in(category)) >= 5
    # in this data every single of a seen gene is in training, so "2 of 2 seen" doubles
    # always have both single profiles available to the additive model
    singles = split.train_singles
    for cond in split.test_in("combo_seen2"):
        assert all(g in singles for g in genes_of(cond))


def test_validation_catches_leaks(data, split):
    conditions = data.conditions.tolist()
    moved = split.test[0]
    with pytest.raises(SplitError):
        validate_split(replace(split, train=(*split.train, moved)), conditions)
    single = split.test_in("unseen_single")[0]
    with pytest.raises(SplitError):  # the gene of a held-out single enters training through a double
        validate_split(replace(split, train=(*split.train, f"{single}+ZZZ")), None)
    wrong = dict(split.category)
    wrong[split.test_in("combo_seen1")[0]] = "combo_seen2"
    with pytest.raises(SplitError):
        validate_split(replace(split, category=wrong), conditions)
    with pytest.raises(SplitError):
        validate_split(replace(split, train=(*split.train, "control")), None)


def test_json_round_trip_and_explicit_lists(split, tmp_path):
    path = tmp_path / "split.json"
    split.to_json(path)
    assert Split.from_json(path) == split
    rebuilt = split_from_lists(split.train, split.test, seed=split.seed)
    assert rebuilt.category == split.category
