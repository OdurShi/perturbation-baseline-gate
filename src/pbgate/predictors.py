"""Simple baseline predictors. Each one is fitted on :class:`TrainData` only.

All predictors return full expression profiles (mean log-normalised expression per
gene), not changes from control.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .data import TrainData, genes_of


class Predictor:
    """Interface: ``fit`` on training data, ``predict`` profiles for perturbation names."""

    name = "predictor"

    def fit(self, train: TrainData) -> Predictor:
        raise NotImplementedError

    def predict(self, conditions: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class NoChange(Predictor):
    """Predict the control mean for every perturbation."""

    name = "no_change"

    def fit(self, train: TrainData) -> NoChange:
        self.control_ = train.control_mean
        return self

    def predict(self, conditions: Sequence[str]) -> np.ndarray:
        return np.tile(self.control_, (len(conditions), 1))


class TrainMean(Predictor):
    """Predict the unweighted mean profile of all training perturbations."""

    name = "train_mean"

    def fit(self, train: TrainData) -> TrainMean:
        self.mean_ = train.mean.mean(axis=0)
        return self

    def predict(self, conditions: Sequence[str]) -> np.ndarray:
        return np.tile(self.mean_, (len(conditions), 1))


class Additive(Predictor):
    """control + sum of single-gene effects.

    The effect of a gene is the change from control of its single perturbation in the
    training set. For a gene without a training single, the mean effect of all training
    singles is substituted. A held-out single perturbation is therefore predicted as
    control + mean training single effect, and a double with two unseen genes as
    control + twice that mean effect.
    """

    name = "additive"

    def fit(self, train: TrainData) -> Additive:
        self.control_ = train.control_mean
        effects = train.effects()
        self.effect_ = {
            cond: effects[i] for i, cond in enumerate(train.conditions) if len(genes_of(cond)) == 1
        }
        if not self.effect_:
            raise ValueError("the additive model needs at least one single perturbation in training")
        self.mean_effect_ = np.mean(list(self.effect_.values()), axis=0)
        return self

    def n_singles_available(self, condition: str) -> int:
        return sum(g in self.effect_ for g in genes_of(condition))

    def predict(self, conditions: Sequence[str]) -> np.ndarray:
        out = np.empty((len(conditions), len(self.control_)))
        for i, cond in enumerate(conditions):
            out[i] = self.control_ + sum(self.effect_.get(g, self.mean_effect_) for g in genes_of(cond))
        return out


def gene_embedding(train: TrainData, rank: int) -> np.ndarray:
    """Gene embedding from PCA of the training pseudobulk (genes are the observations).

    The matrix has one row per gene and one column per training perturbation. Columns
    are centred and the scores on the first ``rank`` principal components are returned,
    shape (G, rank). Only training profiles enter.
    """
    y = train.mean.T  # (G, n_train)
    centred = y - y.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    rank = min(rank, len(s))
    scores = u[:, :rank] * s[:rank]
    # fix the sign of each component so that results do not depend on the SVD backend
    signs = np.sign(scores[np.abs(scores).argmax(axis=0), np.arange(rank)])
    signs[signs == 0] = 1.0
    return scores * signs


def perturbation_embedding(
    conditions: Sequence[str], gene_emb: np.ndarray, gene_index: dict[str, int]
) -> np.ndarray:
    """Embedding of a perturbation: sum of the gene embeddings of its perturbed genes.

    This needs the name of the perturbed genes only, never the perturbation's profile.
    A perturbed gene that is not among the measured genes contributes a zero vector.
    """
    out = np.zeros((len(conditions), gene_emb.shape[1]))
    for i, cond in enumerate(conditions):
        for gene in genes_of(cond):
            j = gene_index.get(gene)
            if j is not None:
                out[i] += gene_emb[j]
    return out


class LinearLowRank(Predictor):
    """Low-rank bilinear model  Y ~ base + G W P'  with ridge-regularised W.

    ``G`` (genes x K) is the gene embedding from PCA of the training pseudobulk and ``P``
    (perturbations x K) holds, for each perturbation, the sum of the rows of ``G`` of its
    perturbed genes. ``W`` minimises the squared error on the training perturbations with
    ridge penalty ``ridge`` on both sides.

    ``base`` is control + n * m, where n is the number of perturbed genes and m the
    least-squares effect per perturbed gene over the training perturbations. When all
    training perturbations are single, ``base`` equals the mean training profile, which
    is the offset ``b`` of the published model. With ``per_gene_shift=False`` the mean
    training profile is used as offset for singles and doubles alike.
    """

    name = "linear"

    def __init__(self, rank: int = 10, ridge: float = 0.1, per_gene_shift: bool = True) -> None:
        self.rank = rank
        self.ridge = ridge
        self.per_gene_shift = per_gene_shift

    def _base(self, conditions: Sequence[str]) -> np.ndarray:
        if not self.per_gene_shift:
            return np.tile(self.b_, (len(conditions), 1))
        n = np.array([len(genes_of(c)) for c in conditions], dtype=float)
        return self.control_ + n[:, None] * self.shift_

    def fit(self, train: TrainData) -> LinearLowRank:
        self.gene_index_ = {g: i for i, g in enumerate(train.genes.tolist())}
        self.control_ = train.control_mean
        self.b_ = train.mean.mean(axis=0)  # (G,)
        n = np.array([len(genes_of(c)) for c in train.conditions], dtype=float)
        self.shift_ = n @ train.effects() / (n @ n)  # (G,) effect per perturbed gene
        g = gene_embedding(train, self.rank)  # (G, K)
        p = perturbation_embedding(train.conditions, g, self.gene_index_)  # (N, K)
        k = g.shape[1]
        y = (train.mean - self._base(train.conditions)).T  # (G, N)
        left = np.linalg.solve(g.T @ g + self.ridge * np.eye(k), g.T @ y)  # (K, N)
        w = np.linalg.solve(p.T @ p + self.ridge * np.eye(k), (left @ p).T).T  # (K, K)
        self.g_, self.w_ = g, w
        return self

    def predict(self, conditions: Sequence[str]) -> np.ndarray:
        p = perturbation_embedding(conditions, self.g_, self.gene_index_)
        return self._base(conditions) + p @ self.w_.T @ self.g_.T


class NearestPerturbations(Predictor):
    """Mean profile of the k training perturbations nearest in embedding space.

    Perturbation embeddings are built as in :class:`LinearLowRank`. Each embedding
    dimension is scaled to unit variance over genes and Euclidean distance is used.
    """

    name = "knn"

    def __init__(self, k: int = 5, rank: int = 10) -> None:
        self.k = k
        self.rank = rank

    def fit(self, train: TrainData) -> NearestPerturbations:
        self.gene_index_ = {g: i for i, g in enumerate(train.genes.tolist())}
        g = gene_embedding(train, self.rank)
        scale = g.std(axis=0)
        scale[scale == 0] = 1.0
        self.g_ = g / scale
        self.train_emb_ = perturbation_embedding(train.conditions, self.g_, self.gene_index_)
        self.train_mean_ = train.mean
        return self

    def predict(self, conditions: Sequence[str]) -> np.ndarray:
        emb = perturbation_embedding(conditions, self.g_, self.gene_index_)
        dist = np.linalg.norm(emb[:, None, :] - self.train_emb_[None, :, :], axis=2)
        k = min(self.k, dist.shape[1])
        nearest = np.argsort(dist, axis=1, kind="stable")[:, :k]
        return self.train_mean_[nearest].mean(axis=1)


def default_predictors() -> list[Predictor]:
    """The baselines every candidate model is compared with, with fixed settings."""
    return [NoChange(), TrainMean(), Additive(), LinearLowRank(rank=10, ridge=0.1), NearestPerturbations(k=5)]


BASELINE_NAMES = tuple(p.name for p in default_predictors())
