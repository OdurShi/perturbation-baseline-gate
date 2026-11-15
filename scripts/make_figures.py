#!/usr/bin/env python
"""Draw the figures from the files in results/.

figures/teaser_baselines_pearson.png   baselines per split type, two Pearson metrics (1600 x 1000 px)
figures/baselines_errors.png           baselines per split type, two error metrics
figures/additivity.png                 additive expectation versus observed double effects
figures/ranking.png                    rank of each baseline under each metric
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pbgate.data import genes_of, load_pseudobulk
from pbgate.metrics import METRICS, target_mask
from pbgate.plotting import (
    CATEGORY_LABELS,
    DPI,
    INK,
    METRIC_LABELS,
    MUTED,
    REFERENCE,
    STYLE,
    apply_style,
    light_grid,
    plt,
    save,
)
from pbgate.splits import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]


def dot_panel(ax, summary, reference, metric, predictors) -> None:
    """Mean and 95% interval of each predictor in each split category."""
    width = 0.72
    offsets = np.linspace(-width / 2, width / 2, len(predictors) + 2)[1:-1]
    for j, predictor in enumerate(predictors):
        rows = summary[(summary["predictor"] == predictor) & (summary["metric"] == metric)]
        rows = rows.set_index("category").loc[list(CATEGORIES)]
        x = np.arange(len(CATEGORIES)) + offsets[j]
        style = STYLE[predictor]
        ax.errorbar(
            x,
            rows["mean"],
            yerr=[rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]],
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.1,
            capsize=0,
            zorder=2,
        )
        ax.plot(
            x,
            rows["mean"],
            linestyle="none",
            marker=style["marker"],
            markersize=6.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=style["label"],
            zorder=3,
        )
    ref = reference[reference["metric"] == metric].set_index("category").loc[list(CATEGORIES)]
    for i, value in enumerate(ref["mean"]):
        ax.plot(
            [i - 0.42, i + 0.42],
            [value, value],
            color=REFERENCE,
            linewidth=1.2,
            zorder=1,
            label="split-half replicate" if i == 0 else None,
        )
    ax.set_xticks(np.arange(len(CATEGORIES)), [CATEGORY_LABELS[c] for c in CATEGORIES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
    ax.set_ylabel(METRIC_LABELS[metric])
    light_grid(ax)


def figure_pearson(summary, reference, out: Path) -> None:
    apply_style(font_size=11)
    fig, axes = plt.subplots(1, 2, figsize=(1600 / DPI, 1000 / DPI), sharey=True)
    predictors = ["train_mean", "additive", "linear", "knn"]
    for ax, metric in zip(axes, ("pearson_all", "pearson_de20"), strict=True):
        dot_panel(ax, summary, reference, metric, predictors)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("Held-out perturbation type")
    axes[1].tick_params(axis="y", labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=10,
        handletextpad=0.3,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.1, right=0.985, top=0.91, bottom=0.17, wspace=0.3)
    save(fig, out)


def figure_errors(summary, reference, out: Path) -> None:
    apply_style(font_size=10)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    predictors = ["no_change", "train_mean", "additive", "linear", "knn"]
    for ax, metric in zip(axes, ("mse_de20", "l2_top1000"), strict=True):
        dot_panel(ax, summary, reference, metric, predictors)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Held-out perturbation type")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        fontsize=8.5,
        handletextpad=0.3,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.18, wspace=0.3)
    save(fig, out)


def figure_additivity(data, per_double: pd.DataFrame, out: Path) -> None:
    apply_style(font_size=10)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    control = data.control_mean
    index = data.index
    xs, ys = [], []
    for cond in per_double["condition"]:
        a, b = genes_of(cond)
        keep = ~target_mask(cond, data)
        xs.append(((data.mean[index[a]] - control) + (data.mean[index[b]] - control))[keep])
        ys.append((data.mean[index[cond]] - control)[keep])
    x, y = np.concatenate(xs), np.concatenate(ys)
    ax = axes[0]
    lim = 2.5
    hb = ax.hexbin(
        x, y, gridsize=70, extent=(-lim, lim, -lim, lim), bins="log", cmap="Blues", mincnt=1, linewidths=0
    )
    ax.plot([-lim, lim], [-lim, lim], color=INK, linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Sum of the two single effects (log expression)")
    ax.set_ylabel("Observed double effect (log expression)")
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Gene x double pairs per bin", color=INK)
    cbar.outline.set_linewidth(0.4)
    shown = float(np.mean((np.abs(x) <= lim) & (np.abs(y) <= lim)))
    ax.text(
        0.03,
        0.97,
        f"{len(per_double)} doubles x {len(control) - 2:,} genes\n{100 * shown:.2f}% of pairs in range",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=MUTED,
    )

    ax = axes[1]
    series = {
        "all": ("all genes", "#2a78d6"),
        "top1000": ("1,000 most expressed", "#eb6834"),
        "de20": ("top 20 DE genes", "#1baf7a"),
    }
    for name, (label, colour) in series.items():
        values = np.sort(per_double[f"r2_{name}"].to_numpy())
        ax.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=colour,
            linewidth=1.4,
            label=label,
        )
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(0, 1.0)
    below = int((per_double[[f"r2_{n}" for n in series]].min(axis=1) < -0.5).sum())
    if below:
        ax.text(
            0.03,
            0.5,
            f"{below} doubles below -0.5\nin at least one gene set",
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            va="center",
        )
    ax.set_xlabel("R$^2$ of the additive model, per double")
    ax.set_ylabel("Cumulative share of double perturbations")
    ax.legend(loc="upper left", fontsize=8.5, title="Gene set", title_fontsize=8.5)
    light_grid(ax, axis="both")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.14, wspace=0.34)
    save(fig, out)


def figure_ranking(ranks: pd.DataFrame, out: Path) -> None:
    apply_style(font_size=10)
    fig, axes = plt.subplots(1, 4, figsize=(8, 3.4), sharey=True)
    short = {
        "pearson_all": "r\nall",
        "pearson_de20": "r\nDE20",
        "mse_de20": "MSE\nDE20",
        "l2_top1000": "L2\n1,000",
    }
    predictors = [p for p in STYLE if p in set(ranks["predictor"])]
    for ax, category in zip(axes, CATEGORIES, strict=True):
        sub = ranks[ranks["category"] == category]
        for predictor in predictors:
            style = STYLE[predictor]
            row = sub[sub["predictor"] == predictor].set_index("metric").loc[list(METRICS)]
            ax.plot(
                np.arange(len(METRICS)),
                row["rank"],
                color=style["color"],
                marker=style["marker"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=1.2,
                label=style["label"],
            )
        ax.set_xticks(np.arange(len(METRICS)), [short[m] for m in METRICS], fontsize=8)
        ax.set_xlim(-0.4, len(METRICS) - 0.6)
        ax.set_xlabel(CATEGORY_LABELS[category].replace("\n", " "), fontsize=9)
        ax.tick_params(axis="x", length=0)
        light_grid(ax)
    axes[0].set_ylim(len(predictors) + 0.4, 0.6)
    axes[0].set_yticks(range(1, len(predictors) + 1))
    axes[0].set_ylabel("Rank among baselines (1 = best)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.24, wspace=0.12)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.results / "baseline_summary.csv")
    reference = pd.read_csv(args.results / "splithalf_summary.csv")
    figure_pearson(summary, reference, args.out / "teaser_baselines_pearson.png")
    figure_errors(summary, reference, args.out / "baselines_errors.png")
    figure_additivity(
        load_pseudobulk(),
        pd.read_csv(args.results / "additivity_per_double.csv"),
        args.out / "additivity.png",
    )
    figure_ranking(pd.read_csv(args.results / "ranking_ranks.csv"), args.out / "ranking.png")
    for path in sorted(args.out.glob("*.png")):
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
