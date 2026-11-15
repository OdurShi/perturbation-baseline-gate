"""Shared figure style: white background, sans-serif, thin marks, colour-blind-safe."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 200
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#9a9994"

# colour and marker per predictor; shape repeats the identity so colour is never the only cue
STYLE = {
    "additive": {"color": "#2a78d6", "marker": "o", "label": "additive"},
    "linear": {"color": "#eb6834", "marker": "s", "label": "linear low-rank"},
    "knn": {"color": "#1baf7a", "marker": "^", "label": "nearest perturbations"},
    "train_mean": {"color": "#52514e", "marker": "D", "label": "training mean"},
    "no_change": {"color": "#9a9994", "marker": "X", "label": "no change"},
}
CATEGORY_LABELS = {
    "unseen_single": "single\nunseen",
    "combo_seen0": "double\n0/2 seen",
    "combo_seen1": "double\n1/2 seen",
    "combo_seen2": "double\n2/2 seen",
}
METRIC_LABELS = {
    "pearson_all": "Pearson r of change from control,\nall genes",
    "pearson_de20": "Pearson r of change from control,\ntop 20 DE genes",
    "mse_de20": "Mean squared error,\ntop 20 DE genes",
    "l2_top1000": "L2 distance,\n1,000 most expressed genes",
}


def apply_style(font_size: float = 10.0) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": font_size,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
        }
    )


def light_grid(ax, axis: str = "y") -> None:
    """Thin solid grid lines behind the marks."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, path) -> None:
    """Save as an opaque RGB PNG at ``DPI`` without changing the figure size."""
    from PIL import Image

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    rgb.save(path, dpi=(DPI, DPI))
