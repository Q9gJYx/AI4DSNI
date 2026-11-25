"""Figures of the notebooks: one style, one function per figure, files saved under images/.

Colours follow one rule each: a single hue for one series, a fixed categorical order for
several (never cycled), ink colours for all text. Values sit at bar tips instead of on a
second axis; grids are hairline and recessive.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dsni.schema import CLASSES, SPLITS, TASKS

IMAGES = Path(__file__).resolve().parents[2] / "images"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed categorical order
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"


def use_style() -> None:
    """Apply the shared matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "figure.dpi": 110,
        }
    )


def save(fig: Figure, name: str) -> Path:
    """Write ``images/<name>.png`` and return its path."""
    IMAGES.mkdir(exist_ok=True)
    path = IMAGES / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def _barh(ax: Axes, values: pd.Series, title: str, fmt: str = "{:,.0f}") -> None:
    """Thin horizontal bars, largest on top, with the value at every tip."""
    bars = ax.barh(
        values.index.astype(str)[::-1], values.to_numpy()[::-1], height=0.55, color=SERIES[0]
    )
    ax.bar_label(
        bars,
        labels=[fmt.format(v) for v in values.to_numpy()[::-1]],
        padding=3,
        color=MUTED,
        fontsize=8,
    )
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)


def attrition(steps: Mapping[str, int]) -> Figure:
    """Rows left after every step of the build."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    tidy = {"16s": "16S", "qc": "QC", "bacdive": "BacDive", "dedup": "deduplication"}
    name = lambda key: " ".join(tidy.get(word, word) for word in key.split("_"))  # noqa: E731
    series = pd.Series(steps).rename(name)
    _barh(ax, series, "Strains left after each build step")
    return fig


def label_coverage(table: pd.DataFrame) -> Figure:
    """Share of strains that carry each label, and all four."""
    fig, ax = plt.subplots(figsize=(6, 2.4))
    share = table[list(TASKS)].notna().mean()
    share["all four"] = table[list(TASKS)].notna().all(axis=1).mean()
    _barh(ax, share, "Share of strains with a label", fmt="{:.1%}")
    ax.set_xlim(0, 1.15)
    return fig


def class_balance(table: pd.DataFrame) -> Figure:
    """Class shares of temperature, pH and oxygen among the labelled strains."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.3))
    for ax, task in zip(axes, CLASSES, strict=True):
        _barh(ax, table[task].value_counts(normalize=True)[list(CLASSES[task])], task, fmt="{:.1%}")
        ax.set_xlim(0, 1.25)
    fig.tight_layout()
    return fig


def media_tail(table: pd.DataFrame, top: int = 12) -> Figure:
    """The most frequent media, and how many media it takes to cover the labelled strains."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), width_ratios=(1.25, 1))
    counts = table["media_id"].value_counts()
    names = table.attrs.get("media_names", {})
    head = counts.head(top).rename(lambda i: f"{names.get(i, 'medium')[:32].title()} ({i})")
    _barh(left, head, f"Top {top} primary media (strains)")
    covered = (counts.cumsum() / counts.sum()).to_numpy()
    n_classes = min(len(table["media"].cat.categories) - 1, len(covered))
    right.plot(np.arange(1, len(covered) + 1), covered, color=SERIES[0])
    point = (n_classes, covered[n_classes - 1])
    right.scatter(*point, s=48, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    label = f"{n_classes} media cover {point[1]:.0%}; the rest are 'Other'"
    right.annotate(
        label, point, xytext=(10, -16), textcoords="offset points", color=INK, fontsize=8
    )
    right.set(xscale="log", xlabel="media, most frequent first", ylim=(0, 1.02))
    right.set_title(f"Cumulative share of labelled strains ({len(counts):,} media)")
    fig.tight_layout()
    return fig


def sequence_lengths(table: pd.DataFrame) -> Figure:
    """Histogram of 16S sequence lengths."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.hist(table["seq_len"], bins=55, color=SERIES[0], rwidth=0.86)
    ax.set(xlabel="sequence length (nt)", ylabel="strains")
    ax.set_title(f"16S sequence length (median {table['seq_len'].median():,.0f} nt)")
    ax.grid(axis="x", visible=False)
    return fig


def taxonomy(table: pd.DataFrame, top: int = 8) -> Figure:
    """Largest phyla, and how unevenly strains are spread over genera."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.0), width_ratios=(1.2, 1))
    _barh(left, table["phylum"].value_counts().head(top), f"Top {top} phyla (strains)")
    sizes = table["genus"].value_counts()
    edges = [1, 2, 3, 6, 11, 51, 10_000]
    names = ["1", "2", "3 to 5", "6 to 10", "11 to 50", "over 50"]
    binned = pd.cut(sizes, edges, right=False, labels=names).value_counts()[names]
    bars = right.bar(names, binned.to_numpy(), width=0.55, color=SERIES[0])
    right.bar_label(bars, padding=2, color=MUTED, fontsize=8)
    right.set(xlabel="strains per genus", ylabel="genera")
    right.set_title(f"{len(sizes):,} genera; {(sizes == 1).mean():.0%} have a single strain")
    right.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def split_balance(table: pd.DataFrame, task: str = "oxygen") -> Figure:
    """Class shares of one task in train, val and test, missing included."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = table[task].astype(object).fillna("missing")
    shares = pd.crosstab(labels, table["split"], normalize="columns")[list(SPLITS)]
    shares.plot.bar(ax=ax, color=list(SERIES[:3]), width=0.8, rot=0, xlabel="")
    ax.set(ylabel="share of the split")
    ax.set_title(f"{task}: class shares per split")
    ax.legend(ncols=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def leakage(similarity: Mapping[str, np.ndarray]) -> Figure:
    """Similarity of every test strain to its closest training strain, per split protocol."""
    fig, ax = plt.subplots(figsize=(7, 2.9))
    bins = np.linspace(0.8, 1.0, 41).tolist()
    for (name, values), color in zip(similarity.items(), SERIES, strict=False):
        label = f"{name} split (median {np.median(values):.3f})"
        ax.hist(
            np.clip(values, bins[0], 1),
            bins=bins,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )
    ax.set(xlabel="k-mer cosine similarity to the nearest training strain", ylabel="test strains")
    ax.set_title("A random split puts near-copies of test strains in training")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def gate_profile(profiles: Mapping[str, np.ndarray]) -> Figure:
    """Mean gate value along the 16S gene, with the positions tagged hypervariable shaded.

    Args:
        profiles: Output of :func:`dsni.models.gate_profile` per curve, for example the trained
            model and the same model at initialisation.
    """
    from dsni.features import V_REGIONS

    fig, ax = plt.subplots(figsize=(8, 2.8))
    length = max(len(profile) for profile in profiles.values())
    for i, (start, end) in enumerate(V_REGIONS):
        if start >= length:  # only the regions the input reaches
            continue
        end = min(end, length)
        ax.axvspan(start, end, color=SERIES[3], alpha=0.15, linewidth=0)
        ax.text(
            (start + end) / 2,
            1.0,
            f"V{i + 1}",
            ha="center",
            va="top",
            fontsize=7,
            color=MUTED,
            transform=ax.get_xaxis_transform(),
        )
    for (name, profile), color in zip(profiles.items(), (SERIES[0], MUTED), strict=False):
        ax.plot(
            np.arange(len(profile)),
            profile,
            color=color,
            linewidth=1.5 if color == MUTED else 2,
            label=name,
        )
    ax.axhline(0.5, color=INK, linewidth=0.8)
    ax.set(xlabel="position in the input (nt)", ylabel="mean gate value", xlim=(0, length))
    ax.set_title("What the variability gate learned along the 16S gene (region tags shaded)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig
