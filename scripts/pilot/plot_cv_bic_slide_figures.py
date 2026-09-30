#!/usr/bin/env python3
"""MCPのCV/BIC比較を既存の罰則比較スライドと同じ図式で描く。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


SCENARIOS = ("oracle", "fine_grid", "off_grid", "small", "no_change")
LABELS = {
    "oracle": "Oracle",
    "fine_grid": "Fine-grid",
    "off_grid": "Off-grid",
    "small": "Small",
    "no_change": "No-change",
}
METHODS = ("BIC", "CV")
COLORS = {"BIC": "#6B7280", "CV": "#E76F51"}
MARKERS = {"BIC": "s", "CV": "o"}
OFFSETS = {"BIC": -0.13, "CV": 0.13}


def plot_prediction_and_rmise(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    y = np.arange(len(SCENARIOS))
    labels = [LABELS[scenario] for scenario in SCENARIOS]

    for method in METHODS:
        subset = summary.loc[summary["method"] == method].set_index("scenario").loc[list(SCENARIOS)]
        positions = y + OFFSETS[method]
        for axis, mean, se, xlabel, padding, digits in (
            (axes[0], "c_td_test_mean", "c_td_test_se", "Independent-test Ctd (mean and 95% CI)", 0.0007, 4),
            (axes[1], "rmise_mean", "rmise_se", "Coefficient RMISE (mean and 95% CI)", 0.003, 3),
        ):
            axis.errorbar(
                subset[mean], positions, xerr=1.96 * subset[se],
                fmt=MARKERS[method], color=COLORS[method], ecolor=COLORS[method],
                markerfacecolor="white" if method == "BIC" else COLORS[method],
                markeredgewidth=2.0, markersize=8.5, elinewidth=2.2,
                capsize=4, label=method if axis is axes[0] else None, zorder=3,
            )
            for index, value in enumerate(subset[mean]):
                label_x = value + 1.96 * subset[se].iloc[index] + padding
                axis.text(label_x, positions[index], f"{value:.{digits}f}", va="center", fontsize=11)
            axis.set_xlabel(xlabel)
            axis.grid(axis="x", alpha=0.25)

    for axis in axes:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.tick_params(axis="both", labelsize=12)
        axis.xaxis.label.set_size(14)
    axes[0].set_xlim(0.64, 0.72)
    axes[1].set_xlim(0.0, 0.24)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_score(axis: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    scenarios = SCENARIOS[:-1]
    x = np.arange(len(scenarios))
    width = 0.36
    for index, method in enumerate(METHODS):
        values = summary.loc[summary["method"] == method].set_index("scenario").loc[list(scenarios), metric]
        bars = axis.bar(x + (index - 0.5) * width, values, width,
                        color=COLORS[method], alpha=0.9, label=method)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    axis.set_xticks(x, [LABELS[scenario] for scenario in scenarios])
    axis.set_ylim(0.0, 1.10)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="both", labelsize=10)


def plot_change_points(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.4))
    plot_score(axes[0, 0], summary, "precision", "A. Precision")
    plot_score(axes[0, 1], summary, "recall", "B. Recall")
    plot_score(axes[1, 0], summary, "f1", "C. F1")

    axis = axes[1, 1]
    no_change = summary.loc[summary["scenario"] == "no_change"].set_index("method").loc[list(METHODS)]
    values = no_change["false_positive"].to_numpy(dtype=float)
    bars = axis.bar(np.arange(len(METHODS)), values, width=0.58,
                    color=[COLORS[method] for method in METHODS], alpha=0.9)
    bar_labels = [
        f"{int(value)}\n(no FP: {int(zero)}/{int(n)})"
        for value, zero, n in zip(values, no_change["zero_false_positive_datasets"], no_change["n"])
    ]
    axis.bar_label(bars, labels=bar_labels, padding=4, fontsize=10)
    axis.set_xticks(np.arange(len(METHODS)), list(METHODS))
    axis.set_ylim(0, max(140, float(values.max()) * 1.15))
    axis.set_ylabel("False change points")
    axis.set_title("D. No-change false positives", fontsize=13, fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="both", labelsize=10)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=2.5)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    records = pd.read_csv(args.records)
    expected = pd.MultiIndex.from_product([METHODS, SCENARIOS], names=["method", "scenario"])
    actual = pd.MultiIndex.from_frame(summary[["method", "scenario"]])
    if not actual.is_unique or not expected.isin(actual).all():
        raise ValueError("summary must contain each BIC/CV scenario exactly once")
    zero_counts = (
        records.assign(zero_false_positive=records["detected"].eq(0))
        .groupby(["method", "scenario"])["zero_false_positive"]
        .sum()
        .rename("zero_false_positive_datasets")
        .reset_index()
    )
    summary = summary.merge(zero_counts, on=["method", "scenario"], validate="one_to_one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_change_points(summary, args.output_dir / "cv_bic_change_point_scores.png")
    plot_prediction_and_rmise(summary, args.output_dir / "cv_bic_prediction_rmise.png")


if __name__ == "__main__":
    main()
