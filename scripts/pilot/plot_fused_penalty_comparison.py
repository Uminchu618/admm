#!/usr/bin/env python3
"""fused lasso と fused MCP の比較を0904スライドと同じ構成で作図する。"""

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
METHODS = ("lasso", "mcp")
METHOD_LABELS = {"lasso": "Fused lasso", "mcp": "Fused MCP"}
METHOD_COLORS = {"lasso": "#6B7280", "mcp": "#E76F51"}
METHOD_MARKERS = {"lasso": "s", "mcp": "o"}
METHOD_OFFSETS = {"lasso": -0.13, "mcp": 0.13}


def plot_prediction_and_rmise(summary: pd.DataFrame, output_path: Path) -> None:
    """0904の予測性能・係数RMISEと同じ横並びerrorbar図。"""

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    y = np.arange(len(SCENARIOS))
    labels = [LABELS[scenario] for scenario in SCENARIOS]

    for method in METHODS:
        subset = summary.loc[summary["method"] == method].set_index("scenario")
        subset = subset.loc[list(SCENARIOS)]
        y_method = y + METHOD_OFFSETS[method]
        for axis, mean, se, xlabel, padding in (
            (axes[0], "c_td_test_mean", "c_td_test_se", "Independent-test Ctd (mean and 95% CI)", 0.0015),
            (axes[1], "rmise_mean", "rmise_se", "Coefficient RMISE (mean and 95% CI)", 0.006),
        ):
            axis.errorbar(
                subset[mean], y_method, xerr=1.96 * subset[se],
                fmt=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                ecolor=METHOD_COLORS[method],
                markerfacecolor="white" if method == "lasso" else METHOD_COLORS[method],
                markeredgewidth=2.0, markersize=8.5, elinewidth=2.2,
                capsize=4, label=METHOD_LABELS[method] if axis is axes[0] else None, zorder=3,
            )
            for index, value in enumerate(subset[mean]):
                axis.text(value + padding, y_method[index], f"{value:.3f}", va="center", fontsize=11)
            axis.set_xlabel(xlabel)
            axis.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0.64, 0.72)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(0.0, 0.24)
    for axis in axes:
        axis.invert_yaxis()
        axis.tick_params(axis="both", labelsize=12)
        axis.xaxis.label.set_size(14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_score(axis: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    scenarios = SCENARIOS[:-1]
    x = np.arange(len(scenarios))
    width = 0.36
    for method_index, method in enumerate(METHODS):
        values = summary.loc[summary["method"] == method].set_index("scenario").loc[list(scenarios), metric]
        bars = axis.bar(x + (method_index - 0.5) * width, values, width, color=METHOD_COLORS[method], alpha=0.9, label=METHOD_LABELS[method])
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    axis.set_xticks(x, [LABELS[scenario] for scenario in scenarios])
    axis.set_ylim(0.0, 1.10)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="both", labelsize=10)


def plot_change_points(summary: pd.DataFrame, output_path: Path) -> None:
    """0904のPrecision/Recall/F1/No-change偽陽性の2×2構成。"""

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.4))
    _plot_score(axes[0, 0], summary, "precision", "A. Precision")
    _plot_score(axes[0, 1], summary, "recall", "B. Recall")
    _plot_score(axes[1, 0], summary, "f1", "C. F1")

    axis = axes[1, 1]
    no_change = summary.loc[summary["scenario"] == "no_change"].set_index("method").loc[list(METHODS)]
    values = no_change["false_positive"].to_numpy(dtype=float)
    bars = axis.bar(np.arange(len(METHODS)), values, width=0.58, color=[METHOD_COLORS[method] for method in METHODS], alpha=0.9)
    labels = [f"{int(value)}\n(no FP: {int(zero)}/{int(n)})" for value, zero, n in zip(values, no_change["zero_false_positive_datasets"], no_change["n"])]
    axis.bar_label(bars, labels=labels, padding=4, fontsize=10)
    axis.set_xticks(np.arange(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    axis.set_ylim(0, max(140, float(values.max()) * 1.15))
    axis.set_ylabel("False change points")
    axis.set_title("D. No-change false positives", fontsize=13, fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="both", labelsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--change-point-output", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    records = pd.read_csv(args.records)
    expected = set(METHODS) | set(SCENARIOS)
    if not set(summary["method"]).issuperset(METHODS) or not set(summary["scenario"]).issuperset(SCENARIOS):
        raise ValueError(f"summary must include {expected}")
    zero_counts = (
        records.assign(zero_false_positive=records["false_positive"].eq(0))
        .groupby(["method", "scenario"])["zero_false_positive"]
        .sum()
        .rename("zero_false_positive_datasets")
        .reset_index()
    )
    summary = summary.merge(zero_counts, on=["method", "scenario"], how="left", validate="one_to_one")
    plot_prediction_and_rmise(summary, args.prediction_output)
    plot_change_points(summary, args.change_point_output)


if __name__ == "__main__":
    main()
