#!/usr/bin/env python3
"""前回BICと局所fine-grid CVの変化点性能を同一データで比較する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pilot.plot_refined_cv_prediction_rmise import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    prepare_bic_records,
    prepare_records,
)
from scripts.pilot.prepare_slides22_results import (  # noqa: E402
    LABELS,
    SCENARIOS,
    change_point_counts,
)


def add_change_point_counts(records: pd.DataFrame, *, z_tolerance: float) -> pd.DataFrame:
    truths = {
        scenario: json.loads(
            (ROOT / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }
    rows: list[tuple[int, int, int]] = []
    candidate_slots: list[int] = []
    for row in records.itertuples(index=False):
        result_path = Path(str(row.result_path))
        if not result_path.is_absolute():
            result_path = ROOT / "outputs" / result_path
        result = json.loads(result_path.read_text(encoding="utf-8"))
        candidate_slots.append(int(np.asarray(result["z_last"]).size))
        rows.append(
            change_point_counts(
                result, truths[row.scenario], row.scenario, z_tolerance
            )
        )
    counted = records.copy()
    counted[["true_positive", "detected", "truth"]] = pd.DataFrame(
        rows, index=counted.index
    )
    counted["candidate_slots"] = candidate_slots
    return counted


def summarize(records: pd.DataFrame, *, method: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario in SCENARIOS:
        subset = records.loc[records["scenario"] == scenario]
        true_positive = int(subset["true_positive"].sum())
        detected = int(subset["detected"].sum())
        truth = int(subset["truth"].sum())
        precision = true_positive / detected if detected else np.nan
        recall = true_positive / truth if truth else np.nan
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision + recall) and precision + recall > 0
            else np.nan
        )
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "n": len(subset),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "zero_change_datasets": int((subset["detected"] == 0).sum()),
                "detected_change_points": detected,
                "true_positive": true_positive,
                "false_positive": detected - true_positive,
                "false_negative": truth - true_positive,
                "true_negative": int(subset["candidate_slots"].sum())
                - truth
                - (detected - true_positive),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric(ax: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    scenarios = SCENARIOS[:-1]
    x = np.arange(len(scenarios))
    width = 0.36
    for method_index, method in enumerate(METHODS):
        subset = summary.loc[summary["method"] == method].set_index("scenario")
        values = subset.loc[scenarios, metric].to_numpy(dtype=float)
        bars = ax.bar(
            x + (method_index - 0.5) * width,
            values,
            width,
            color=METHOD_COLORS[method],
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x, [LABELS[scenario] for scenario in scenarios])
    ax.set_ylim(0.0, 1.10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="both", labelsize=10)


def plot_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.4))
    _plot_metric(axes[0, 0], summary, "precision", "A. Precision")
    _plot_metric(axes[0, 1], summary, "recall", "B. Recall")
    _plot_metric(axes[1, 0], summary, "f1", "C. F1")

    ax = axes[1, 1]
    no_change = summary.loc[summary["scenario"] == "no_change"].set_index("method")
    values = no_change.loc[list(METHODS), "detected_change_points"].to_numpy(
        dtype=float
    )
    zero_counts = no_change.loc[list(METHODS), "zero_change_datasets"].to_numpy(
        dtype=int
    )
    sample_sizes = no_change.loc[list(METHODS), "n"].to_numpy(dtype=int)
    bars = ax.bar(
        np.arange(len(METHODS)),
        values,
        width=0.58,
        color=[METHOD_COLORS[method] for method in METHODS],
        alpha=0.9,
    )
    labels = [
        f"{int(value)}\n(no FP: {zero}/{n})"
        for value, zero, n in zip(values, zero_counts, sample_sizes)
    ]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
    ax.set_xticks(np.arange(len(METHODS)), ["BIC", "Local CV"])
    ax.set_ylim(0, 120)
    ax.set_ylabel("False change points")
    ax.set_title("D. No-change false positives", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="both", labelsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrices(summary: pd.DataFrame, output_path: Path) -> None:
    scenarios = ("small", "no_change")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.4))
    values = summary.loc[
        summary["scenario"].isin(scenarios),
        ["true_negative", "false_positive", "false_negative", "true_positive"],
    ].to_numpy(dtype=float)
    vmax = float(values.max())

    for row_index, scenario in enumerate(scenarios):
        for column_index, method in enumerate(METHODS):
            ax = axes[row_index, column_index]
            record = summary.loc[
                (summary["scenario"] == scenario) & (summary["method"] == method)
            ].iloc[0]
            matrix = np.asarray(
                [
                    [record["true_negative"], record["false_positive"]],
                    [record["false_negative"], record["true_positive"]],
                ],
                dtype=float,
            )
            ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
            cell_names = (("TN", "FP"), ("FN", "TP"))
            for actual_index in range(2):
                for detected_index in range(2):
                    value = int(matrix[actual_index, detected_index])
                    color = "white" if matrix[actual_index, detected_index] > 0.5 * vmax else "#111827"
                    ax.text(
                        detected_index,
                        actual_index,
                        f"{cell_names[actual_index][detected_index]}\n{value}",
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontweight="bold",
                        color=color,
                    )
            ax.set_xticks([0, 1], ["No", "Yes"])
            ax.set_yticks([0, 1], ["No", "Yes"])
            ax.set_xlabel("Detected change")
            ax.set_ylabel("Actual change")
            ax.set_title(
                f"{LABELS[scenario]} — {METHOD_LABELS[method]}",
                fontsize=14,
                fontweight="bold",
            )
            ax.tick_params(axis="both", labelsize=11)

    fig.text(
        0.5,
        0.99,
        "Unit: coefficient × analysis-grid boundary",
        ha="center",
        va="top",
        fontsize=12,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2, w_pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-selections", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument("--bic-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confusion-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--z-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    cv_records, _ = prepare_records(
        pd.read_csv(args.cv_selections), pd.read_csv(args.refit_summary)
    )
    cv_records = add_change_point_counts(
        cv_records, z_tolerance=args.z_tolerance
    )
    bic_records = prepare_bic_records(
        pd.read_csv(args.bic_metrics), set(cv_records["data_name"])
    )
    bic_records = bic_records.merge(
        cv_records[["data_name", "candidate_slots"]],
        on="data_name",
        how="left",
        validate="one_to_one",
    )
    summary = pd.concat(
        [
            summarize(bic_records, method="bic"),
            summarize(cv_records, method="refined_cv"),
        ],
        ignore_index=True,
    )
    plot_comparison(summary, args.output)
    if args.confusion_output is not None:
        plot_confusion_matrices(summary, args.confusion_output)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
    print(f"Saved BIC/CV change-point comparison for {len(cv_records)} datasets")


if __name__ == "__main__":
    main()
