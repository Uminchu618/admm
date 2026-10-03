#!/usr/bin/env python3
"""BIC選択fitの予測・係数・変化点回復を集計して可視化する。"""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pilot.prepare_slides22_results import (  # noqa: E402
    COLORS,
    LABELS,
    SCENARIOS,
    change_point_counts,
    coefficient_rmise,
)


def _result_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / "outputs" / path


def build_selected_metrics(
    selected: pd.DataFrame,
    *,
    z_tolerance: float,
) -> pd.DataFrame:
    required = {
        "data_name",
        "lambda_fuse",
        "bic",
        "c_td_test",
        "result_path",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    truths = {
        scenario: json.loads(
            (ROOT / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }

    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        scenario = next(
            (
                candidate
                for candidate in SCENARIOS
                if str(row.data_name).startswith(f"{candidate}_seed_")
            ),
            None,
        )
        if scenario is None:
            raise ValueError(f"unknown pilot data name: {row.data_name}")
        seed = int(str(row.data_name).removeprefix(f"{scenario}_seed_"))
        result_path = _result_path(str(row.result_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        true_positive, detected, truth = change_point_counts(
            result, truths[scenario], scenario, z_tolerance
        )
        rows.append(
            {
                "data_name": row.data_name,
                "scenario": scenario,
                "seed": seed,
                "lambda_fuse": float(row.lambda_fuse),
                "bic": float(row.bic),
                "c_td_test": float(row.c_td_test),
                "rmise": coefficient_rmise(result, truths[scenario]),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
                "false_positive": detected - true_positive,
                "result_path": str(row.result_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "seed"]).reset_index(drop=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        subset = metrics.loc[metrics["scenario"] == scenario]
        n = len(subset)
        if n == 0:
            continue
        true_positive = int(subset["true_positive"].sum())
        detected = int(subset["detected"].sum())
        truth = int(subset["truth"].sum())
        precision = true_positive / detected if detected else np.nan
        recall = true_positive / truth if truth else np.nan
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision + recall) and precision + recall > 0.0
            else np.nan
        )
        rows.append(
            {
                "scenario": scenario,
                "n": n,
                "lambda_mean": subset["lambda_fuse"].mean(),
                "lambda_median": subset["lambda_fuse"].median(),
                "c_td_test_mean": subset["c_td_test"].mean(),
                "c_td_test_se": subset["c_td_test"].std(ddof=1) / np.sqrt(n),
                "rmise_mean": subset["rmise"].mean(),
                "rmise_se": subset["rmise"].std(ddof=1) / np.sqrt(n),
                "change_points_mean": subset["detected"].mean(),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
                "false_positive": detected - true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def _mean_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    mean_column: str,
    se_column: str,
    xlabel: str,
) -> None:
    indexed = summary.set_index("scenario").reindex(SCENARIOS)
    y = np.arange(len(SCENARIOS))
    means = indexed[mean_column].to_numpy(dtype=float)
    errors = 1.96 * indexed[se_column].to_numpy(dtype=float)
    colors = [COLORS[scenario] for scenario in SCENARIOS]
    for position, mean, error, color in zip(y, means, errors, colors):
        ax.errorbar(
            mean,
            position,
            xerr=error,
            fmt="none",
            ecolor=color,
            capsize=3,
        )
    ax.scatter(means, y, c=colors, s=65, zorder=3)
    ax.set_yticks(y, [LABELS[scenario] for scenario in SCENARIOS])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)


def plot_summary(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    lambdas = sorted(metrics["lambda_fuse"].unique())
    counts = pd.crosstab(metrics["scenario"], metrics["lambda_fuse"]).reindex(
        index=SCENARIOS, columns=lambdas, fill_value=0
    )
    left = np.zeros(len(SCENARIOS))
    for index, value in enumerate(lambdas):
        width = counts[value].to_numpy(dtype=float)
        axes[0, 0].barh(
            [LABELS[scenario] for scenario in SCENARIOS],
            width,
            left=left,
            label=f"{value:g}",
            color=plt.cm.viridis(index / max(1, len(lambdas) - 1)),
        )
        left += width
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Datasets")
    axes[0, 0].set_title("A. Lambda selected by minimum BIC")
    axes[0, 0].legend(title="lambda", fontsize=8, ncol=3)

    _mean_panel(
        axes[0, 1], summary, "c_td_test_mean", "c_td_test_se", "Independent-test Ctd"
    )
    axes[0, 1].set_title("B. Prediction")
    _mean_panel(
        axes[1, 0], summary, "rmise_mean", "rmise_se", "Coefficient RMISE"
    )
    axes[1, 0].set_title("C. Coefficient recovery")

    score_summary = summary.loc[summary["scenario"] != "no_change"].set_index(
        "scenario"
    ).reindex(SCENARIOS[:-1])
    x = np.arange(len(SCENARIOS) - 1)
    width = 0.24
    for offset, metric, label in (
        (-width, "precision", "Precision"),
        (0.0, "recall", "Recall"),
        (width, "f1", "F1"),
    ):
        axes[1, 1].bar(x + offset, score_summary[metric], width, label=label)
    axes[1, 1].set_xticks(x, [LABELS[s] for s in SCENARIOS[:-1]])
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylabel("Micro-averaged score")
    axes[1, 1].set_title("D. Change-point recovery")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)
    no_change = summary.loc[summary["scenario"] == "no_change"]
    if not no_change.empty:
        axes[1, 1].text(
            0.02,
            0.03,
            f"No-change false positives: {int(no_change.iloc[0]['false_positive'])}",
            transform=axes[1, 1].transAxes,
        )

    fig.suptitle("Full-data BIC selection", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    metrics = build_selected_metrics(
        pd.read_csv(args.selected), z_tolerance=args.z_tolerance
    )
    summary = summarize_metrics(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "bic_selected_truth_metrics.csv", index=False)
    summary.to_csv(
        args.output_dir / "bic_selected_truth_summary_by_scenario.csv", index=False
    )
    plot_summary(metrics, summary, args.output_dir / "bic_selected_performance.png")
    print(f"Saved BIC-selected truth-based evaluation to: {args.output_dir}")


if __name__ == "__main__":
    main()
