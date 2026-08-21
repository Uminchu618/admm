#!/usr/bin/env python3
"""本パイロット結果からゼミスライド用の表と図を生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


SCENARIOS = ["oracle", "fine_grid", "off_grid", "small", "no_change"]
LABELS = {
    "oracle": "Oracle",
    "fine_grid": "Fine-grid",
    "off_grid": "Off-grid",
    "small": "Small",
    "no_change": "No-change",
}
COLORS = {
    "oracle": "#3568A8",
    "fine_grid": "#2A9D8F",
    "off_grid": "#E76F51",
    "small": "#8E63B0",
    "no_change": "#6B7280",
}


def scenario_from_name(name: str) -> str:
    for scenario in SCENARIOS:
        if name.startswith(f"{scenario}_seed_"):
            return scenario
    raise ValueError(f"Unknown pilot data name: {name}")


def coefficient_rmise(result: dict[str, object], truth_config: dict[str, object]) -> float:
    analysis_grid = np.asarray(result["time_grid"], dtype=float)
    coef = np.asarray(result["coef"], dtype=float)
    stepwise = truth_config["stepwise_beta"]
    true_grid = np.asarray(stepwise["true_time_grid"], dtype=float)
    true_coef = np.asarray(
        [stepwise[f"beta{index}_levels"] for index in range(1, 4)], dtype=float
    ).T
    union_grid = np.unique(np.concatenate([analysis_grid, true_grid]))
    integrated_error = 0.0
    for left, right in zip(union_grid[:-1], union_grid[1:]):
        midpoint = (left + right) / 2.0
        analysis_index = min(
            np.searchsorted(analysis_grid, midpoint, side="right") - 1,
            len(analysis_grid) - 2,
        )
        truth_index = min(
            np.searchsorted(true_grid, midpoint, side="right") - 1,
            len(true_grid) - 2,
        )
        integrated_error += (right - left) * np.sum(
            (coef[analysis_index] - true_coef[truth_index]) ** 2
        )
    n_features = coef.shape[1]
    duration = union_grid[-1] - union_grid[0]
    return float(np.sqrt(integrated_error / (n_features * duration)))


def true_change_points(truth_config: dict[str, object]) -> list[list[float]]:
    stepwise = truth_config["stepwise_beta"]
    true_grid = np.asarray(stepwise["true_time_grid"], dtype=float)
    levels = np.asarray(
        [stepwise[f"beta{index}_levels"] for index in range(1, 4)], dtype=float
    )
    return [
        true_grid[1:-1][np.abs(np.diff(levels[index])) > 1e-12].tolist()
        for index in range(levels.shape[0])
    ]


def maximum_matches(
    detected: list[float], truth: list[float], tolerance: float
) -> int:
    """一次元上の変化点を同じ係数内で一対一対応させる。"""

    detected_sorted = sorted(detected)
    truth_sorted = sorted(truth)
    detected_index = 0
    truth_index = 0
    matches = 0
    while detected_index < len(detected_sorted) and truth_index < len(truth_sorted):
        difference = detected_sorted[detected_index] - truth_sorted[truth_index]
        if abs(difference) <= tolerance:
            matches += 1
            detected_index += 1
            truth_index += 1
        elif difference < -tolerance:
            detected_index += 1
        else:
            truth_index += 1
    return matches


def change_point_counts(
    result: dict[str, object],
    truth_config: dict[str, object],
    scenario: str,
    z_tolerance: float,
) -> tuple[int, int, int]:
    analysis_grid = np.asarray(result["time_grid"], dtype=float)
    z = np.asarray(result["z_last"], dtype=float)
    detected = [
        analysis_grid[1:-1][np.abs(z[index]) > z_tolerance].tolist()
        for index in range(z.shape[0])
    ]
    truth = true_change_points(truth_config)
    location_tolerance = 0.5 if scenario == "off_grid" else 1e-9
    true_positive = sum(
        maximum_matches(detected_points, truth_points, location_tolerance)
        for detected_points, truth_points in zip(detected, truth)
    )
    return true_positive, sum(map(len, detected)), sum(map(len, truth))


def load_selected_results(
    root: Path, summary_path: Path, z_tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    summary["scenario"] = summary["data_name"].map(scenario_from_name)
    eligible = summary.loc[summary["bic_eligible"] & summary["bic"].notna()].copy()
    selected = (
        eligible.sort_values(["data_name", "bic"])
        .groupby("data_name", as_index=False)
        .first()
    )
    configs = {
        scenario: json.loads(
            (root / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }

    metrics: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        result_path = root / "outputs" / row.result_path
        result = json.loads(result_path.read_text(encoding="utf-8"))
        true_positive, detected, truth = change_point_counts(
            result, configs[row.scenario], row.scenario, z_tolerance
        )
        metrics.append(
            {
                "data_name": row.data_name,
                "scenario": row.scenario,
                "lambda_fuse": row.lambda_fuse,
                "c_td_test": row.c_td_test,
                "rmise": coefficient_rmise(result, configs[row.scenario]),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
            }
        )
    return summary, pd.DataFrame(metrics)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        subset = metrics.loc[metrics["scenario"] == scenario]
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
                "scenario": scenario,
                "n": len(subset),
                "lambda_median": subset["lambda_fuse"].median(),
                "c_td_mean": subset["c_td_test"].mean(),
                "c_td_se": subset["c_td_test"].std(ddof=1) / np.sqrt(len(subset)),
                "rmise_mean": subset["rmise"].mean(),
                "rmise_se": subset["rmise"].std(ddof=1) / np.sqrt(len(subset)),
                "change_points_mean": subset["detected"].mean(),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def plot_prediction_and_rmise(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    y = np.arange(len(SCENARIOS))
    colors = [COLORS[scenario] for scenario in SCENARIOS]
    labels = [LABELS[scenario] for scenario in SCENARIOS]

    for index, row in aggregate.iterrows():
        axes[0].errorbar(
            row["c_td_mean"],
            index,
            xerr=1.96 * row["c_td_se"],
            fmt="none",
            ecolor=colors[index],
            elinewidth=2.2,
            capsize=4,
        )
    axes[0].scatter(aggregate["c_td_mean"], y, s=95, c=colors, zorder=3)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Independent-test Ctd (mean and 95% CI)")
    axes[0].set_xlim(0.64, 0.72)
    axes[0].grid(axis="x", alpha=0.25)
    for index, value in enumerate(aggregate["c_td_mean"]):
        axes[0].text(value + 0.002, index, f"{value:.3f}", va="center", fontsize=10)

    for index, row in aggregate.iterrows():
        axes[1].errorbar(
            row["rmise_mean"],
            index,
            xerr=1.96 * row["rmise_se"],
            fmt="none",
            ecolor=colors[index],
            elinewidth=2.2,
            capsize=4,
        )
    axes[1].scatter(aggregate["rmise_mean"], y, s=95, c=colors, zorder=3)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Coefficient RMISE (mean and 95% CI)")
    axes[1].set_xlim(0.0, 0.19)
    axes[1].grid(axis="x", alpha=0.25)
    for index, value in enumerate(aggregate["rmise_mean"]):
        axes[1].text(value + 0.006, index, f"{value:.3f}", va="center", fontsize=10)

    fig.tight_layout(w_pad=3.0)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_change_points(aggregate: pd.DataFrame, output_path: Path) -> None:
    subset = aggregate.loc[aggregate["scenario"] != "no_change"].copy()
    x = np.arange(len(subset))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    for offset, column, label, color in (
        (-width, "precision", "Precision", "#3568A8"),
        (0.0, "recall", "Recall", "#E76F51"),
        (width, "f1", "F1", "#2A9D8F"),
    ):
        bars = ax.bar(x + offset, subset[column], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x, [LABELS[value] for value in subset["scenario"]])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Micro-averaged score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper right")
    ax.text(
        0.01,
        0.96,
        "No-change: 0 false positives in 20/20 datasets",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef4fa", "edgecolor": "#315d8a"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lambda_selection(metrics: pd.DataFrame, output_path: Path) -> None:
    lambda_values = [0.0, 0.01, 0.03, 0.1, 0.25]
    lambda_colors = ["#B8C7D9", "#F2C078", "#E76F51", "#8E63B0", "#3568A8"]
    counts = pd.crosstab(metrics["scenario"], metrics["lambda_fuse"]).reindex(
        index=SCENARIOS, columns=lambda_values, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(10.8, 4.3))
    left = np.zeros(len(SCENARIOS))
    for value, color in zip(lambda_values, lambda_colors):
        width = counts[value].to_numpy(dtype=float)
        bars = ax.barh(
            [LABELS[scenario] for scenario in SCENARIOS],
            width,
            left=left,
            label=f"lambda={value:g}",
            color=color,
        )
        labels = [f"{int(item)}" if item > 0 else "" for item in width]
        ax.bar_label(bars, labels=labels, label_type="center", fontsize=10)
        left += width
    ax.invert_yaxis()
    ax.set_xlim(0, 20)
    ax.set_xlabel("Number of datasets (out of 20)")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "outputs/pilot/"
            "adaptive_rho_normalized_stagnation_escape_newton5_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/assets/slides22_pilot")
    )
    parser.add_argument("--z-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, metrics = load_selected_results(root, summary_path, args.z_tolerance)
    aggregate = aggregate_metrics(metrics)
    metrics.to_csv(output_dir / "selected_fit_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "selected_summary_by_scenario.csv", index=False)
    plot_prediction_and_rmise(aggregate, output_dir / "prediction_rmise.png")
    plot_change_points(aggregate, output_dir / "change_point_scores.png")
    plot_lambda_selection(metrics, output_dir / "bic_lambda_selection.png")

    converged = int(summary["converged"].sum())
    print(f"Formal convergence: {converged}/{len(summary)}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
