#!/usr/bin/env python3
"""Pilot simulation でCV選択とBIC選択を同一指標で比較する。"""

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

from scripts.pilot.plot_bic_selected_beta import (
    LABELS,
    SCENARIOS,
    resolve_result_path,
    scenario_and_seed,
    select_bic_fits,
)
from scripts.pilot.prepare_slides22_results import (
    change_point_counts,
    coefficient_rmise,
)
from scripts.pilot.visualize_cv_results import prepare_cv_selected_records


METHODS = ("BIC", "CV")
METHOD_COLORS = {"BIC": "#3568A8", "CV": "#E76F51"}


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_truth(root: Path) -> dict[str, dict[str, object]]:
    return {
        scenario: json.loads(
            (root / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }


def add_truth_metrics(
    rows: pd.DataFrame,
    root: Path,
    truths: dict[str, dict[str, object]],
    *,
    z_tolerance: float,
) -> pd.DataFrame:
    """各選択済みfitにRMISEと変化点照合結果を付与する。"""

    required = {"data_name", "result_path", "lambda_fuse", "c_td_test", "converged"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"missing selected-fit columns: {missing}")

    records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        scenario, seed = scenario_and_seed(row.data_name)
        result_path = resolve_result_path(root, str(row.result_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        true_positive, detected, truth = change_point_counts(
            result, truths[scenario], scenario, z_tolerance
        )
        records.append(
            {
                "data_name": row.data_name,
                "scenario": scenario,
                "seed": seed,
                "lambda_fuse": float(row.lambda_fuse),
                "c_td_train": float(row.c_td_train),
                "c_td_test": float(row.c_td_test),
                "n_change_points": int(row.n_change_points),
                "converged": as_bool(row.converged),
                "result_path": str(row.result_path),
                "rmise": coefficient_rmise(result, truths[scenario]),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
            }
        )
    return pd.DataFrame(records).sort_values(["scenario", "seed"]).reset_index(drop=True)


def summarize_method(records: pd.DataFrame) -> pd.DataFrame:
    """シナリオ・選択法別に予測、係数、変化点を集計する。"""

    summary: list[dict[str, object]] = []
    for method in METHODS:
        for scenario in SCENARIOS:
            subset = records.loc[
                (records["method"] == method) & (records["scenario"] == scenario)
            ]
            n = len(subset)
            if n == 0:
                raise ValueError(f"no records for {method}/{scenario}")
            true_positive = int(subset["true_positive"].sum())
            detected = int(subset["detected"].sum())
            truth = int(subset["truth"].sum())
            precision = true_positive / detected if detected else np.nan
            recall = true_positive / truth if truth else np.nan
            f1 = (
                2 * precision * recall / (precision + recall)
                if np.isfinite(precision + recall) and precision + recall > 0
                else np.nan
            )
            summary.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "n": n,
                    "convergence_rate": subset["converged"].mean(),
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
    return pd.DataFrame(summary)


def paired_comparison(records: pd.DataFrame) -> pd.DataFrame:
    bic = records.loc[records["method"] == "BIC"].drop(columns="method")
    cv = records.loc[records["method"] == "CV"].drop(columns="method")
    paired = bic.merge(cv, on=["data_name", "scenario", "seed"], suffixes=("_bic", "_cv"), validate="one_to_one")
    for column in ("lambda_fuse", "c_td_test", "rmise", "detected", "true_positive"):
        paired[f"cv_minus_bic_{column}"] = paired[f"{column}_cv"] - paired[f"{column}_bic"]
    paired["same_lambda"] = np.isclose(
        paired["lambda_fuse_cv"], paired["lambda_fuse_bic"], rtol=0.0, atol=1e-12
    )
    return paired.sort_values(["scenario", "seed"]).reset_index(drop=True)


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        subset = paired.loc[paired["scenario"] == scenario]
        n = len(subset)
        row: dict[str, object] = {"scenario": scenario, "n": n, "same_lambda_rate": subset["same_lambda"].mean()}
        for metric in ("lambda_fuse", "c_td_test", "rmise", "detected", "true_positive"):
            values = subset[f"cv_minus_bic_{metric}"]
            row[f"cv_minus_bic_{metric}_mean"] = values.mean()
            row[f"cv_minus_bic_{metric}_se"] = values.std(ddof=1) / np.sqrt(n)
        rows.append(row)
    return pd.DataFrame(rows)


def _errorbar(ax: plt.Axes, summary: pd.DataFrame, column: str, label: str) -> None:
    y = np.arange(len(SCENARIOS))
    offsets = {"BIC": -0.13, "CV": 0.13}
    for method in METHODS:
        subset = summary.loc[summary["method"] == method].set_index("scenario").loc[SCENARIOS]
        ax.errorbar(
            subset[f"{column}_mean"],
            y + offsets[method],
            xerr=1.96 * subset[f"{column}_se"],
            fmt="o",
            color=METHOD_COLORS[method],
            capsize=3,
            label=method,
        )
    ax.set_yticks(y, [LABELS[scenario] for scenario in SCENARIOS])
    ax.invert_yaxis()
    ax.set_xlabel(label + " (mean and 95% MC CI)")
    ax.grid(axis="x", alpha=0.25)


def plot_method_comparison(summary: pd.DataFrame, paired: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _errorbar(axes[0, 0], summary, "c_td_test", "Independent-test Ctd")
    axes[0, 0].axvline(0.5, color="#6B7280", linestyle=":", linewidth=1)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("A. Predictive performance")

    _errorbar(axes[0, 1], summary, "rmise", "Coefficient RMISE")
    axes[0, 1].set_title("B. Coefficient recovery")

    score_summary = summary.loc[summary["scenario"] != "no_change"]
    x = np.arange(len(score_summary["scenario"].unique()))
    width = 0.12
    for method_index, method in enumerate(METHODS):
        subset = score_summary.loc[score_summary["method"] == method].set_index("scenario").loc[SCENARIOS[:-1]]
        for metric_index, (metric, metric_label) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))):
            axes[1, 0].bar(
                x + (method_index - 0.5) * 3 * width + (metric_index - 1) * width,
                subset[metric], width,
                color=METHOD_COLORS[method], alpha=(0.45, 0.72, 1.0)[metric_index],
                label=f"{method} {metric_label}",
            )
    axes[1, 0].set_xticks(x, [LABELS[s] for s in SCENARIOS[:-1]])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Micro-averaged score")
    axes[1, 0].set_title("C. Change-point recovery")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8, ncol=2, frameon=False)

    values = [
        paired.loc[paired["scenario"] == scenario, "cv_minus_bic_rmise"].to_numpy()
        for scenario in SCENARIOS
    ]
    boxes = axes[1, 1].boxplot(values, tick_labels=[LABELS[s] for s in SCENARIOS], showfliers=False)
    for box in boxes["boxes"]:
        box.set_color("#374151")
    axes[1, 1].axhline(0, color="#6B7280", linestyle=":", linewidth=1)
    axes[1, 1].set_ylabel("CV RMISE − BIC RMISE")
    axes[1, 1].set_title("D. Paired coefficient-error difference")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("CV-selected refit versus BIC-selected fit", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lambda_comparison(records: pd.DataFrame, output_path: Path) -> None:
    values = sorted(records["lambda_fuse"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    for ax, method in zip(axes, METHODS):
        counts = pd.crosstab(
            records.loc[records["method"] == method, "scenario"],
            records.loc[records["method"] == method, "lambda_fuse"],
        ).reindex(index=SCENARIOS, columns=values, fill_value=0)
        left = np.zeros(len(SCENARIOS))
        for value in values:
            bars = ax.barh([LABELS[s] for s in SCENARIOS], counts[value], left=left, label=f"{value:g}")
            ax.bar_label(bars, labels=[str(int(v)) if v else "" for v in counts[value]], label_type="center", fontsize=8)
            left += counts[value].to_numpy()
        ax.set_xlim(0, 20)
        ax.set_xlabel("Datasets (out of 20)")
        ax.set_title(f"{method}-selected lambda")
        ax.grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    axes[1].legend(title="lambda", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-selections", type=Path, required=True)
    parser.add_argument("--cv-refit-summary", type=Path, required=True)
    parser.add_argument("--bic-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    truths = read_truth(root)

    cv_selected = prepare_cv_selected_records(
        pd.read_csv(args.cv_selections), pd.read_csv(args.cv_refit_summary)
    )
    bic_selected = select_bic_fits(
        pd.read_csv(args.bic_summary), SCENARIOS, list(range(42, 62))
    )
    if not bic_selected["converged"].map(as_bool).all():
        raise ValueError("BIC-selected fits include nonconverged results")

    cv_metrics = add_truth_metrics(cv_selected, root, truths, z_tolerance=args.z_tolerance)
    bic_metrics = add_truth_metrics(bic_selected, root, truths, z_tolerance=args.z_tolerance)
    cv_metrics.insert(0, "method", "CV")
    bic_metrics.insert(0, "method", "BIC")
    records = pd.concat([bic_metrics, cv_metrics], ignore_index=True)
    paired_all = paired_comparison(records)
    paired = paired_all.loc[
        paired_all["converged_bic"] & paired_all["converged_cv"]
    ].copy()
    if paired.empty:
        raise ValueError("no paired converged CV/BIC fits")
    valid_data_names = set(paired["data_name"])
    valid_records = records.loc[records["data_name"].isin(valid_data_names)].copy()
    summary = summarize_method(valid_records)
    pair_summary = summarize_pairs(paired)

    records.to_csv(output_dir / "method_fit_metrics.csv", index=False)
    valid_records.to_csv(output_dir / "converged_method_fit_metrics.csv", index=False)
    summary.to_csv(output_dir / "method_summary_by_scenario.csv", index=False)
    paired_all.to_csv(output_dir / "cv_bic_paired_differences_all.csv", index=False)
    paired.to_csv(output_dir / "cv_bic_paired_differences.csv", index=False)
    pair_summary.to_csv(output_dir / "cv_bic_comparison_by_scenario.csv", index=False)
    plot_method_comparison(summary, paired, output_dir / "cv_bic_method_comparison.png")
    plot_lambda_comparison(records, output_dir / "cv_bic_lambda_selection.png")
    failed_cv = records.loc[(records["method"] == "CV") & ~records["converged"], "data_name"].tolist()
    print(
        f"Saved {len(records)} selected-fit records and {len(paired)} converged paired comparisons to: {output_dir}"
    )
    if failed_cv:
        print(f"Excluded nonconverged CV refits from primary paired comparison: {failed_cv}")


if __name__ == "__main__":
    main()
