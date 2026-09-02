#!/usr/bin/env python3
"""粗いCVと局所fine-grid CVを同一seedで比較する。"""

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
from scripts.pilot.visualize_cv_results import prepare_cv_selected_records  # noqa: E402


METHODS = ("coarse_cv", "refined_cv")
METHOD_LABELS = {"coarse_cv": "Coarse CV", "refined_cv": "Refined CV"}
METHOD_COLORS = {"coarse_cv": "#3568A8", "refined_cv": "#E76F51"}


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _scenario_and_seed(data_name: str) -> tuple[str, int]:
    for scenario in SCENARIOS:
        prefix = f"{scenario}_seed_"
        if data_name.startswith(prefix):
            return scenario, int(data_name.removeprefix(prefix))
    raise ValueError(f"unknown pilot data name: {data_name}")


def _resolve_result_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / "outputs" / path


def _truth_configs() -> dict[str, dict[str, object]]:
    return {
        scenario: json.loads(
            (ROOT / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }


def evaluate_cv_method(
    selections: pd.DataFrame,
    refits: pd.DataFrame,
    *,
    method: str,
    z_tolerance: float,
) -> pd.DataFrame:
    records = prepare_cv_selected_records(selections, refits)
    truths = _truth_configs()
    metrics: list[dict[str, object]] = []
    for row in records.itertuples(index=False):
        scenario, seed = _scenario_and_seed(row.data_name)
        result_path = _resolve_result_path(str(row.result_path))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        true_positive, detected, truth = change_point_counts(
            result, truths[scenario], scenario, z_tolerance
        )
        metrics.append(
            {
                "method": method,
                "data_name": row.data_name,
                "scenario": scenario,
                "seed": seed,
                "lambda_fuse": float(row.selected_lambda),
                "cv_mean_c_td": float(row.mean_c_td),
                "c_td_train": float(row.c_td_train),
                "c_td_test": float(row.c_td_test),
                "converged": _as_bool(row.converged),
                "rmise": coefficient_rmise(result, truths[scenario]),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
                "result_path": str(row.result_path),
                "selected_at_local_boundary": _as_bool(
                    getattr(row, "selected_at_local_boundary", False)
                ),
            }
        )
    return pd.DataFrame(metrics)


def pair_methods(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coarse = records.loc[records["method"] == "coarse_cv"].drop(columns="method")
    refined = records.loc[records["method"] == "refined_cv"].drop(columns="method")
    paired_all = coarse.merge(
        refined,
        on=["data_name", "scenario", "seed"],
        suffixes=("_coarse", "_refined"),
        validate="one_to_one",
    )
    paired = paired_all.loc[
        paired_all["converged_coarse"] & paired_all["converged_refined"]
    ].copy()
    for metric in (
        "lambda_fuse",
        "cv_mean_c_td",
        "c_td_test",
        "rmise",
        "detected",
        "true_positive",
    ):
        paired[f"refined_minus_coarse_{metric}"] = (
            paired[f"{metric}_refined"] - paired[f"{metric}_coarse"]
        )
    paired["lambda_changed"] = ~np.isclose(
        paired["lambda_fuse_refined"],
        paired["lambda_fuse_coarse"],
        rtol=0.0,
        atol=1e-12,
    )
    return paired_all, paired


def summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        subset = paired.loc[paired["scenario"] == scenario]
        n = len(subset)
        row: dict[str, object] = {
            "scenario": scenario,
            "n": n,
            "lambda_changed_rate": subset["lambda_changed"].mean(),
            "refined_boundary_rate": subset[
                "selected_at_local_boundary_refined"
            ].mean(),
        }
        for metric in (
            "lambda_fuse",
            "cv_mean_c_td",
            "c_td_test",
            "rmise",
            "detected",
            "true_positive",
        ):
            values = subset[f"refined_minus_coarse_{metric}"]
            row[f"delta_{metric}_mean"] = values.mean()
            row[f"delta_{metric}_se"] = values.std(ddof=1) / np.sqrt(n)
        rows.append(row)
    return pd.DataFrame(rows)


def _micro_scores(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for scenario in SCENARIOS[:-1]:
            subset = records.loc[
                (records["method"] == method) & (records["scenario"] == scenario)
            ]
            tp = int(subset["true_positive"].sum())
            detected = int(subset["detected"].sum())
            truth = int(subset["truth"].sum())
            precision = tp / detected if detected else np.nan
            recall = tp / truth if truth else np.nan
            f1 = (
                2 * precision * recall / (precision + recall)
                if np.isfinite(precision + recall) and precision + recall > 0
                else np.nan
            )
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(records: pd.DataFrame, paired: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for scenario in SCENARIOS:
        subset = paired.loc[paired["scenario"] == scenario]
        axes[0, 0].scatter(
            subset["lambda_fuse_coarse"],
            subset["lambda_fuse_refined"],
            label=LABELS[scenario],
            color=COLORS[scenario],
            alpha=0.75,
        )
    limits = [0.0, max(0.75, float(paired[["lambda_fuse_coarse", "lambda_fuse_refined"]].max().max()))]
    axes[0, 0].plot(limits, limits, linestyle=":", color="#6B7280")
    axes[0, 0].set_xscale("symlog", linthresh=1e-4)
    axes[0, 0].set_yscale("symlog", linthresh=1e-4)
    axes[0, 0].set_xlabel("Coarse-CV lambda")
    axes[0, 0].set_ylabel("Refined-CV lambda")
    axes[0, 0].set_title("A. Paired selected lambda")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    for ax, metric, ylabel, title in (
        (axes[0, 1], "c_td_test", "Refined − coarse Ctd", "B. Independent-test performance"),
        (axes[1, 0], "rmise", "Refined − coarse RMISE", "C. Coefficient recovery"),
    ):
        values = [
            paired.loc[paired["scenario"] == scenario, f"refined_minus_coarse_{metric}"]
            for scenario in SCENARIOS
        ]
        ax.boxplot(values, tick_labels=[LABELS[s] for s in SCENARIOS], showfliers=False)
        ax.axhline(0.0, linestyle=":", color="#6B7280")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

    valid_names = set(paired["data_name"])
    valid_records = records.loc[records["data_name"].isin(valid_names)]
    scores = _micro_scores(valid_records)
    x = np.arange(len(SCENARIOS) - 1)
    width = 0.12
    for method_index, method in enumerate(METHODS):
        subset = scores.loc[scores["method"] == method].set_index("scenario").loc[SCENARIOS[:-1]]
        for metric_index, metric in enumerate(("precision", "recall", "f1")):
            axes[1, 1].bar(
                x + (method_index - 0.5) * 3 * width + (metric_index - 1) * width,
                subset[metric],
                width,
                color=METHOD_COLORS[method],
                alpha=(0.45, 0.72, 1.0)[metric_index],
                label=f"{METHOD_LABELS[method]} {metric.title()}",
            )
    axes[1, 1].set_xticks(x, [LABELS[s] for s in SCENARIOS[:-1]])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("Micro-averaged score")
    axes[1, 1].set_title("D. Change-point recovery")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle("Local fine-grid CV versus coarse-grid CV", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-selections", type=Path, required=True)
    parser.add_argument("--coarse-refit-summary", type=Path, required=True)
    parser.add_argument("--refined-selections", type=Path, required=True)
    parser.add_argument("--refined-refit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse = evaluate_cv_method(
        pd.read_csv(args.coarse_selections),
        pd.read_csv(args.coarse_refit_summary),
        method="coarse_cv",
        z_tolerance=args.z_tolerance,
    )
    refined = evaluate_cv_method(
        pd.read_csv(args.refined_selections),
        pd.read_csv(args.refined_refit_summary),
        method="refined_cv",
        z_tolerance=args.z_tolerance,
    )
    records = pd.concat([coarse, refined], ignore_index=True)
    paired_all, paired = pair_methods(records)
    summary = summarize_paired(paired)

    records.to_csv(output_dir / "coarse_refined_fit_metrics.csv", index=False)
    paired_all.to_csv(output_dir / "coarse_refined_pairs_all.csv", index=False)
    paired.to_csv(output_dir / "coarse_refined_pairs_converged.csv", index=False)
    summary.to_csv(output_dir / "coarse_refined_summary_by_scenario.csv", index=False)
    plot_comparison(records, paired, output_dir / "coarse_refined_cv_comparison.png")
    print(f"Saved {len(paired)} paired coarse/refined CV comparisons to: {output_dir}")


if __name__ == "__main__":
    main()
