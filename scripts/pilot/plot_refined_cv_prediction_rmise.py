#!/usr/bin/env python3
"""局所fine-grid CV選択lambdaでの独立評価CtdとRMISEを作図する。"""

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

from scripts.pilot.prepare_slides22_results import (
    LABELS,
    SCENARIOS,
    coefficient_rmise,
)


METHODS = ("bic", "refined_cv")
METHOD_LABELS = {"bic": "Previous: BIC", "refined_cv": "Current: local CV"}
METHOD_COLORS = {"bic": "#6B7280", "refined_cv": "#E76F51"}
METHOD_MARKERS = {"bic": "s", "refined_cv": "o"}
METHOD_OFFSETS = {"bic": -0.13, "refined_cv": 0.13}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def _scenario(data_name: str) -> str:
    for scenario in SCENARIOS:
        if data_name.startswith(f"{scenario}_seed_"):
            return scenario
    raise ValueError(f"unknown pilot data name: {data_name}")


def prepare_records(
    selections: pd.DataFrame,
    refits: pd.DataFrame,
    *,
    lambda_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """選択lambdaとrefitを突合し、解析対象と除外監査表を返す。"""

    required_selections = {"data_name", "selected_lambda", "n_folds"}
    required_refits = {
        "data_name",
        "lambda_fuse",
        "c_td_test",
        "converged",
        "result_path",
    }
    missing_selections = sorted(required_selections - set(selections.columns))
    missing_refits = sorted(required_refits - set(refits.columns))
    if missing_selections or missing_refits:
        raise ValueError(
            f"Missing columns: selections={missing_selections}, refits={missing_refits}"
        )
    if selections["data_name"].duplicated().any():
        raise ValueError("selections must contain one row per data_name")
    if refits["data_name"].duplicated().any():
        raise ValueError("refits must contain one row per data_name")

    merged = selections.merge(
        refits,
        on="data_name",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    merged["scenario"] = merged["data_name"].map(_scenario)
    merged["converged"] = _as_bool(merged["converged"])
    merged["lambda_matches"] = np.isclose(
        pd.to_numeric(merged["selected_lambda"], errors="coerce"),
        pd.to_numeric(merged["lambda_fuse"], errors="coerce"),
        rtol=0.0,
        atol=lambda_tolerance,
    )
    merged["exclusion_reason"] = ""
    merged.loc[merged["_merge"] != "both", "exclusion_reason"] = "missing_pair"
    merged.loc[
        (merged["_merge"] == "both") & ~merged["lambda_matches"],
        "exclusion_reason",
    ] = "lambda_mismatch"
    merged.loc[
        (merged["_merge"] == "both")
        & merged["lambda_matches"]
        & ~merged["converged"],
        "exclusion_reason",
    ] = "not_converged"

    if not (
        pd.to_numeric(merged.loc[merged["_merge"] == "both", "n_folds"])
        == 5
    ).all():
        raise ValueError("all selections must come from 5-fold CV")

    usable = merged.loc[merged["exclusion_reason"] == ""].copy()
    truths = {
        scenario: json.loads(
            (ROOT / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }
    rmise_values = []
    for row in usable.itertuples(index=False):
        result_path = Path(str(row.result_path))
        if not result_path.is_absolute():
            result_path = ROOT / "outputs" / result_path
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rmise_values.append(coefficient_rmise(result, truths[row.scenario]))
    usable["rmise"] = rmise_values

    audit_columns = [
        "data_name",
        "scenario",
        "selected_lambda",
        "lambda_fuse",
        "converged",
        "lambda_matches",
        "exclusion_reason",
    ]
    return usable, merged[audit_columns]


def prepare_bic_records(
    bic_metrics: pd.DataFrame, data_names: set[str]
) -> pd.DataFrame:
    """CV側の解析対象と同じデータセットにBIC結果を限定する。"""

    required = {"data_name", "scenario", "c_td_test", "rmise"}
    missing = sorted(required - set(bic_metrics.columns))
    if missing:
        raise ValueError(f"Missing BIC columns: {missing}")
    if bic_metrics["data_name"].duplicated().any():
        raise ValueError("BIC metrics must contain one row per data_name")
    bic = bic_metrics.loc[bic_metrics["data_name"].isin(data_names)].copy()
    missing_names = sorted(data_names - set(bic["data_name"]))
    if missing_names:
        raise ValueError(f"Missing BIC records for: {missing_names}")
    return bic


def summarize(records: pd.DataFrame, *, method: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario in SCENARIOS:
        subset = records.loc[records["scenario"] == scenario]
        n = len(subset)
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "n": n,
                "c_td_mean": subset["c_td_test"].mean(),
                "c_td_se": subset["c_td_test"].std(ddof=1) / np.sqrt(n),
                "rmise_mean": subset["rmise"].mean(),
                "rmise_se": subset["rmise"].std(ddof=1) / np.sqrt(n),
            }
        )
    return pd.DataFrame(rows)


def plot_prediction_and_rmise(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    y = np.arange(len(SCENARIOS))
    labels = [LABELS[scenario] for scenario in SCENARIOS]

    for method in METHODS:
        subset = summary.loc[summary["method"] == method].set_index("scenario")
        subset = subset.loc[SCENARIOS]
        y_method = y + METHOD_OFFSETS[method]
        axes[0].errorbar(
            subset["c_td_mean"],
            y_method,
            xerr=1.96 * subset["c_td_se"],
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            markerfacecolor=(
                "white" if method == "bic" else METHOD_COLORS[method]
            ),
            markeredgewidth=2.0,
            markersize=8.5,
            elinewidth=2.2,
            capsize=4,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for index, value in enumerate(subset["c_td_mean"]):
            axes[0].text(
                value + 0.0015,
                y_method[index],
                f"{value:.3f}",
                va="center",
                fontsize=11,
            )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Independent-test Ctd (mean and 95% CI)")
    axes[0].set_xlim(0.64, 0.72)
    axes[0].grid(axis="x", alpha=0.25)

    for method in METHODS:
        subset = summary.loc[summary["method"] == method].set_index("scenario")
        subset = subset.loc[SCENARIOS]
        y_method = y + METHOD_OFFSETS[method]
        axes[1].errorbar(
            subset["rmise_mean"],
            y_method,
            xerr=1.96 * subset["rmise_se"],
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            markerfacecolor=(
                "white" if method == "bic" else METHOD_COLORS[method]
            ),
            markeredgewidth=2.0,
            markersize=8.5,
            elinewidth=2.2,
            capsize=4,
            zorder=3,
        )
        for index, value in enumerate(subset["rmise_mean"]):
            axes[1].text(
                value + 0.006,
                y_method[index],
                f"{value:.3f}",
                va="center",
                fontsize=11,
            )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Coefficient RMISE (mean and 95% CI)")
    axes[1].set_xlim(0.0, 0.24)
    axes[1].grid(axis="x", alpha=0.25)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.label.set_size(14)
    fig.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-selections", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument("--bic-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    records, audit = prepare_records(
        pd.read_csv(args.cv_selections), pd.read_csv(args.refit_summary)
    )
    bic_records = prepare_bic_records(
        pd.read_csv(args.bic_metrics), set(records["data_name"])
    )
    summary = pd.concat(
        [
            summarize(bic_records, method="bic"),
            summarize(records, method="refined_cv"),
        ],
        ignore_index=True,
    )
    plot_prediction_and_rmise(summary, args.output)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_output, index=False)
    print(f"Saved {len(records)} CV-selected converged refits to: {args.output}")


if __name__ == "__main__":
    main()
