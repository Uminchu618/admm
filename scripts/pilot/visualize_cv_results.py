#!/usr/bin/env python3
"""5-fold CV 選択 lambda の再学習・独立評価結果を可視化する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = ["oracle", "fine_grid", "off_grid", "small", "no_change"]
SCENARIO_LABELS = {
    "oracle": "Oracle",
    "fine_grid": "Fine-grid",
    "off_grid": "Off-grid",
    "small": "Small",
    "no_change": "No-change",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def _scenario_seed(data_name: str) -> tuple[str, int]:
    for scenario in SCENARIO_ORDER:
        prefix = f"{scenario}_seed_"
        if data_name.startswith(prefix):
            return scenario, int(data_name.removeprefix(prefix))
    raise ValueError(f"Could not parse scenario/seed from {data_name}")


def prepare_cv_selected_records(
    selections: pd.DataFrame,
    refits: pd.DataFrame,
    *,
    lambda_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """CV選択表と独立評価refit表を1対1で結合する。"""

    selection_required = {"data_name", "selected_lambda", "mean_c_td", "n_folds"}
    refit_required = {
        "data_name",
        "lambda_fuse",
        "c_td_test",
        "c_td_train",
        "converged",
        "n_change_points",
        "result_path",
    }
    missing_selection = sorted(selection_required - set(selections.columns))
    missing_refit = sorted(refit_required - set(refits.columns))
    if missing_selection or missing_refit:
        raise ValueError(
            f"Missing columns: selections={missing_selection}, refits={missing_refit}"
        )
    if selections["data_name"].duplicated().any():
        raise ValueError("selections must contain one row per data_name")
    if refits["data_name"].duplicated().any():
        raise ValueError("refits must contain one row per data_name")

    records = selections.merge(
        refits,
        on="data_name",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not records["_merge"].eq("both").all():
        unmatched = records.loc[records["_merge"] != "both", ["data_name", "_merge"]]
        raise ValueError(f"Selection/refit mismatch:\n{unmatched.to_string(index=False)}")
    records = records.drop(columns="_merge")

    selected_lambda = pd.to_numeric(records["selected_lambda"], errors="raise")
    fitted_lambda = pd.to_numeric(records["lambda_fuse"], errors="raise")
    if not np.allclose(
        selected_lambda,
        fitted_lambda,
        rtol=0.0,
        atol=lambda_tolerance,
    ):
        raise ValueError("refit lambda does not match CV-selected lambda")
    if not (pd.to_numeric(records["n_folds"], errors="raise") == 5).all():
        raise ValueError("all selections must come from 5-fold CV")

    parsed = records["data_name"].map(_scenario_seed)
    records["scenario"] = parsed.map(lambda value: value[0])
    records["seed"] = parsed.map(lambda value: value[1])
    records["converged"] = _as_bool(records["converged"])
    records["cv_to_independent_gap"] = (
        pd.to_numeric(records["mean_c_td"], errors="coerce")
        - pd.to_numeric(records["c_td_test"], errors="coerce")
    )
    return records.sort_values(["scenario", "seed"]).reset_index(drop=True)


def summarize_selected(records: pd.DataFrame) -> pd.DataFrame:
    """scenario 別の選択lambdaと独立評価Ctdを要約する。"""

    return (
        records.groupby("scenario")
        .agg(
            n=("seed", "count"),
            convergence_rate=("converged", "mean"),
            lambda_mean=("selected_lambda", "mean"),
            lambda_median=("selected_lambda", "median"),
            cv_c_td_mean=("mean_c_td", "mean"),
            independent_c_td_mean=("c_td_test", "mean"),
            independent_c_td_sd=("c_td_test", "std"),
            cv_to_independent_gap_mean=("cv_to_independent_gap", "mean"),
            change_points_mean=("n_change_points", "mean"),
        )
        .reindex(SCENARIO_ORDER)
        .reset_index()
    )


def plot_cv_selected_diagnostics(records: pd.DataFrame, output_path: Path) -> None:
    """選択lambda・独立評価Ctd・変化点・CV楽観差を4面で描く。"""

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ax_lambda, ax_ctd, ax_cp, ax_gap = axes.ravel()
    labels = [SCENARIO_LABELS[value] for value in SCENARIO_ORDER]

    lambda_counts = pd.crosstab(records["scenario"], records["selected_lambda"])
    lambda_counts = lambda_counts.reindex(SCENARIO_ORDER, fill_value=0)
    bottoms = np.zeros(len(lambda_counts), dtype=float)
    for value in sorted(lambda_counts.columns.astype(float)):
        counts = lambda_counts[value].to_numpy(dtype=float)
        ax_lambda.bar(labels, counts, bottom=bottoms, label=f"{value:g}")
        bottoms += counts
    ax_lambda.set_title("A. Lambda selected by 5-fold mean Ctd")
    ax_lambda.set_ylabel("Dataset count")
    ax_lambda.tick_params(axis="x", rotation=25)
    ax_lambda.legend(title="lambda", fontsize="small")

    ctd_values = [
        records.loc[records["scenario"] == scenario, "c_td_test"].dropna()
        for scenario in SCENARIO_ORDER
    ]
    ax_ctd.boxplot(ctd_values, tick_labels=labels, showmeans=True)
    ax_ctd.set_title("B. Independent-test Ctd after full-data refit")
    ax_ctd.set_ylabel("Ctd")
    ax_ctd.tick_params(axis="x", rotation=25)

    cp_values = [
        records.loc[records["scenario"] == scenario, "n_change_points"].dropna()
        for scenario in SCENARIO_ORDER
    ]
    ax_cp.boxplot(cp_values, tick_labels=labels, showmeans=True)
    ax_cp.set_title("C. Change points in CV-selected refit")
    ax_cp.set_ylabel("Estimated change-point count")
    ax_cp.tick_params(axis="x", rotation=25)

    for scenario in SCENARIO_ORDER:
        subset = records.loc[records["scenario"] == scenario]
        ax_gap.scatter(
            subset["mean_c_td"],
            subset["c_td_test"],
            alpha=0.7,
            label=SCENARIO_LABELS[scenario],
        )
    finite = records[["mean_c_td", "c_td_test"]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        ax_gap.plot([low, high], [low, high], "--", color="#6B7280")
    ax_gap.set_title("D. CV Ctd versus independent-test Ctd")
    ax_gap.set_xlabel("5-fold mean validation Ctd")
    ax_gap.set_ylabel("Independent-test Ctd")
    ax_gap.legend(fontsize="small")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-selections", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = prepare_cv_selected_records(
        pd.read_csv(args.cv_selections), pd.read_csv(args.refit_summary)
    )
    summary = summarize_selected(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "cv_selected_records.csv", index=False)
    summary.to_csv(
        args.output_dir / "cv_selected_summary_by_scenario.csv", index=False
    )
    plot_cv_selected_diagnostics(
        records, args.output_dir / "pilot_cv_selected_diagnostics.png"
    )
    print(f"Saved CV-selected visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
