#!/usr/bin/env python3
"""Compute Cox regression metrics for comparison with ADMM lambda results.

This script fits a Cox proportional hazards model per dataset and outputs:
- c_td_cox: time-dependent C-index computed with the existing HazardAFTEvaluator
- c_index_harrell: Harrell's concordance index from Cox partial hazards

It supports both a single CSV (--data-file) and a directory batch (--data-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admm.evaluator import HazardAFTEvaluator

EXCLUDED_COLUMNS = {
    "id",
    "k",
    "time",
    "event",
    "time_true",
    "c1",
    "c2",
}


class CoxPHSurvivalAdapter:
    """Adapter that exposes ADMM-like predict API for HazardAFTEvaluator."""

    def __init__(self, fitter: CoxPHFitter, feature_cols: list[str]) -> None:
        self._fitter = fitter
        self._feature_cols = list(feature_cols)

    def _prepare_predict_X(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("X must be a 2D array for Cox prediction.")
        if x_arr.shape[1] != len(self._feature_cols):
            raise ValueError("Feature count does not match fitted Cox model.")
        return x_arr

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Iterable[float] | None = None,
    ) -> np.ndarray:
        x_arr = self._prepare_predict_X(X)
        x_df = pd.DataFrame(x_arr, columns=self._feature_cols)

        if times is None:
            survival_df = self._fitter.predict_survival_function(x_df)
            return survival_df.to_numpy(dtype=float).T

        times_arr = np.asarray(list(times), dtype=float).reshape(-1)
        if times_arr.size == 0:
            return np.zeros((x_arr.shape[0], 0), dtype=float)

        survival_df = self._fitter.predict_survival_function(
            x_df, times=times_arr.tolist()
        )
        index_arr = np.asarray(survival_df.index, dtype=float)
        survival_values = survival_df.to_numpy(dtype=float)

        # lifelines can internally adjust the time index; align back to requested times.
        if survival_values.shape[0] != times_arr.size or not np.allclose(
            index_arr, times_arr, rtol=0.0, atol=1e-12
        ):
            aligned = np.empty((times_arr.size, survival_values.shape[1]), dtype=float)
            for col_idx in range(survival_values.shape[1]):
                aligned[:, col_idx] = np.interp(
                    times_arr,
                    index_arr,
                    survival_values[:, col_idx],
                    left=float(survival_values[0, col_idx]),
                    right=float(survival_values[-1, col_idx]),
                )
            survival_values = aligned

        return survival_values.T


def _load_subject_level(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = pd.read_csv(path)
    required = {"id", "k", "time", "event"}
    if not required.issubset(data.columns):
        missing = sorted(required - set(data.columns))
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [col for col in data.columns if col not in EXCLUDED_COLUMNS]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found.")

    sorted_data = data.sort_values(["id", "k"]).reset_index(drop=True)

    grouped = sorted_data.groupby("id", sort=True)
    if (grouped["time"].nunique() > 1).any():
        raise ValueError("time must be constant across k for each id.")
    if (grouped["event"].nunique() > 1).any():
        raise ValueError("event must be constant across k for each id.")

    subject_data = grouped.first().reset_index()
    X = subject_data[feature_cols].to_numpy(dtype=float)
    y = subject_data[["time", "event"]].to_numpy(dtype=float)
    return X, y, feature_cols


def _fit_and_score_cox(path: Path) -> dict[str, float | int | str]:
    X, y, feature_cols = _load_subject_level(path)
    y_time = np.asarray(y[:, 0], dtype=float)
    y_event = np.asarray(y[:, 1], dtype=int)

    fit_df = pd.DataFrame(X, columns=feature_cols)
    fit_df["time"] = y_time
    fit_df["event"] = y_event

    cox = CoxPHFitter()
    cox.fit(fit_df, duration_col="time", event_col="event")

    adapter = CoxPHSurvivalAdapter(cox, feature_cols)
    evaluator = HazardAFTEvaluator()
    c_td = evaluator.compute_c_td(adapter, X, y)

    partial_hazard = np.asarray(
        cox.predict_partial_hazard(fit_df[feature_cols]), dtype=float
    )
    c_harrell = concordance_index(y_time, -partial_hazard, y_event)

    return {
        "data_name": path.stem,
        "data_path": str(path),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_events": int(np.sum(y_event == 1)),
        "c_td_cox": float(c_td),
        "c_index_harrell": float(c_harrell),
    }


def _build_comparison_table(
    cox_df: pd.DataFrame,
    lambda_summary_path: Path,
) -> pd.DataFrame:
    lambda_df = pd.read_csv(lambda_summary_path)
    if "data_name" not in lambda_df.columns or "c_td" not in lambda_df.columns:
        raise ValueError("lambda summary must contain data_name and c_td columns.")

    admm_summary = (
        lambda_df.groupby("data_name", as_index=False)["c_td"]
        .agg(admm_best_c_td="max", admm_median_c_td="median", admm_mean_c_td="mean")
        .reset_index(drop=True)
    )

    merged = cox_df.merge(admm_summary, on="data_name", how="left")
    merged["delta_best_admm_minus_cox"] = merged["admm_best_c_td"] - merged["c_td_cox"]
    merged["delta_median_admm_minus_cox"] = (
        merged["admm_median_c_td"] - merged["c_td_cox"]
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit CoxPH model(s) and compute c_td / Harrell C-index"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Single CSV file (long format) to process.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/extended_aft_step"),
        help="Directory containing CSV files for batch processing.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Glob pattern used with --data-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cox_summary.csv"),
        help="Output CSV path for Cox metrics.",
    )
    parser.add_argument(
        "--lambda-summary",
        type=Path,
        default=Path("outputs/lambda_summary.csv"),
        help="Lambda summary CSV used for ADMM comparison table.",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=Path("outputs/cox_vs_lambda_c_td.csv"),
        help="Output CSV path for merged ADMM vs Cox summary.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip creating merged ADMM vs Cox summary table.",
    )
    args = parser.parse_args()

    if args.data_file is not None:
        data_files = [args.data_file]
    else:
        if not args.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        data_files = sorted(args.data_dir.glob(args.glob))

    if len(data_files) == 0:
        raise FileNotFoundError("No CSV files found for Cox evaluation.")

    rows: list[dict[str, float | int | str]] = []
    errors: list[tuple[str, str]] = []

    print(f"Processing {len(data_files)} file(s)...")
    for path in data_files:
        try:
            row = _fit_and_score_cox(path)
            rows.append(row)
            print(
                f"  OK  {path.name}: c_td={row['c_td_cox']:.4f}, "
                f"harrell={row['c_index_harrell']:.4f}"
            )
        except Exception as exc:  # pragma: no cover - CLI robustness
            errors.append((path.name, str(exc)))
            print(f"  ERR {path.name}: {exc}")

    if len(rows) == 0:
        raise RuntimeError("All Cox fits failed; no summary to write.")

    result_df = pd.DataFrame(rows).sort_values("data_name").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved Cox summary to: {args.output}")

    if not args.skip_compare:
        if args.lambda_summary.exists():
            merged_df = _build_comparison_table(result_df, args.lambda_summary)
            args.compare_output.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(args.compare_output, index=False, encoding="utf-8")
            print(f"Saved ADMM vs Cox summary to: {args.compare_output}")
        else:
            print(
                f"Skipping merged table: lambda summary not found at {args.lambda_summary}"
            )

    if len(errors) > 0:
        print("\nFailed files:")
        for name, message in errors:
            print(f"- {name}: {message}")


if __name__ == "__main__":
    main()
