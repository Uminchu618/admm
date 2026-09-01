#!/usr/bin/env python3
"""小規模パイロット診断が本実験へ進む条件を満たすか検査する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    """CSV由来の bool / 文字列 / 0-1 を安全に真偽値へ変換する。"""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def evaluate(summary: pd.DataFrame, expected_rows: int) -> dict[str, object]:
    required = {
        "data_name",
        "lambda_fuse",
        "converged",
        "bic_eligible",
        "bic",
        "returned_iter",
        "returned_from",
        "returned_primal_residual",
        "returned_dual_residual",
        "returned_primal_tolerance",
        "returned_dual_tolerance",
        "n_change_points",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    converged = _as_bool(summary["converged"])
    eligible = _as_bool(summary["bic_eligible"]) & summary["bic"].notna()
    residuals_ok = (
        summary["returned_primal_residual"].notna()
        & summary["returned_dual_residual"].notna()
        & summary["returned_primal_tolerance"].notna()
        & summary["returned_dual_tolerance"].notna()
        & (
            summary["returned_primal_residual"]
            <= summary["returned_primal_tolerance"]
        )
        & (
            summary["returned_dual_residual"]
            <= summary["returned_dual_tolerance"]
        )
    )
    data_has_bic = eligible.groupby(summary["data_name"]).any()
    path_counts = summary.groupby("data_name")["n_change_points"].nunique(dropna=True)

    checks = {
        "complete": len(summary) == expected_rows,
        "all_formally_converged": bool(converged.all()),
        "all_returned_residuals_within_tolerance": bool(residuals_ok.all()),
        "no_missing_returned_iter": bool(summary["returned_iter"].notna().all()),
        "no_initial_fallback": not bool(
            summary["returned_from"].fillna("").str.contains("initial").any()
        ),
        "every_dataset_has_bic_candidate": bool(data_has_bic.all()),
        "regularization_path_changes": bool((path_counts >= 2).all()),
    }
    return {
        "passed": all(checks.values()),
        "rows": int(len(summary)),
        "datasets": int(summary["data_name"].nunique()),
        "lambdas": int(summary["lambda_fuse"].nunique()),
        "converged_rows": int(converged.sum()),
        "bic_eligible_rows": int(eligible.sum()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=54)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate(pd.read_csv(args.summary), args.expected_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
