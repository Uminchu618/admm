#!/usr/bin/env python3
"""既存CVと局所fine-grid追加結果を統合し、データセットごとにlambdaを選ぶ。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.aggregate_results import (  # noqa: E402
    _as_bool,
    collect_results,
    mark_selected_lambda,
    selection_payload,
    summarize_by_lambda,
)


def _lambda_key(value: float) -> str:
    return f"{float(value):.15g}"


def collect_dataset_results(
    data_name: str,
    *,
    coarse_base_dir: Path,
    additions_base_dir: Path,
) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    for priority, (source, base_dir) in enumerate(
        (("coarse", coarse_base_dir), ("refined_addition", additions_base_dir))
    ):
        dataset_dir = base_dir / data_name
        if not dataset_dir.is_dir():
            continue
        frame = collect_results(dataset_dir)
        if frame.empty:
            continue
        frame["result_source"] = source
        frame["source_priority"] = priority
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), 0

    data = pd.concat(frames, ignore_index=True)
    data["lambda_key"] = pd.to_numeric(data["lambda_fuse"], errors="raise").map(
        _lambda_key
    )
    data["fold"] = pd.to_numeric(data["fold"], errors="raise").astype(int)
    duplicate_count = int(data.duplicated(["lambda_key", "fold"], keep=False).sum())
    data = (
        data.sort_values("source_priority")
        .drop_duplicates(["lambda_key", "fold"], keep="last")
        .drop(columns=["source_priority"])
    )
    return data, duplicate_count


def filter_to_local_grid(results: pd.DataFrame, local_grid: pd.DataFrame) -> pd.DataFrame:
    keys = set(local_grid["lambda_fuse"].astype(float).map(_lambda_key))
    selected = results.loc[results["lambda_key"].isin(keys)].copy()
    return selected.sort_values(["lambda_fuse", "fold"]).reset_index(drop=True)


def aggregate_refined_cv(
    *,
    coarse_base_dir: Path,
    additions_base_dir: Path,
    grid_path: Path,
    output_dir: Path,
    n_folds: int,
    tie_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_table = pd.read_csv(grid_path)
    required = {
        "data_name",
        "coarse_selected_lambda",
        "grid_index",
        "lambda_fuse",
        "is_local_boundary",
    }
    missing = sorted(required - set(grid_table.columns))
    if missing:
        raise ValueError(f"refined grid is missing columns: {missing}")

    selections: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    failures: list[str] = []
    for data_name, local_grid in grid_table.groupby("data_name", sort=True):
        results, duplicate_count = collect_dataset_results(
            data_name,
            coarse_base_dir=coarse_base_dir,
            additions_base_dir=additions_base_dir,
        )
        fold_results = filter_to_local_grid(results, local_grid)
        dataset_output = output_dir / data_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        fold_results.to_csv(dataset_output / "fold_results.csv", index=False)

        expected_results = int(len(local_grid) * n_folds)
        audit: dict[str, object] = {
            "data_name": data_name,
            "coarse_selected_lambda": float(
                local_grid["coarse_selected_lambda"].iloc[0]
            ),
            "n_local_lambda": int(len(local_grid)),
            "expected_results": expected_results,
            "observed_results": int(len(fold_results)),
            "missing_results": expected_results - int(len(fold_results)),
            "duplicate_input_rows": duplicate_count,
        }
        if fold_results.empty:
            audit.update(
                {
                    "eligible_lambdas": 0,
                    "selected_lambda": np.nan,
                    "selected_grid_index": np.nan,
                    "selected_at_local_boundary": False,
                    "status": "no_results",
                }
            )
            failures.append(data_name)
            audits.append(audit)
            continue

        summary = mark_selected_lambda(
            summarize_by_lambda(fold_results, expected_n_folds=n_folds),
            tie_tolerance=tie_tolerance,
        )
        summary.to_csv(dataset_output / "summary_by_lambda.csv", index=False)
        eligible_lambdas = int(_as_bool(summary["cv_eligible"]).sum())
        grid_complete = (
            len(fold_results) == expected_results
            and eligible_lambdas == len(local_grid)
        )
        if not grid_complete:
            audit.update(
                {
                    "eligible_lambdas": eligible_lambdas,
                    "selected_lambda": np.nan,
                    "selected_grid_index": np.nan,
                    "selected_at_local_boundary": False,
                    "status": "incomplete_or_ineligible_grid",
                }
            )
            failures.append(data_name)
            audits.append(audit)
            continue
        selected_rows = summary.loc[_as_bool(summary["selected"])]
        if len(selected_rows) != 1:
            audit.update(
                {
                    "eligible_lambdas": eligible_lambdas,
                    "selected_lambda": np.nan,
                    "selected_grid_index": np.nan,
                    "selected_at_local_boundary": False,
                    "status": "selection_failed",
                }
            )
            failures.append(data_name)
            audits.append(audit)
            continue

        payload = selection_payload(
            summary, base_dir=dataset_output, tie_tolerance=tie_tolerance
        )
        selected_lambda = float(payload["selected_lambda"])
        grid_match = local_grid.loc[
            local_grid["lambda_fuse"].astype(float).map(_lambda_key)
            == _lambda_key(selected_lambda)
        ]
        if len(grid_match) != 1:
            raise ValueError(f"selected lambda not found in local grid for {data_name}")
        grid_row = grid_match.iloc[0]
        payload.update(
            {
                "data_name": data_name,
                "coarse_selected_lambda": float(
                    local_grid["coarse_selected_lambda"].iloc[0]
                ),
                "selected_grid_index": int(grid_row["grid_index"]),
                "selected_at_local_boundary": bool(grid_row["is_local_boundary"]),
            }
        )
        (dataset_output / "selected_lambda.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        selections.append(payload)
        audit.update(
            {
                "eligible_lambdas": eligible_lambdas,
                "selected_lambda": selected_lambda,
                "selected_grid_index": int(grid_row["grid_index"]),
                "selected_at_local_boundary": bool(grid_row["is_local_boundary"]),
                "status": "selected",
            }
        )
        audits.append(audit)

    selection_table = pd.DataFrame(selections)
    if not selection_table.empty:
        selection_table = selection_table.sort_values("data_name")
    audit_table = pd.DataFrame(audits).sort_values("data_name")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_table.to_csv(output_dir / "cv_selections.csv", index=False)
    audit_table.to_csv(output_dir / "refined_cv_audit.csv", index=False)
    if failures:
        raise RuntimeError(
            "fine-grid CV selection failed for datasets: " + ", ".join(failures)
        )
    return selection_table, audit_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-base-dir", type=Path, required=True)
    parser.add_argument("--additions-base-dir", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    selections, audit = aggregate_refined_cv(
        coarse_base_dir=args.coarse_base_dir,
        additions_base_dir=args.additions_base_dir,
        grid_path=args.grid,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        tie_tolerance=args.tie_tolerance,
    )
    print(f"Selected lambda for {len(selections)} datasets")
    print(
        f"Selections at a local-grid boundary: "
        f"{int(audit['selected_at_local_boundary'].sum())}/{len(audit)}"
    )


if __name__ == "__main__":
    main()
