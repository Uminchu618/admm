#!/usr/bin/env python3
"""全学習データの lambda path から正式収束候補のBIC最小値を選ぶ。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def select_by_bic(
    summary: pd.DataFrame,
    *,
    expected_lambdas: list[float] | None = None,
    tie_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "data_name",
        "lambda_fuse",
        "bic",
        "bic_eligible",
        "converged",
        "result_path",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if summary.duplicated(["data_name", "lambda_fuse"]).any():
        raise ValueError("duplicate data_name/lambda_fuse rows")

    data = summary.copy()
    data["lambda_fuse"] = pd.to_numeric(data["lambda_fuse"], errors="raise")
    data["bic"] = pd.to_numeric(data["bic"], errors="coerce")
    data["eligible"] = (
        _as_bool(data["bic_eligible"])
        & _as_bool(data["converged"])
        & np.isfinite(data["bic"])
    )

    selected_rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    expected_keys = (
        {float(value) for value in expected_lambdas}
        if expected_lambdas is not None
        else None
    )
    for data_name, group in data.groupby("data_name", sort=True):
        observed = set(group["lambda_fuse"].astype(float))
        eligible = group.loc[group["eligible"]].copy()
        if eligible.empty:
            raise ValueError(f"no BIC-eligible lambda for {data_name}")
        minimum = float(eligible["bic"].min())
        tied = eligible.loc[(eligible["bic"] - minimum).abs() <= tie_tolerance]
        chosen = tied.sort_values("lambda_fuse", ascending=False).iloc[0].copy()
        selected_rows.append(chosen)
        audit_rows.append(
            {
                "data_name": data_name,
                "n_rows": int(len(group)),
                "n_eligible": int(len(eligible)),
                "n_ineligible": int(len(group) - len(eligible)),
                "n_missing_lambdas": (
                    int(len(expected_keys - observed))
                    if expected_keys is not None
                    else 0
                ),
                "selected_lambda": float(chosen["lambda_fuse"]),
                "selected_bic": float(chosen["bic"]),
                "n_bic_ties": int(len(tied)),
                "selected_at_grid_boundary": bool(
                    float(chosen["lambda_fuse"])
                    in {min(observed), max(observed)}
                ),
            }
        )

    selected = pd.DataFrame(selected_rows).drop(columns="eligible")
    selected = selected.sort_values("data_name").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("data_name").reset_index(drop=True)
    return selected, audit


def _write_selection_manifests(base_dir: Path, selected: pd.DataFrame) -> None:
    for row in selected.itertuples(index=False):
        payload = {
            "selection_method": "minimum_bic_full_training_data",
            "data_name": row.data_name,
            "selected_lambda": float(row.lambda_fuse),
            "bic": float(row.bic),
            "bic_eligible": True,
            "result_path": str(row.result_path),
        }
        path = base_dir / str(row.data_name) / "selected_bic.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--lambda-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-datasets", type=int, default=100)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    grid = json.loads(args.lambda_grid.read_text(encoding="utf-8"))["lambda_values"]
    selected, audit = select_by_bic(
        summary,
        expected_lambdas=[float(value) for value in grid],
        tie_tolerance=args.tie_tolerance,
    )
    if len(selected) != args.expected_datasets:
        raise ValueError(
            f"expected {args.expected_datasets} selected datasets; got {len(selected)}"
        )
    incomplete = audit.loc[audit["n_missing_lambdas"] > 0, "data_name"].tolist()
    if incomplete:
        raise ValueError(f"incomplete lambda paths: {incomplete}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    audit.to_csv(args.audit_output, index=False)
    _write_selection_manifests(args.base_dir, selected)
    print(f"Saved {len(selected)} BIC selections to: {args.output}")


if __name__ == "__main__":
    main()
