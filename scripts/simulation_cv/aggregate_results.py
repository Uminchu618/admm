#!/usr/bin/env python3
"""シミュレーションの dataset × lambda × fold 結果を集計する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.aggregate_results import (  # noqa: E402
    collect_results,
    mark_selected_lambda,
    selection_payload,
    summarize_by_lambda,
)


def aggregate_simulation_cv(
    base_dir: Path,
    *,
    n_folds: int = 5,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """データセットごとに CV 選択し、選択レコードを結合して返す。"""

    selection_rows: list[dict[str, object]] = []
    dataset_dirs = sorted(path for path in base_dir.iterdir() if path.is_dir())
    for dataset_dir in dataset_dirs:
        fold_df = collect_results(dataset_dir)
        if fold_df.empty:
            continue
        summary = mark_selected_lambda(
            summarize_by_lambda(fold_df, expected_n_folds=n_folds),
            tie_tolerance=tie_tolerance,
        )
        fold_df.to_csv(dataset_dir / "fold_results.csv", index=False)
        summary.to_csv(dataset_dir / "summary_by_lambda.csv", index=False)
        payload = selection_payload(
            summary,
            base_dir=dataset_dir,
            tie_tolerance=tie_tolerance,
        )
        payload["data_name"] = dataset_dir.name
        (dataset_dir / "selected_lambda.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        selection_rows.append(payload)

    if not selection_rows:
        raise FileNotFoundError(f"No CV result.json files found under {base_dir}")
    return pd.DataFrame(selection_rows).sort_values("data_name").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    selections = aggregate_simulation_cv(
        args.base_dir,
        n_folds=args.n_folds,
        tie_tolerance=args.tie_tolerance,
    )
    output = args.output or (args.base_dir / "cv_selections.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output, index=False)
    print(f"Saved CV selections to: {output}")
    print(selections[["data_name", "selected_lambda", "mean_c_td", "n_folds"]])


if __name__ == "__main__":
    main()
