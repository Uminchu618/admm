#!/usr/bin/env python3
"""実データ CV の qsub 実行結果を fold 別・lambda 別に集計する。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _parse_path_metadata(result_path: Path) -> tuple[float | None, int | None]:
    """lambda_x/fold_yy/result.json から補助情報を読む。"""

    lambda_value = None
    fold = None
    for part in result_path.parts:
        if part.startswith("lambda_"):
            try:
                lambda_value = float(part.replace("lambda_", "", 1))
            except ValueError:
                pass
        if part.startswith("fold_"):
            try:
                fold = int(part.replace("fold_", "", 1))
            except ValueError:
                pass
    return lambda_value, fold


def _parse_dataset(result_path: Path) -> str | None:
    """outputs/real_cv/{dataset}/... から dataset 名を読む。"""

    parts = list(result_path.parts)
    if "real_cv" not in parts:
        return None
    idx = parts.index("real_cv")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def collect_results(base_dir: Path) -> pd.DataFrame:
    """base_dir 以下の result.json を DataFrame にする。"""

    rows: list[dict[str, Any]] = []
    for result_path in sorted(base_dir.rglob("result.json")):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except Exception as exc:
            print(f"Warning: failed to load {result_path}: {exc}", file=sys.stderr)
            continue

        path_lambda, path_fold = _parse_path_metadata(result_path)
        path_dataset = _parse_dataset(result_path)
        summary = result.get("summary", {})
        config = result.get("config", {})
        lambda_fuse = config.get("lambda_fuse", path_lambda)

        c_td_test = summary.get("c_td_test")
        if c_td_test is None:
            c_td_test = summary.get("c_td_eval", summary.get("c_td"))

        rows.append(
            {
                "dataset": result.get("dataset", path_dataset),
                "lambda_fuse": lambda_fuse,
                "fold": path_fold,
                "n_train": result.get("n_samples"),
                "n_test": result.get("n_eval_samples"),
                "n_features": result.get("n_features"),
                "c_td_train": summary.get("c_td_train"),
                "c_td_test": c_td_test,
                "objective_last": summary.get("objective_last"),
                "neg_loglik_last": summary.get("neg_loglik_last"),
                "primal_residual_last": summary.get("primal_residual_last"),
                "dual_residual_last": summary.get("dual_residual_last"),
                "stopping_reason": summary.get("stopping_reason"),
                "n_admm_iter": summary.get("n_admm_iter"),
                "result_path": str(result_path),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_lambda(fold_df: pd.DataFrame) -> pd.DataFrame:
    """fold 結果から lambda 別の平均・標準偏差を作る。"""

    if fold_df.empty:
        return fold_df

    grouped = (
        fold_df.groupby("lambda_fuse", dropna=False)
        .agg(
            n_folds=("fold", "count"),
            c_td_test_mean=("c_td_test", "mean"),
            c_td_test_std=("c_td_test", "std"),
            c_td_train_mean=("c_td_train", "mean"),
            c_td_train_std=("c_td_train", "std"),
            objective_last_mean=("objective_last", "mean"),
            primal_residual_last_mean=("primal_residual_last", "mean"),
            dual_residual_last_mean=("dual_residual_last", "mean"),
        )
        .reset_index()
    )
    grouped["c_td_test_se"] = grouped.apply(
        lambda row: (
            float(row["c_td_test_std"]) / math.sqrt(float(row["n_folds"]))
            if row["n_folds"] and pd.notna(row["c_td_test_std"])
            else None
        ),
        axis=1,
    )
    return grouped.sort_values("c_td_test_mean", ascending=False).reset_index(
        drop=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate real-data CV results")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/real_cv/support2/support2_5fold_seed1234"),
        help="Experiment directory containing lambda_*/fold_*/result.json.",
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=None,
        help="Path to write fold-level CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Path to write lambda-level summary CSV.",
    )
    args = parser.parse_args()

    fold_output = args.fold_output or (args.base_dir / "fold_results.csv")
    summary_output = args.summary_output or (args.base_dir / "summary_by_lambda.csv")

    fold_df = collect_results(args.base_dir)
    if fold_df.empty:
        print(f"No result.json files found under {args.base_dir}")
        return

    summary_df = summarize_by_lambda(fold_df)

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(fold_output, index=False, encoding="utf-8")
    summary_df.to_csv(summary_output, index=False, encoding="utf-8")

    print(f"Saved fold results to: {fold_output}")
    print(f"Saved lambda summary to: {summary_output}")
    print("\n=== Top lambda by test Ctd ===")
    print(summary_df.head(10))


if __name__ == "__main__":
    main()
