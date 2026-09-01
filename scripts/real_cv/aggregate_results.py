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


def _as_bool(series: pd.Series) -> pd.Series:
    """bool/文字列が混在した列を安全に真偽値へ変換する。"""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


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
        history = result.get("history", {})
        config = result.get("config", {})
        lambda_fuse = config.get("lambda_fuse", path_lambda)
        n_train = result.get("n_samples")
        lambda_fuse_effective = summary.get(
            "lambda_fuse_effective",
            result.get("history", {}).get("lambda_fuse_effective"),
        )
        if (
            lambda_fuse_effective is None
            and lambda_fuse is not None
            and n_train is not None
        ):
            lambda_fuse_effective = float(n_train) * float(lambda_fuse)

        c_td_test = summary.get("c_td_test")
        if c_td_test is None:
            c_td_test = summary.get("c_td_eval", summary.get("c_td"))

        rows.append(
            {
                "dataset": result.get("dataset", path_dataset),
                "lambda_fuse": lambda_fuse,
                "lambda_fuse_effective": lambda_fuse_effective,
                "fold": path_fold,
                "n_train": n_train,
                "n_test": result.get("n_eval_samples"),
                "n_features": result.get("n_features"),
                "c_td_train": summary.get("c_td_train"),
                "c_td_test": c_td_test,
                "objective_last": summary.get("objective_last"),
                "neg_loglik_last": summary.get("neg_loglik_last"),
                "primal_residual_last": summary.get("primal_residual_last"),
                "dual_residual_last": summary.get("dual_residual_last"),
                "primal_tolerance_last": summary.get("primal_tolerance_last"),
                "dual_tolerance_last": summary.get("dual_tolerance_last"),
                "stopping_reason": summary.get("stopping_reason"),
                "n_admm_iter": summary.get("n_admm_iter"),
                "returned_iter": summary.get(
                    "returned_iter", history.get("returned_iter")
                ),
                "returned_primal_residual": summary.get(
                    "returned_primal_residual",
                    history.get("returned_primal_residual"),
                ),
                "returned_dual_residual": summary.get(
                    "returned_dual_residual", history.get("returned_dual_residual")
                ),
                "returned_primal_tolerance": summary.get(
                    "returned_primal_tolerance",
                    history.get("returned_primal_tolerance"),
                ),
                "returned_dual_tolerance": summary.get(
                    "returned_dual_tolerance",
                    history.get("returned_dual_tolerance"),
                ),
                "converged": bool(
                    summary.get("converged", history.get("converged", False))
                ),
                "result_path": str(result_path),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_lambda(
    fold_df: pd.DataFrame,
    expected_n_folds: int | None = None,
) -> pd.DataFrame:
    """fold 結果を集計し、CV 選択候補としての適格性も判定する。"""

    if fold_df.empty:
        return fold_df

    data = fold_df.copy()
    data["lambda_fuse"] = pd.to_numeric(data["lambda_fuse"], errors="coerce")
    data["fold"] = pd.to_numeric(data["fold"], errors="coerce")
    data["c_td_test"] = pd.to_numeric(data["c_td_test"], errors="coerce")
    data["converged"] = _as_bool(data["converged"])
    data["finite_c_td_test"] = data["c_td_test"].map(
        lambda value: bool(pd.notna(value) and math.isfinite(float(value)))
    )

    observed_folds = sorted(data["fold"].dropna().astype(int).unique().tolist())
    if expected_n_folds is None:
        expected_n_folds = len(observed_folds)
    if expected_n_folds < 2:
        raise ValueError("expected_n_folds must be >= 2")

    grouped = (
        data.groupby("lambda_fuse", dropna=False)
        .agg(
            n_results=("fold", "size"),
            n_folds=("fold", "nunique"),
            n_converged_folds=("converged", "sum"),
            n_finite_c_td_folds=("finite_c_td_test", "sum"),
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
    grouped["n_folds_expected"] = int(expected_n_folds)
    grouped["cv_eligible"] = (
        (grouped["n_results"] == expected_n_folds)
        & (grouped["n_folds"] == expected_n_folds)
        & (grouped["n_converged_folds"] == expected_n_folds)
        & (grouped["n_finite_c_td_folds"] == expected_n_folds)
    )

    expected_fold_set = set(range(expected_n_folds))
    fold_sets = data.groupby("lambda_fuse", dropna=False)["fold"].apply(
        lambda values: set(values.dropna().astype(int).tolist())
    )
    grouped["fold_ids_complete"] = grouped["lambda_fuse"].map(fold_sets).map(
        lambda folds: folds == expected_fold_set
    )
    grouped["cv_eligible"] &= grouped["fold_ids_complete"]

    def exclusion_reason(row: pd.Series) -> str | None:
        if bool(row["cv_eligible"]):
            return None
        reasons = []
        if row["n_results"] != expected_n_folds or row["n_folds"] != expected_n_folds:
            reasons.append("incomplete_or_duplicate_folds")
        if not bool(row["fold_ids_complete"]):
            reasons.append("unexpected_fold_ids")
        if row["n_converged_folds"] != expected_n_folds:
            reasons.append("nonconverged_fold")
        if row["n_finite_c_td_folds"] != expected_n_folds:
            reasons.append("nonfinite_c_td")
        return ";".join(reasons)

    grouped["cv_exclusion_reason"] = grouped.apply(exclusion_reason, axis=1)
    grouped["c_td_test_se"] = grouped.apply(
        lambda row: (
            float(row["c_td_test_std"]) / math.sqrt(float(row["n_folds"]))
            if row["n_folds"] and pd.notna(row["c_td_test_std"])
            else None
        ),
        axis=1,
    )
    return grouped.sort_values(
        ["cv_eligible", "c_td_test_mean", "lambda_fuse"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def mark_selected_lambda(
    summary_df: pd.DataFrame,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """平均 test Ctd 最大の lambda を選び、selected 列を付与する。"""

    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")

    summary = summary_df.copy()
    summary["selected"] = False
    if summary.empty:
        return summary

    eligible = summary.loc[
        _as_bool(summary["cv_eligible"])
        & pd.to_numeric(summary["c_td_test_mean"], errors="coerce").notna()
    ].copy()
    if eligible.empty:
        return summary

    best_score = float(eligible["c_td_test_mean"].max())
    tied = eligible.loc[
        (eligible["c_td_test_mean"].astype(float) - best_score).abs()
        <= tie_tolerance
    ]
    selected_index = tied["lambda_fuse"].astype(float).idxmax()
    summary.loc[selected_index, "selected"] = True
    return summary


def selection_payload(
    selected_summary: pd.DataFrame,
    *,
    base_dir: Path,
    tie_tolerance: float,
) -> dict[str, Any]:
    """選択済み summary から後続の再学習用 JSON payload を作る。"""

    selected = selected_summary.loc[_as_bool(selected_summary["selected"])]
    if len(selected) != 1:
        raise RuntimeError(
            "CV-eligible lambda could not be selected; inspect summary_by_lambda.csv"
        )
    row = selected.iloc[0]
    return {
        "selection_method": "five_fold_cv_mean_c_td",
        "base_dir": str(base_dir),
        "selected_lambda": float(row["lambda_fuse"]),
        "mean_c_td": float(row["c_td_test_mean"]),
        "std_c_td": (
            float(row["c_td_test_std"])
            if pd.notna(row["c_td_test_std"])
            else None
        ),
        "se_c_td": (
            float(row["c_td_test_se"])
            if pd.notna(row["c_td_test_se"])
            else None
        ),
        "n_folds": int(row["n_folds"]),
        "tie_break": "largest_lambda_within_tolerance",
        "tie_tolerance": float(tie_tolerance),
    }


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
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=None,
        help="Path to write selected lambda JSON.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    fold_output = args.fold_output or (args.base_dir / "fold_results.csv")
    summary_output = args.summary_output or (args.base_dir / "summary_by_lambda.csv")
    selection_output = args.selection_output or (args.base_dir / "selected_lambda.json")

    fold_df = collect_results(args.base_dir)
    if fold_df.empty:
        print(f"No result.json files found under {args.base_dir}")
        return

    summary_df = mark_selected_lambda(
        summarize_by_lambda(fold_df, expected_n_folds=args.n_folds),
        tie_tolerance=args.tie_tolerance,
    )

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(fold_output, index=False, encoding="utf-8")
    summary_df.to_csv(summary_output, index=False, encoding="utf-8")
    payload = selection_payload(
        summary_df,
        base_dir=args.base_dir,
        tie_tolerance=args.tie_tolerance,
    )
    with selection_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved fold results to: {fold_output}")
    print(f"Saved lambda summary to: {summary_output}")
    print(f"Saved selected lambda to: {selection_output}")
    print("\n=== Selected lambda by 5-fold mean test Ctd ===")
    print(summary_df.loc[summary_df["selected"]])


if __name__ == "__main__":
    main()
