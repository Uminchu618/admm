#!/usr/bin/env python3
"""実データ CV の fold ごとに CoxPH baseline を評価する。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from admm.evaluator import HazardAFTEvaluator  # noqa: E402
from scripts.compute_cox_metrics import (  # noqa: E402
    CoxPHSurvivalAdapter,
    _load_subject_level,
)


def _parse_path_metadata(path: Path) -> tuple[float | None, int | None]:
    lambda_value = None
    fold = None
    for part in path.parts:
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


def _parse_dataset(path: Path) -> str | None:
    parts = list(path.parts)
    if "real_cv" not in parts:
        return None
    idx = parts.index("real_cv")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def collect_fold_data_paths(base_dir: Path) -> pd.DataFrame:
    """base_dir 以下から fold ごとの train/test CSV を重複なく集める。"""

    rows: list[dict[str, Any]] = []
    seen_folds: set[int] = set()

    meta_paths = sorted(base_dir.rglob("fold_meta.json"))
    for meta_path in meta_paths:
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception as exc:
            print(f"Warning: failed to read {meta_path}: {exc}", file=sys.stderr)
            continue

        path_lambda, path_fold = _parse_path_metadata(meta_path)
        fold = meta.get("fold", path_fold)
        if fold is None:
            continue
        fold = int(fold)
        if fold in seen_folds:
            continue

        train_path = Path(
            meta.get("train_path", meta_path.parent / "data" / "train.csv")
        )
        test_path = Path(meta.get("test_path", meta_path.parent / "data" / "test.csv"))
        if not train_path.is_absolute():
            train_path = ROOT / train_path
        if not test_path.is_absolute():
            test_path = ROOT / test_path
        if not train_path.exists() or not test_path.exists():
            continue

        rows.append(
            {
                "dataset": meta.get("dataset", _parse_dataset(meta_path)),
                "fold": fold,
                "source_lambda_fuse": meta.get("lambda_fuse", path_lambda),
                "train_path": str(train_path),
                "test_path": str(test_path),
            }
        )
        seen_folds.add(fold)

    if rows:
        return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    for train_path in sorted(base_dir.rglob("data/train.csv")):
        _, fold = _parse_path_metadata(train_path)
        if fold is None or fold in seen_folds:
            continue
        test_path = train_path.parent / "test.csv"
        if not test_path.exists():
            continue
        rows.append(
            {
                "dataset": _parse_dataset(train_path),
                "fold": int(fold),
                "source_lambda_fuse": None,
                "train_path": str(train_path),
                "test_path": str(test_path),
            }
        )
        seen_folds.add(int(fold))

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def fit_and_score_cox_fold(
    train_path: Path,
    test_path: Path,
    *,
    penalizer: float = 0.0,
) -> dict[str, float | int | str]:
    """1 fold の train/test long-format CSV で CoxPH を評価する。"""

    X_train, y_train, feature_cols = _load_subject_level(train_path)
    X_test, y_test, test_feature_cols = _load_subject_level(test_path)
    if test_feature_cols != feature_cols:
        raise ValueError("train/test feature columns do not match")

    fit_df = pd.DataFrame(X_train, columns=feature_cols)
    fit_df["time"] = y_train[:, 0].astype(float)
    fit_df["event"] = y_train[:, 1].astype(int)

    cox = CoxPHFitter(penalizer=float(penalizer))
    cox.fit(fit_df, duration_col="time", event_col="event")

    adapter = CoxPHSurvivalAdapter(cox, feature_cols)
    evaluator = HazardAFTEvaluator()
    c_td_train = evaluator.compute_c_td(adapter, X_train, y_train)
    c_td_test = evaluator.compute_c_td(adapter, X_test, y_test)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    partial_hazard_test = np.asarray(
        cox.predict_partial_hazard(test_df), dtype=float
    )
    c_harrell_test = concordance_index(
        y_test[:, 0].astype(float),
        -partial_hazard_test,
        y_test[:, 1].astype(int),
    )

    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_train_events": int(np.sum(y_train[:, 1] == 1)),
        "n_test_events": int(np.sum(y_test[:, 1] == 1)),
        "c_td_train_cox": float(c_td_train),
        "c_td_test_cox": float(c_td_test),
        "c_index_harrell_test": float(c_harrell_test),
    }


def summarize_cox_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """fold 別 Cox 結果から平均・標準偏差・標準誤差を作る。"""

    if fold_df.empty:
        return pd.DataFrame()

    row: dict[str, Any] = {
        "dataset": fold_df["dataset"].dropna().iloc[0]
        if "dataset" in fold_df and fold_df["dataset"].notna().any()
        else None,
        "n_folds": int(fold_df["fold"].nunique()),
    }
    for metric in ["c_td_test_cox", "c_td_train_cox", "c_index_harrell_test"]:
        values = pd.to_numeric(fold_df[metric], errors="coerce").dropna()
        row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
        row[f"{metric}_std"] = (
            float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan
        )
        row[f"{metric}_se"] = (
            row[f"{metric}_std"] / math.sqrt(values.shape[0])
            if values.shape[0] > 1 and pd.notna(row[f"{metric}_std"])
            else np.nan
        )
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit CoxPH baseline per real-data CV fold"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/real_cv/support2/support2_5fold_seed1234"),
        help="Experiment directory containing lambda_*/fold_*/data/train.csv.",
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=None,
        help="Output CSV for fold-level Cox metrics.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output CSV for Cox summary metrics.",
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.0,
        help="Optional lifelines CoxPHFitter penalizer.",
    )
    args = parser.parse_args()

    fold_output = args.fold_output or (args.base_dir / "cox_fold_results.csv")
    summary_output = args.summary_output or (args.base_dir / "cox_summary.csv")

    path_df = collect_fold_data_paths(args.base_dir)
    if path_df.empty:
        raise FileNotFoundError(f"No fold train/test CSVs found under {args.base_dir}")

    rows: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []
    for row in path_df.to_dict(orient="records"):
        fold = int(row["fold"])
        try:
            scored = fit_and_score_cox_fold(
                Path(row["train_path"]),
                Path(row["test_path"]),
                penalizer=args.penalizer,
            )
            rows.append({**row, **scored, "penalizer": float(args.penalizer)})
            print(
                f"OK fold {fold:02d}: "
                f"test c_td={scored['c_td_test_cox']:.4f}, "
                f"Harrell={scored['c_index_harrell_test']:.4f}"
            )
        except Exception as exc:  # pragma: no cover - CLI robustness
            errors.append((fold, str(exc)))
            print(f"ERR fold {fold:02d}: {exc}", file=sys.stderr)

    if not rows:
        raise RuntimeError("All Cox fold fits failed; no summary to write.")

    result_df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    summary_df = summarize_cox_folds(result_df)

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(fold_output, index=False, encoding="utf-8")
    summary_df.to_csv(summary_output, index=False, encoding="utf-8")
    print(f"Saved Cox fold results to: {fold_output}")
    print(f"Saved Cox summary to: {summary_output}")

    if errors:
        print("\nFailed folds:")
        for fold, message in errors:
            print(f"- fold {fold:02d}: {message}")


if __name__ == "__main__":
    main()
