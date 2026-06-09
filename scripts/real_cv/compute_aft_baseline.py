#!/usr/bin/env python3
"""実データ CV の fold ごとに単純な parametric AFT baseline を評価する。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from admm.evaluator import HazardAFTEvaluator  # noqa: E402
from scripts.compute_cox_metrics import _load_subject_level  # noqa: E402
from scripts.real_cv.compute_cox_baseline import (  # noqa: E402
    collect_fold_data_paths,
)


AFT_FITTERS = {
    "weibull": WeibullAFTFitter,
    "log_normal": LogNormalAFTFitter,
    "log_logistic": LogLogisticAFTFitter,
}


class AFTSurvivalAdapter:
    """lifelines AFT fitter を HazardAFTEvaluator 互換 API に合わせる。"""

    def __init__(self, fitter: Any, feature_cols: list[str]) -> None:
        self._fitter = fitter
        self._feature_cols = list(feature_cols)

    def _prepare_predict_X(self, X: np.ndarray) -> pd.DataFrame:
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("X must be a 2D array for AFT prediction.")
        if x_arr.shape[1] != len(self._feature_cols):
            raise ValueError("Feature count does not match fitted AFT model.")
        return pd.DataFrame(x_arr, columns=self._feature_cols)

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Iterable[float] | None = None,
    ) -> np.ndarray:
        x_df = self._prepare_predict_X(X)

        if times is None:
            survival_df = self._fitter.predict_survival_function(x_df)
            return survival_df.to_numpy(dtype=float).T

        times_arr = np.asarray(list(times), dtype=float).reshape(-1)
        if times_arr.size == 0:
            return np.zeros((x_df.shape[0], 0), dtype=float)

        survival_df = self._fitter.predict_survival_function(
            x_df, times=times_arr.tolist()
        )
        index_arr = np.asarray(survival_df.index, dtype=float)
        survival_values = survival_df.to_numpy(dtype=float)

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


def _parse_aft_models(raw: str) -> list[str]:
    model_names = [item.strip().lower().replace("-", "_") for item in raw.split(",")]
    model_names = [item for item in model_names if item]
    if not model_names:
        raise ValueError("At least one AFT model must be specified.")

    unsupported = sorted(set(model_names) - set(AFT_FITTERS))
    if unsupported:
        supported = ", ".join(sorted(AFT_FITTERS))
        raise ValueError(f"Unsupported AFT model(s): {unsupported}. Supported: {supported}")
    return model_names


def fit_and_score_aft_fold(
    train_path: Path,
    test_path: Path,
    *,
    aft_model: str,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
) -> dict[str, float | int | str]:
    """1 fold の train/test long-format CSV で parametric AFT を評価する。"""

    X_train, y_train, feature_cols = _load_subject_level(train_path)
    X_test, y_test, test_feature_cols = _load_subject_level(test_path)
    if test_feature_cols != feature_cols:
        raise ValueError("train/test feature columns do not match")

    fit_df = pd.DataFrame(X_train, columns=feature_cols)
    fit_df["time"] = y_train[:, 0].astype(float)
    fit_df["event"] = y_train[:, 1].astype(int)

    fitter_cls = AFT_FITTERS[aft_model]
    fitter = fitter_cls(penalizer=float(penalizer), l1_ratio=float(l1_ratio))
    fitter.fit(fit_df, duration_col="time", event_col="event")

    adapter = AFTSurvivalAdapter(fitter, feature_cols)
    evaluator = HazardAFTEvaluator()
    c_td_train = evaluator.compute_c_td(adapter, X_train, y_train)
    c_td_test = evaluator.compute_c_td(adapter, X_test, y_test)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    median_survival = np.asarray(fitter.predict_median(test_df), dtype=float).reshape(-1)
    finite = np.isfinite(median_survival)
    if not finite.all():
        finite_values = median_survival[finite]
        fill_value = (
            float(np.max(finite_values)) * 10.0
            if finite_values.size
            else float(np.max(y_train[:, 0]))
        )
        median_survival = np.where(finite, median_survival, fill_value)
    c_harrell_test = concordance_index(
        y_test[:, 0].astype(float),
        median_survival,
        y_test[:, 1].astype(int),
    )

    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_train_events": int(np.sum(y_train[:, 1] == 1)),
        "n_test_events": int(np.sum(y_test[:, 1] == 1)),
        "c_td_train_aft": float(c_td_train),
        "c_td_test_aft": float(c_td_test),
        "c_index_harrell_test": float(c_harrell_test),
        "log_likelihood": float(getattr(fitter, "log_likelihood_", np.nan)),
        "aic": float(getattr(fitter, "AIC_", np.nan)),
    }


def summarize_aft_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """fold 別 AFT 結果から model 別の平均・標準偏差・標準誤差を作る。"""

    if fold_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for aft_model, subset in fold_df.groupby("aft_model", dropna=False):
        row: dict[str, Any] = {
            "dataset": subset["dataset"].dropna().iloc[0]
            if "dataset" in subset and subset["dataset"].notna().any()
            else None,
            "aft_model": aft_model,
            "n_folds": int(subset["fold"].nunique()),
            "penalizer": float(subset["penalizer"].dropna().iloc[0])
            if "penalizer" in subset and subset["penalizer"].notna().any()
            else np.nan,
            "l1_ratio": float(subset["l1_ratio"].dropna().iloc[0])
            if "l1_ratio" in subset and subset["l1_ratio"].notna().any()
            else np.nan,
        }
        for metric in [
            "c_td_test_aft",
            "c_td_train_aft",
            "c_index_harrell_test",
            "log_likelihood",
            "aic",
        ]:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan
            )
            row[f"{metric}_se"] = (
                row[f"{metric}_std"] / math.sqrt(values.shape[0])
                if values.shape[0] > 1 and pd.notna(row[f"{metric}_std"])
                else np.nan
            )
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("c_td_test_aft_mean", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit parametric AFT baselines per real-data CV fold"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/real_cv/support2/support2_5fold_seed1234"),
        help="Experiment directory containing lambda_*/fold_*/data/train.csv.",
    )
    parser.add_argument(
        "--aft-models",
        type=str,
        default="weibull,log_normal,log_logistic",
        help="Comma-separated AFT models: weibull, log_normal, log_logistic.",
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=None,
        help="Output CSV for fold-level AFT metrics.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output CSV for AFT summary metrics.",
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.0,
        help="Optional lifelines AFT penalizer.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.0,
        help="Optional lifelines AFT l1_ratio.",
    )
    args = parser.parse_args()

    aft_models = _parse_aft_models(args.aft_models)
    fold_output = args.fold_output or (args.base_dir / "aft_fold_results.csv")
    summary_output = args.summary_output or (args.base_dir / "aft_summary.csv")

    path_df = collect_fold_data_paths(args.base_dir)
    if path_df.empty:
        raise FileNotFoundError(f"No fold train/test CSVs found under {args.base_dir}")

    rows: list[dict[str, Any]] = []
    errors: list[tuple[int, str, str]] = []
    for path_row in path_df.to_dict(orient="records"):
        fold = int(path_row["fold"])
        for aft_model in aft_models:
            try:
                scored = fit_and_score_aft_fold(
                    Path(path_row["train_path"]),
                    Path(path_row["test_path"]),
                    aft_model=aft_model,
                    penalizer=args.penalizer,
                    l1_ratio=args.l1_ratio,
                )
                rows.append(
                    {
                        **path_row,
                        **scored,
                        "aft_model": aft_model,
                        "penalizer": float(args.penalizer),
                        "l1_ratio": float(args.l1_ratio),
                    }
                )
                print(
                    f"OK fold {fold:02d} {aft_model}: "
                    f"test c_td={scored['c_td_test_aft']:.4f}, "
                    f"Harrell={scored['c_index_harrell_test']:.4f}"
                )
            except Exception as exc:  # pragma: no cover - CLI robustness
                errors.append((fold, aft_model, str(exc)))
                print(f"ERR fold {fold:02d} {aft_model}: {exc}", file=sys.stderr)

    if not rows:
        raise RuntimeError("All AFT fold fits failed; no summary to write.")

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["aft_model", "fold"])
        .reset_index(drop=True)
    )
    summary_df = summarize_aft_folds(result_df)

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(fold_output, index=False, encoding="utf-8")
    summary_df.to_csv(summary_output, index=False, encoding="utf-8")
    print(f"Saved AFT fold results to: {fold_output}")
    print(f"Saved AFT summary to: {summary_output}")

    if errors:
        print("\nFailed folds/models:")
        for fold, aft_model, message in errors:
            print(f"- fold {fold:02d} {aft_model}: {message}")


if __name__ == "__main__":
    main()
