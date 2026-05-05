"""実データの qsub 並列 CV で共通利用する関数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from admm.config import load_config
from scripts.real_cv.datasets import RealDatasetSpec


def lambda_label(value: float) -> str:
    """lambda 値をディレクトリ名に埋め込むためのラベルへ変換する。"""

    return f"lambda_{value:.15g}"


def fold_label(fold: int) -> str:
    """fold 番号をディレクトリ名に埋め込むためのラベルへ変換する。"""

    return f"fold_{int(fold):02d}"


def load_lambda_values(path: Path) -> list[float]:
    """lambda_grid.json から lambda 値を読む。"""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = [float(value) for value in payload.get("lambda_values", [])]
    if not values:
        raise ValueError(f"No lambda_values found in {path}")
    return values


def load_time_grid(config_path: Path) -> np.ndarray:
    """config から time_grid を読む。"""

    config = load_config(config_path)
    time_grid = np.asarray(config["time_grid"], dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be a one-dimensional array with >= 2 values")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")
    return time_grid


def make_fold_assignments(
    base: pd.DataFrame,
    n_folds: int,
    random_state: int,
    stratify_event: bool = True,
) -> pd.DataFrame:
    """id 単位の fold 割当を作る。"""

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if base["id"].duplicated().any():
        raise ValueError("base must contain one row per id")
    if base.shape[0] < n_folds:
        raise ValueError("n_folds must be <= number of subjects")

    rng = np.random.default_rng(random_state)
    folds = np.empty(base.shape[0], dtype=int)

    if stratify_event:
        events = base["event_original"].astype(int).to_numpy()
        for event in sorted(np.unique(events).tolist()):
            indices = np.flatnonzero(events == event)
            rng.shuffle(indices)
            for pos, row_idx in enumerate(indices):
                folds[row_idx] = pos % n_folds
    else:
        indices = np.arange(base.shape[0])
        rng.shuffle(indices)
        for pos, row_idx in enumerate(indices):
            folds[row_idx] = pos % n_folds

    assignments = base[["id", "event_original"]].copy()
    assignments["fold"] = folds
    return assignments.sort_values("id").reset_index(drop=True)


def _standardize_fold(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    continuous_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """train の係数だけで train/test の連続特徴量を標準化する。"""

    train_out = train_base.copy()
    test_out = test_base.copy()
    standardization: dict[str, float] = {}

    for col in continuous_cols:
        mean = float(train_out[col].mean())
        std = float(train_out[col].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid standard deviation for {col}")

        train_out[col] = (train_out[col] - mean) / std
        test_out[col] = (test_out[col] - mean) / std
        standardization[f"{col}_mean"] = mean
        standardization[f"{col}_std"] = std

    return train_out, test_out, standardization


def _add_scaled_outcome(
    frame: pd.DataFrame,
    t0: float,
    tK: float,
    time_scale_max: float,
) -> pd.DataFrame:
    """time_original/event_original から main.py 用 time/event を作る。"""

    out = frame.copy()
    out["time"] = (out["time_original"] * (tK / time_scale_max)).clip(
        lower=t0, upper=tK
    )
    out["event"] = out["event_original"].astype(int)
    return out


def to_long_format(base: pd.DataFrame, k_count: int, feature_cols: list[str]) -> pd.DataFrame:
    """1 患者 1 行のデータを main.py 用 long format へ変換する。"""

    base_keep = base[["id", *feature_cols, "time", "event"]].copy()
    long_df = base_keep.loc[base_keep.index.repeat(k_count)].reset_index(drop=True)
    long_df["k"] = np.tile(np.arange(k_count, dtype=int), base_keep.shape[0])
    long_df = long_df[["id", "k", "time", "event", *feature_cols]]
    return long_df.sort_values(["id", "k"]).reset_index(drop=True)


def build_fold_long_data(
    base: pd.DataFrame,
    assignments: pd.DataFrame,
    fold: int,
    time_grid: np.ndarray,
    spec: RealDatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """指定 fold の train/test long format とメタ情報を作る。"""

    if fold < 0:
        raise ValueError("fold must be non-negative")

    merged = base.merge(assignments[["id", "fold"]], on="id", how="inner")
    if merged["id"].nunique() != base["id"].nunique():
        raise ValueError("split assignments do not cover all complete-case ids")

    train_base = merged.loc[merged["fold"] != fold].reset_index(drop=True)
    test_base = merged.loc[merged["fold"] == fold].reset_index(drop=True)
    if train_base.empty or test_base.empty:
        raise ValueError(f"fold {fold} has empty train or test data")

    train_base, test_base, standardization = _standardize_fold(
        train_base=train_base,
        test_base=test_base,
        continuous_cols=spec.continuous_feature_cols,
    )

    t0 = float(time_grid[0])
    tK = float(time_grid[-1])
    k_count = int(time_grid.size - 1)
    time_scale_max = (
        float(spec.time_scale_max)
        if spec.time_scale_max is not None
        else float(train_base["time_original"].max())
    )
    if time_scale_max <= 0:
        raise ValueError("time scale max must be positive")

    train_base = _add_scaled_outcome(train_base, t0, tK, time_scale_max)
    test_base = _add_scaled_outcome(test_base, t0, tK, time_scale_max)

    train_long = to_long_format(train_base, k_count, spec.feature_cols)
    test_long = to_long_format(test_base, k_count, spec.feature_cols)

    summary = {
        "dataset": spec.name,
        "fold": int(fold),
        "K": k_count,
        "time_grid": time_grid.tolist(),
        "t0": t0,
        "tK": tK,
        "n_train": int(train_base.shape[0]),
        "n_test": int(test_base.shape[0]),
        "n_train_events": int(train_base["event"].sum()),
        "n_test_events": int(test_base["event"].sum()),
        "train_rows": int(train_long.shape[0]),
        "test_rows": int(test_long.shape[0]),
        "time_scale_max_original": time_scale_max,
        "raw_feature_cols": spec.raw_feature_cols,
        "feature_cols": spec.feature_cols,
        "categorical_feature_cols": spec.categorical_feature_cols,
        "categorical_reference_levels": spec.categorical_reference_levels,
        "standardize_cols": spec.continuous_feature_cols,
        "standardization": standardization,
    }
    return train_long, test_long, summary
