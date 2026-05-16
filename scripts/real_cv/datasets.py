"""実データ CV で使う dataset 固有の前処理定義。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.real.support.prepare_support2_inference import (
    BINARY_FEATURE_COLS as SUPPORT2_BINARY_FEATURE_COLS,
)
from data.real.support.prepare_support2_inference import (
    CATEGORICAL_LEVELS as SUPPORT2_CATEGORICAL_LEVELS,
)
from data.real.support.prepare_support2_inference import (
    CATEGORICAL_FEATURE_COLS as SUPPORT2_CATEGORICAL_FEATURE_COLS,
)
from data.real.support.prepare_support2_inference import (
    CONTINUOUS_FEATURE_COLS as SUPPORT2_CONTINUOUS_FEATURE_COLS,
)
from data.real.support.prepare_support2_inference import (
    DEFAULT_FEATURE_COLS as SUPPORT2_FEATURE_COLS,
)
from data.real.support.prepare_support2_inference import (
    RAW_FEATURE_COLS as SUPPORT2_RAW_FEATURE_COLS,
)


@dataclass(frozen=True)
class RealDatasetSpec:
    """raw 実データを CV 用 base DataFrame へ変換するための定義。"""

    name: str
    default_input: Path
    raw_feature_cols: list[str]
    feature_cols: list[str]
    continuous_feature_cols: list[str]
    categorical_feature_cols: list[str]
    categorical_reference_levels: dict[str, str]
    time_scale_max: float | None
    loader: Callable[[Path], pd.DataFrame]


def _load_support2_base(input_path: Path) -> pd.DataFrame:
    source = pd.read_csv(input_path, na_values=["NA"])
    source = source.reset_index(names="source_row")
    source["id"] = source["source_row"].astype(int) + 1

    required_source_cols = ["id", "d.time", "death", *SUPPORT2_RAW_FEATURE_COLS]
    base = source.loc[:, required_source_cols].copy()

    numeric_cols = [
        "id",
        "d.time",
        "death",
        *SUPPORT2_CONTINUOUS_FEATURE_COLS,
        *SUPPORT2_BINARY_FEATURE_COLS,
    ]
    for col in numeric_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    for col, levels in SUPPORT2_CATEGORICAL_LEVELS.items():
        base[col] = base[col].astype("string").str.strip().str.lower()
        observed = set(base[col].dropna().unique())
        unexpected = sorted(observed - set(levels))
        if unexpected:
            raise ValueError(f"Unexpected categories in {col}: {unexpected}")

    base = base.dropna(subset=required_source_cols).reset_index(drop=True)
    for col, levels in SUPPORT2_CATEGORICAL_LEVELS.items():
        for level in levels[1:]:
            encoded_col = f"{col}_{level}"
            base[encoded_col] = (base[col] == level).astype(int)

    base = base.loc[base["d.time"] > 0].reset_index(drop=True)
    if base.empty:
        raise ValueError("No rows remain after SUPPORT2 preprocessing")

    base["time_original"] = base["d.time"].astype(float)
    base["event_original"] = base["death"].astype(int)
    return base[["id", "time_original", "event_original", *SUPPORT2_FEATURE_COLS]]


def _load_framingham_base(input_path: Path) -> pd.DataFrame:
    source = pd.read_csv(input_path, na_values=["NA"])
    feature_cols = ["AGE", "SEX", "BMI", "SYSBP", "DIABP"]
    required_cols = ["RANDID", "PERIOD", *feature_cols, "TIMEHYP"]
    base = source.loc[:, required_cols].copy()

    for col in required_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    base = (
        base.dropna(subset=required_cols)
        .sort_values(["RANDID", "PERIOD"])
        .drop_duplicates(subset=["RANDID"], keep="first")
        .reset_index(drop=True)
    )

    censor_sentinel = 8766.0
    base = base.loc[base["TIMEHYP"] != 0].reset_index(drop=True)
    if base.empty:
        raise ValueError("No rows remain after Framingham preprocessing")

    base["id"] = base["RANDID"].astype(int)
    base["time_original"] = base["TIMEHYP"].astype(float)
    base["event_original"] = (base["TIMEHYP"] != censor_sentinel).astype(int)
    return base[["id", "time_original", "event_original", *feature_cols]]


DATASET_SPECS: dict[str, RealDatasetSpec] = {
    "support2": RealDatasetSpec(
        name="support2",
        default_input=Path("data/real/support/support2.csv"),
        raw_feature_cols=SUPPORT2_RAW_FEATURE_COLS,
        feature_cols=SUPPORT2_FEATURE_COLS,
        continuous_feature_cols=SUPPORT2_CONTINUOUS_FEATURE_COLS,
        categorical_feature_cols=SUPPORT2_CATEGORICAL_FEATURE_COLS,
        categorical_reference_levels={
            col: levels[0] for col, levels in SUPPORT2_CATEGORICAL_LEVELS.items()
        },
        time_scale_max=None,
        loader=_load_support2_base,
    ),
    "framingham": RealDatasetSpec(
        name="framingham",
        default_input=Path("data/real/framingham/framingham.csv"),
        raw_feature_cols=["AGE", "SEX", "BMI", "SYSBP", "DIABP"],
        feature_cols=["AGE", "SEX", "BMI", "SYSBP", "DIABP"],
        continuous_feature_cols=["AGE", "BMI", "SYSBP", "DIABP"],
        categorical_feature_cols=[],
        categorical_reference_levels={},
        time_scale_max=8766.0,
        loader=_load_framingham_base,
    ),
}


def get_dataset_spec(name: str) -> RealDatasetSpec:
    """dataset 名から前処理定義を返す。"""

    key = name.lower()
    if key not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset: {name}. Supported: {supported}")
    return DATASET_SPECS[key]


def load_real_base(dataset: str, input_path: Path | None = None) -> pd.DataFrame:
    """dataset 名に応じて raw CSV から 1 患者 1 行の base DataFrame を作る。"""

    spec = get_dataset_spec(dataset)
    path = input_path or spec.default_input
    base = spec.loader(path)
    if base["id"].duplicated().any():
        raise ValueError(f"{dataset} base data must contain one row per id")
    if np.any(base["time_original"].to_numpy(dtype=float) <= 0):
        raise ValueError("time_original must be positive after preprocessing")
    return base
