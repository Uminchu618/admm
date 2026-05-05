from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.real_cv.common import build_fold_long_data, make_fold_assignments
from scripts.real_cv.datasets import get_dataset_spec, load_real_base


def test_real_cv_support2_fold_preprocessing(tmp_path: Path) -> None:
    raw_path = tmp_path / "support2.csv"
    pd.DataFrame(
        {
            "age": [60, 70, 50, 80],
            "death": [1, 0, 1, 0],
            "sex": ["male", "female", "female", "male"],
            "d.time": [100, 200, 300, 400],
            "race": ["white", "black", "asian", "other"],
            "num.co": [1, 2, 0, 3],
            "diabetes": [0, 1, 0, 1],
            "dementia": [0, 0, 1, 1],
            "ca": ["no", "yes", "metastatic", "no"],
            "meanbp": [80, 90, 85, 95],
            "hrt": [90, 100, 95, 105],
            "resp": [20, 22, 18, 24],
            "temp": [37.0, 37.5, 36.8, 38.0],
            "wblc": [10.0, 12.0, 8.0, 14.0],
            "sod": [138, 140, 136, 142],
            "crea": [1.0, 1.2, 0.8, 1.4],
        }
    ).to_csv(raw_path, index=False)

    base = load_real_base("support2", raw_path)
    spec = get_dataset_spec("support2")
    assignments = make_fold_assignments(
        base, n_folds=2, random_state=0, stratify_event=True
    )
    train_long, test_long, summary = build_fold_long_data(
        base=base,
        assignments=assignments,
        fold=0,
        time_grid=np.array([0.0, 1.0, 2.0]),
        spec=spec,
    )

    assert summary["dataset"] == "support2"
    assert summary["feature_cols"] == spec.feature_cols
    assert set(train_long["k"].unique()) == {0, 1}
    assert set(test_long["k"].unique()) == {0, 1}
    assert train_long.isna().sum().sum() == 0
    assert test_long.isna().sum().sum() == 0
    assert "sex_male" in train_long.columns
    assert "race_black" in train_long.columns
    assert "ca_metastatic" in train_long.columns


def test_real_cv_framingham_fold_preprocessing(tmp_path: Path) -> None:
    raw_path = tmp_path / "framingham.csv"
    pd.DataFrame(
        {
            "RANDID": [1, 1, 2, 3, 4],
            "PERIOD": [1, 2, 1, 1, 1],
            "AGE": [40, 45, 50, 60, 70],
            "SEX": [1, 1, 2, 1, 2],
            "BMI": [25.0, 26.0, 27.0, 28.0, 29.0],
            "SYSBP": [120.0, 122.0, 130.0, 140.0, 150.0],
            "DIABP": [80.0, 82.0, 85.0, 90.0, 95.0],
            "TIMEHYP": [8766, 7000, 3000, 4000, 8766],
        }
    ).to_csv(raw_path, index=False)

    base = load_real_base("framingham", raw_path)
    spec = get_dataset_spec("framingham")
    assignments = make_fold_assignments(
        base, n_folds=2, random_state=0, stratify_event=True
    )
    train_long, test_long, summary = build_fold_long_data(
        base=base,
        assignments=assignments,
        fold=0,
        time_grid=np.array([0.0, 1.0, 2.0]),
        spec=spec,
    )

    assert summary["dataset"] == "framingham"
    assert summary["feature_cols"] == ["AGE", "SEX", "BMI", "SYSBP", "DIABP"]
    assert summary["time_scale_max_original"] == 8766.0
    assert base["id"].tolist() == [1, 2, 3, 4]
    assert train_long.isna().sum().sum() == 0
    assert test_long.isna().sum().sum() == 0
    assert train_long["time"].between(0.0, 2.0).all()
    assert test_long["time"].between(0.0, 2.0).all()
