from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.simulation_cv.prepare_fold import split_long_data


def _long_data(n_subjects: int = 20, n_intervals: int = 3) -> pd.DataFrame:
    ids = np.repeat(np.arange(n_subjects), n_intervals)
    return pd.DataFrame(
        {
            "id": ids,
            "k": np.tile(np.arange(n_intervals), n_subjects),
            "time": np.repeat(np.linspace(1.0, 5.0, n_subjects), n_intervals),
            "event": np.repeat(np.arange(n_subjects) % 2, n_intervals),
            "x1": np.repeat(np.linspace(-1.0, 1.0, n_subjects), n_intervals),
        }
    )


def test_simulation_cv_split_is_subject_level_and_deterministic() -> None:
    data = _long_data()

    train_a, test_a, summary_a = split_long_data(
        data, fold=2, n_folds=5, random_state=1234
    )
    train_b, test_b, summary_b = split_long_data(
        data, fold=2, n_folds=5, random_state=1234
    )

    assert set(train_a["id"]).isdisjoint(set(test_a["id"]))
    assert set(train_a["id"]) | set(test_a["id"]) == set(data["id"])
    assert test_a["id"].tolist() == test_b["id"].tolist()
    assert summary_a == summary_b
    assert summary_a["n_folds"] == 5
    assert summary_a["n_train"] == 16
    assert summary_a["n_test"] == 4


def test_every_subject_is_test_once_across_five_folds() -> None:
    data = _long_data()
    test_ids = []
    for fold in range(5):
        _, test, _ = split_long_data(
            data, fold=fold, n_folds=5, random_state=1234
        )
        test_ids.extend(test["id"].unique().tolist())

    assert sorted(test_ids) == sorted(data["id"].unique().tolist())
