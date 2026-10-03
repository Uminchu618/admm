from __future__ import annotations

import pandas as pd
import pytest

from scripts.pilot.visualize_cv_results import prepare_cv_selected_records
from scripts.pilot.plot_cv_selected_beta import select_cv_refits


def _selections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_name": ["oracle_seed_42", "fine_grid_seed_42"],
            "selected_lambda": [0.01, 0.03],
            "mean_c_td": [0.72, 0.70],
            "n_folds": [5, 5],
        }
    )


def _refits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_name": ["oracle_seed_42", "fine_grid_seed_42"],
            "lambda_fuse": [0.01, 0.03],
            "c_td_test": [0.71, 0.69],
            "c_td_train": [0.75, 0.74],
            "converged": [True, True],
            "n_change_points": [4, 5],
            "result_path": ["a/result.json", "b/result.json"],
        }
    )


def test_prepare_cv_selected_records_keeps_independent_test_score() -> None:
    records = prepare_cv_selected_records(_selections(), _refits())

    assert records["scenario"].tolist() == ["fine_grid", "oracle"]
    oracle = records.loc[records["scenario"] == "oracle"].iloc[0]
    assert oracle["c_td_test"] == 0.71
    assert oracle["mean_c_td"] == 0.72
    assert oracle["cv_to_independent_gap"] == pytest.approx(0.01)


def test_prepare_cv_selected_records_rejects_wrong_refit_lambda() -> None:
    refits = _refits()
    refits.loc[0, "lambda_fuse"] = 0.1

    with pytest.raises(ValueError, match="does not match"):
        prepare_cv_selected_records(_selections(), refits)


def test_select_cv_refits_requires_requested_scenarios_and_seeds() -> None:
    selected = select_cv_refits(
        _selections(),
        _refits(),
        scenarios=["oracle", "fine_grid"],
        seeds=[42],
    )

    assert len(selected) == 2
    assert set(selected["lambda_fuse"]) == {0.01, 0.03}
