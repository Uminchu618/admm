from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.real_cv.aggregate_results import (
    mark_selected_lambda,
    selection_payload,
    summarize_by_lambda,
)


def _fold_rows(
    lambda_fuse: float,
    scores: list[float],
    *,
    converged: list[bool] | None = None,
) -> list[dict[str, object]]:
    if converged is None:
        converged = [True] * len(scores)
    return [
        {
            "lambda_fuse": lambda_fuse,
            "fold": fold,
            "converged": converged[fold],
            "c_td_test": score,
            "c_td_train": score + 0.05,
            "objective_last": 100.0,
            "primal_residual_last": 1e-5,
            "dual_residual_last": 1e-5,
        }
        for fold, score in enumerate(scores)
    ]


def test_selects_largest_mean_test_ctd() -> None:
    rows = [
        *_fold_rows(0.1, [0.60, 0.61, 0.62, 0.63, 0.64]),
        *_fold_rows(1.0, [0.70, 0.71, 0.72, 0.73, 0.74]),
        *_fold_rows(10.0, [0.65, 0.66, 0.67, 0.68, 0.69]),
    ]

    summary = mark_selected_lambda(
        summarize_by_lambda(pd.DataFrame(rows), expected_n_folds=5)
    )

    selected = summary.loc[summary["selected"]]
    assert len(selected) == 1
    assert float(selected.iloc[0]["lambda_fuse"]) == 1.0
    assert math.isclose(float(selected.iloc[0]["c_td_test_mean"]), 0.72)


def test_excludes_incomplete_nonconverged_and_nonfinite_candidates() -> None:
    rows = [
        *_fold_rows(0.1, [0.90, 0.91, 0.92, 0.93]),
        *_fold_rows(
            1.0,
            [0.80, 0.81, 0.82, 0.83, 0.84],
            converged=[True, True, False, True, True],
        ),
        *_fold_rows(10.0, [0.70, 0.71, float("nan"), 0.73, 0.74]),
        *_fold_rows(100.0, [0.60, 0.61, 0.62, 0.63, 0.64]),
    ]

    summary = mark_selected_lambda(
        summarize_by_lambda(pd.DataFrame(rows), expected_n_folds=5)
    )

    eligibility = summary.set_index("lambda_fuse")["cv_eligible"].to_dict()
    assert eligibility == {0.1: False, 1.0: False, 10.0: False, 100.0: True}
    assert float(summary.loc[summary["selected"], "lambda_fuse"].iloc[0]) == 100.0


def test_tie_is_resolved_in_favor_of_larger_lambda() -> None:
    rows = [
        *_fold_rows(0.1, [0.70] * 5),
        *_fold_rows(1.0, [0.70 + 5e-13] * 5),
    ]

    summary = mark_selected_lambda(
        summarize_by_lambda(pd.DataFrame(rows), expected_n_folds=5),
        tie_tolerance=1e-12,
    )

    assert float(summary.loc[summary["selected"], "lambda_fuse"].iloc[0]) == 1.0


def test_selection_payload_contains_refit_contract() -> None:
    rows = [*_fold_rows(1.0, [0.70, 0.71, 0.72, 0.73, 0.74])]
    summary = mark_selected_lambda(
        summarize_by_lambda(pd.DataFrame(rows), expected_n_folds=5)
    )

    payload = selection_payload(
        summary,
        base_dir=Path("outputs/real_cv/support2/example"),
        tie_tolerance=1e-12,
    )

    assert payload["selection_method"] == "five_fold_cv_mean_c_td"
    assert payload["selected_lambda"] == 1.0
    assert payload["n_folds"] == 5


def test_unexpected_fold_ids_are_not_eligible() -> None:
    rows = _fold_rows(1.0, [0.70, 0.71, 0.72, 0.73, 0.74])
    for row in rows:
        row["fold"] = int(row["fold"]) + 1

    summary = summarize_by_lambda(pd.DataFrame(rows), expected_n_folds=5)

    assert not bool(summary.iloc[0]["cv_eligible"])
    assert "unexpected_fold_ids" in summary.iloc[0]["cv_exclusion_reason"]
