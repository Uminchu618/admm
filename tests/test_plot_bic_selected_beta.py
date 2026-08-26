from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pilot.plot_bic_selected_beta import (
    parse_seeds,
    select_bic_fits,
    step_values,
)


def test_select_bic_fits_uses_only_eligible_candidates() -> None:
    summary = pd.DataFrame(
        [
            {
                "data_name": "oracle_seed_42",
                "lambda_fuse": 0.0,
                "bic": 100.0,
                "bic_eligible": True,
                "result_path": "eligible.json",
            },
            {
                "data_name": "oracle_seed_42",
                "lambda_fuse": 0.1,
                "bic": 50.0,
                "bic_eligible": False,
                "result_path": "ineligible.json",
            },
            {
                "data_name": "oracle_seed_42",
                "lambda_fuse": 0.03,
                "bic": 90.0,
                "bic_eligible": True,
                "result_path": "selected.json",
            },
        ]
    )

    selected = select_bic_fits(summary, ["oracle"], [42])

    assert len(selected) == 1
    assert selected.iloc[0]["lambda_fuse"] == 0.03
    assert selected.iloc[0]["result_path"] == "selected.json"


def test_select_bic_fits_requires_every_requested_seed() -> None:
    summary = pd.DataFrame(
        [
            {
                "data_name": "oracle_seed_42",
                "lambda_fuse": 0.0,
                "bic": 100.0,
                "bic_eligible": True,
                "result_path": "result.json",
            }
        ]
    )

    with pytest.raises(ValueError, match="oracle.*43"):
        select_bic_fits(summary, ["oracle"], [42, 43])


def test_seed_and_step_helpers() -> None:
    assert parse_seeds("42, 43,44") == [42, 43, 44]
    np.testing.assert_allclose(step_values(np.array([0.2, 0.6])), [0.2, 0.6, 0.6])
