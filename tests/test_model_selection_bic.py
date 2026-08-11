from __future__ import annotations

import json
import math
from pathlib import Path

from admm.model_selection import (
    compute_bic,
    count_change_points,
    effective_degrees_of_freedom,
)
from scripts.aggregate_lambda_results import collect_results


def test_bic_uses_baseline_features_and_change_points() -> None:
    z = [[0.0, 0.2], [0.3, 0.0], [0.0, 0.0]]

    assert count_change_points(z, tol=1e-8) == 2
    df = effective_degrees_of_freedom(
        n_baseline_basis=8,
        n_features=3,
        z=z,
        z_tol=1e-8,
    )
    assert df == 13
    assert compute_bic(
        neg_loglik=100.0,
        n_samples=1000,
        degrees_of_freedom=df,
    ) == 200.0 + 13.0 * math.log(1000)


def test_lambda_aggregation_keeps_test_ctd_and_correct_df(tmp_path: Path) -> None:
    base_dir = tmp_path / "outputs" / "lambda_experiments"
    result_path = base_dir / "scenario_seed_7" / "lambda_1.0" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "n_samples": 1000,
                "n_eval_samples": 1000,
                "n_features": 3,
                "z_last": [[0.0, 0.2], [0.3, 0.0], [0.0, 0.0]],
                "summary": {
                    "neg_loglik_last": 100.0,
                    "c_td": 0.71,
                    "c_td_train": 0.80,
                    "c_td_test": 0.71,
                },
                "config": {"n_baseline_basis": 8, "lambda_fuse": 1.0},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_results(base_dir, z_tol=1e-8)

    assert len(rows) == 1
    row = rows[0]
    assert row["n_change_points"] == 2
    assert row["n_params"] == 13
    assert row["bic"] == 200.0 + 13.0 * math.log(1000)
    assert row["c_td"] == row["c_td_test"] == 0.71
    assert row["c_td_train"] == 0.80
