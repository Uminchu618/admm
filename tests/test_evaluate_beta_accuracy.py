from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.evaluate_beta_accuracy import (
    collect_beta_accuracy,
    compute_integrated_errors,
    compute_true_beta_by_interval,
)


def test_compute_integrated_errors_uses_interval_widths() -> None:
    time_grid = np.array([0.0, 1.0, 3.0])
    coef = np.array([[1.0, 0.0], [3.0, -2.0]])
    true_beta = np.array([[0.0, 1.0], [1.0, -1.0]])

    iae, ise = compute_integrated_errors(coef, true_beta, time_grid)

    np.testing.assert_allclose(iae, np.array([5.0, 3.0]))
    np.testing.assert_allclose(ise, np.array([9.0, 3.0]))


def test_collect_beta_accuracy_from_result_json(tmp_path: Path) -> None:
    config_path = tmp_path / "generator.config.json"
    config_path.write_text(
        json.dumps(
            {
                "stepwise_beta": {
                    "time_grid": [0.0, 1.0, 3.0],
                    "beta1_levels": [0.0, 1.0],
                    "beta2_levels": [1.0, -1.0],
                }
            }
        ),
        encoding="utf-8",
    )

    result_dir = tmp_path / "lambda_experiments" / "seed_1" / "lambda_0.5"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "time_grid": [0.0, 1.0, 3.0],
                "coef": [[1.0, 0.0], [3.0, -2.0]],
            }
        ),
        encoding="utf-8",
    )

    df = collect_beta_accuracy(tmp_path / "lambda_experiments", config_path)
    row = df.iloc[0]

    assert row["data_name"] == "seed_1"
    assert row["lambda_fuse"] == 0.5
    assert row["iae_mean"] == 4.0
    assert row["ise_mean"] == 6.0


def test_compute_true_beta_by_interval_for_stepwise_config(tmp_path: Path) -> None:
    config_path = tmp_path / "generator.config.json"
    config_path.write_text(
        json.dumps(
            {
                "stepwise_beta": {
                    "time_grid": [0.0, 1.0, 2.0, 4.0],
                    "beta1_levels": [-0.2, 0.6, 0.6],
                    "beta2_levels": [-0.4, -0.4, 0.1],
                }
            }
        ),
        encoding="utf-8",
    )

    true_beta = compute_true_beta_by_interval(
        np.array([0.0, 1.0, 2.0, 4.0]), config_path
    )

    np.testing.assert_allclose(true_beta[:, 0], np.array([-0.2, 0.6, 0.6]))
    np.testing.assert_allclose(true_beta[:, 1], np.array([-0.4, -0.4, 0.1]))
