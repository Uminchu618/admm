from __future__ import annotations

import json
import tomllib
from pathlib import Path

from generation.extended_aft_step_generator import build_generator


def test_pilot_scenario_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "generation" / "pilot"
    expected_intervals = {
        "oracle": 6,
        "fine_grid": 12,
        "off_grid": 12,
        "small": 12,
        "no_change": 12,
    }

    for scenario, n_intervals in expected_intervals.items():
        config = json.loads(
            (config_dir / f"{scenario}.json").read_text(encoding="utf-8")
        )
        generator = build_generator(config)
        assert generator.n == 1000
        assert len(generator.analysis_time_grid) - 1 == n_intervals
        assert generator.censoring.admin_time == 6.0
        assert generator.censoring.random_enabled is False

    off_grid = json.loads(
        (config_dir / "off_grid.json").read_text(encoding="utf-8")
    )
    assert off_grid["stepwise_beta"]["true_time_grid"] == [
        0.0,
        1.3,
        2.3,
        3.2,
        4.2,
        6.0,
    ]

    lambda_grid = json.loads(
        (config_dir / "lambda_grid.json").read_text(encoding="utf-8")
    )
    diagnostic_lambda_grid = json.loads(
        (config_dir / "diagnostic_lambda_grid.json").read_text(encoding="utf-8")
    )
    assert lambda_grid["lambda_values"] == [
        0.0,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.25,
    ]
    assert lambda_grid["lambda_values"] == diagnostic_lambda_grid["lambda_values"]

    solver_config = tomllib.loads(
        (config_dir / "diagnostic_config.toml").read_text(encoding="utf-8")
    )
    assert solver_config["adaptive_rho"] is True
    assert solver_config["newton_steps_per_admm"] == 5
    assert solver_config["rho_update_interval"] == 5
    assert solver_config["rho_balance_mu"] == 10.0
    assert solver_config["max_admm_iter"] == 1000
