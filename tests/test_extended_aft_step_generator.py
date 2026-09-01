from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from generation.extended_aft_step_generator import build_generator
from generation.generate_extended_aft_step_datasets import generate_datasets


def _config() -> dict:
    return {
        "n": 3,
        "x23_dist": "normal",
        "seed": 42,
        "analysis_time_grid": np.arange(0.0, 6.5, 0.5).tolist(),
        "baseline": {"alpha": 1.5, "rho": 3.0},
        "stepwise_beta": {
            "true_time_grid": [0.0, 1.3, 2.3, 3.2, 4.2, 6.0],
            "beta1_levels": [0.2, 0.2, 0.6, 0.6, 0.6],
            "beta2_levels": [0.4, 0.4, 0.4, 0.4, 0.1],
            "beta3_levels": [0.1, 0.5, 0.5, 0.3, 0.3],
        },
        "censoring": {
            "admin_time": 6.0,
            "random_a": 1.0,
            "random_b": 8.0,
            "random_enabled": False,
        },
        "grid": {"dt": 0.1, "epsilon": 0.01, "t_max": 8.0},
        "interval_covariates": {"noise_scale": 0.0},
    }


def test_true_and_analysis_time_grids_are_independent() -> None:
    generator = build_generator(_config())
    data = generator.simulate()

    assert generator.step_params.time_grid == [0.0, 1.3, 2.3, 3.2, 4.2, 6.0]
    assert generator.analysis_time_grid == np.arange(0.0, 6.5, 0.5).tolist()
    assert data.shape[0] == 3 * 12
    assert data.groupby("id")["k"].apply(list).tolist() == [list(range(12))] * 3
    assert np.allclose(generator._beta1(np.array([2.2, 2.4])), [0.2, 0.6])
    assert generator.metadata() == {
        "time_grid": np.arange(0.0, 6.5, 0.5).tolist(),
        "analysis_time_grid": np.arange(0.0, 6.5, 0.5).tolist(),
        "true_time_grid": [0.0, 1.3, 2.3, 3.2, 4.2, 6.0],
    }


def test_legacy_time_grid_defaults_to_analysis_grid() -> None:
    config = _config()
    config.pop("analysis_time_grid")
    step_config = config["stepwise_beta"]
    step_config["time_grid"] = step_config.pop("true_time_grid")

    generator = build_generator(config)

    assert generator.analysis_time_grid == generator.step_params.time_grid


def test_batch_generation_writes_paired_independent_evaluation_data(
    tmp_path: Path,
) -> None:
    config = _config()
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"

    generate_datasets(
        cfg=copy.deepcopy(config),
        output_dir=train_dir,
        seed_start=7,
        seed_end=7,
        prefix="scenario_seed_",
        overwrite=False,
        baseline_alpha=None,
        eval_output_dir=eval_dir,
        eval_seed_offset=100,
        eval_n=2,
    )

    train_path = train_dir / "scenario_seed_7.csv"
    eval_path = eval_dir / "scenario_seed_7.csv"
    assert train_path.exists()
    assert eval_path.exists()
    assert train_path.read_bytes() != eval_path.read_bytes()

    train_meta = json.loads(
        train_path.with_suffix(".csv.meta.json").read_text(encoding="utf-8")
    )
    eval_meta = json.loads(
        eval_path.with_suffix(".csv.meta.json").read_text(encoding="utf-8")
    )
    assert train_meta == eval_meta == build_generator(config).metadata()
