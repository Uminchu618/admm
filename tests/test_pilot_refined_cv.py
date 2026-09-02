from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.pilot.aggregate_refined_cv import aggregate_refined_cv
from scripts.pilot.compare_coarse_refined_cv import pair_methods
from scripts.pilot.generate_refined_cv_manifest import (
    build_grid_table,
    build_task_manifest,
    local_lambda_grid,
    result_path,
)


COARSE_GRID = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25]


def test_local_grid_has_21_values_and_contains_selected_lambda() -> None:
    for selected in COARSE_GRID:
        values = local_lambda_grid(selected, COARSE_GRID)

        assert len(values) == 21
        assert np.all(np.diff(values) > 0)
        assert np.isclose(values, selected, rtol=0.0, atol=1e-12).any()


def test_local_grid_uses_expected_brackets() -> None:
    middle = local_lambda_grid(0.03, COARSE_GRID)
    np.testing.assert_allclose([middle[0], middle[10], middle[-1]], [0.01, 0.03, 0.1])

    zero = local_lambda_grid(0.0, COARSE_GRID)
    np.testing.assert_allclose([zero[0], zero[10], zero[-1]], [0.0, 0.0001, 0.0003])

    upper = local_lambda_grid(0.25, COARSE_GRID)
    np.testing.assert_allclose([upper[0], upper[10], upper[-1]], [0.1, 0.25, 0.75])


def test_manifest_skips_existing_results(tmp_path: Path) -> None:
    selections = pd.DataFrame(
        {"data_name": ["oracle_seed_42"], "selected_lambda": [0.03]}
    )
    grid_table = build_grid_table(selections, COARSE_GRID)
    existing_dir = tmp_path / "existing"
    output_dir = tmp_path / "additions"
    existing = result_path(existing_dir, "oracle_seed_42", 0.03, 0)
    existing.parent.mkdir(parents=True)
    existing.write_text(
        json.dumps({"summary": {"converged": True, "c_td_test": 0.7}}),
        encoding="utf-8",
    )

    manifest = build_task_manifest(
        grid_table,
        n_folds=5,
        existing_base_dirs=[existing_dir],
        output_base_dir=output_dir,
    )

    assert len(manifest) == 104
    assert manifest["task_id"].tolist() == list(range(1, 105))
    assert not (
        np.isclose(manifest["lambda_fuse"], 0.03, rtol=0.0, atol=1e-12)
        & manifest["fold"].eq(0)
    ).any()


def test_aggregate_refined_cv_selects_best_eligible_lambda(tmp_path: Path) -> None:
    selections = pd.DataFrame(
        {"data_name": ["oracle_seed_42"], "selected_lambda": [0.03]}
    )
    grid_table = build_grid_table(selections, COARSE_GRID)
    grid_path = tmp_path / "refined_grid.csv"
    grid_table.to_csv(grid_path, index=False)
    coarse_dir = tmp_path / "coarse"
    additions_dir = tmp_path / "additions"
    output_dir = tmp_path / "aggregated"

    for row in grid_table.itertuples(index=False):
        for fold in range(5):
            path = result_path(
                additions_dir, row.data_name, row.lambda_fuse, fold
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            score = 0.7 - abs(np.log(row.lambda_fuse / 0.03)) * 0.01
            payload = {
                "n_samples": 800,
                "n_eval_samples": 200,
                "n_features": 3,
                "summary": {
                    "c_td_train": score + 0.01,
                    "c_td_test": score,
                    "objective_last": 100.0,
                    "primal_residual_last": 0.0,
                    "dual_residual_last": 0.0,
                    "converged": True,
                },
                "config": {"lambda_fuse": row.lambda_fuse},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

    selected, audit = aggregate_refined_cv(
        coarse_base_dir=coarse_dir,
        additions_base_dir=additions_dir,
        grid_path=grid_path,
        output_dir=output_dir,
        n_folds=5,
        tie_tolerance=1e-12,
    )

    assert selected.iloc[0]["selected_lambda"] == 0.03
    assert audit.iloc[0]["observed_results"] == 105
    assert audit.iloc[0]["eligible_lambdas"] == 21
    assert not bool(audit.iloc[0]["selected_at_local_boundary"])
    assert (output_dir / "oracle_seed_42" / "selected_lambda.json").is_file()


def test_pair_methods_computes_refined_minus_coarse_and_excludes_failed_refit() -> None:
    records = pd.DataFrame(
        [
            {
                "method": method,
                "data_name": f"oracle_seed_{seed}",
                "scenario": "oracle",
                "seed": seed,
                "lambda_fuse": lambda_fuse,
                "cv_mean_c_td": c_td,
                "c_td_test": c_td,
                "rmise": rmise,
                "detected": detected,
                "true_positive": 2,
                "converged": converged,
                "selected_at_local_boundary": False,
            }
            for method, seed, lambda_fuse, c_td, rmise, detected, converged in (
                ("coarse_cv", 42, 0.03, 0.70, 0.12, 5, True),
                ("refined_cv", 42, 0.04, 0.71, 0.10, 4, True),
                ("coarse_cv", 43, 0.03, 0.69, 0.11, 5, True),
                ("refined_cv", 43, 0.05, 0.70, 0.09, 4, False),
            )
        ]
    )

    paired_all, paired = pair_methods(records)

    assert len(paired_all) == 2
    assert len(paired) == 1
    assert paired.iloc[0]["refined_minus_coarse_c_td_test"] == pytest.approx(0.01)
    assert paired.iloc[0]["refined_minus_coarse_rmise"] == pytest.approx(-0.02)
    assert bool(paired.iloc[0]["lambda_changed"])
