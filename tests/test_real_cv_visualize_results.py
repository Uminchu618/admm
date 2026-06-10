from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.real_cv.visualize_results import (
    create_all_plots,
    load_or_collect_cv_results,
    write_cv_tables,
)


def _write_result(
    base_dir: Path,
    lambda_fuse: float,
    fold: int,
    c_td_test: float,
    c_td_train: float,
) -> None:
    result_dir = base_dir / f"lambda_{lambda_fuse}" / f"fold_{fold:02d}"
    result_dir.mkdir(parents=True)
    payload = {
        "dataset": "support2",
        "n_samples": 12,
        "n_eval_samples": 4,
        "n_features": 3,
        "summary": {
            "objective_last": 100.0 + lambda_fuse,
            "neg_loglik_last": 90.0 + lambda_fuse,
            "primal_residual_last": 0.01 * (fold + 1),
            "dual_residual_last": 0.0 if fold == 0 else 0.001 * fold,
            "stopping_reason": "stagnated" if fold == 0 else "converged",
            "n_admm_iter": 20 + fold,
            "c_td": c_td_test,
            "c_td_train": c_td_train,
            "c_td_test": c_td_test,
        },
        "config": {"lambda_fuse": lambda_fuse},
    }
    with (result_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_real_cv_visualize_results_creates_tables_and_pngs(tmp_path: Path) -> None:
    base_dir = tmp_path / "cv"
    for lambda_fuse, offset in [(0.1, 0.0), (1.0, 0.02), (10.0, 0.01)]:
        _write_result(base_dir, lambda_fuse, 0, 0.61 + offset, 0.66 + offset)
        _write_result(base_dir, lambda_fuse, 1, 0.63 + offset, 0.67 + offset)

    fold_df, summary_df = load_or_collect_cv_results(base_dir)
    assert len(fold_df) == 6
    assert set(summary_df["lambda_fuse"]) == {0.1, 1.0, 10.0}

    fold_output = tmp_path / "fold_results.csv"
    summary_output = tmp_path / "summary_by_lambda.csv"
    write_cv_tables(fold_df, summary_df, fold_output, summary_output)
    assert fold_output.exists()
    assert summary_output.exists()

    output_dir = tmp_path / "plots"
    outputs = create_all_plots(fold_df, summary_df, output_dir)
    expected_names = {
        "cv_lambda_vs_c_td.png",
        "cv_train_test_c_td.png",
        "cv_fold_spaghetti.png",
        "cv_convergence_diagnostics.png",
    }
    assert {path.name for path in outputs} == expected_names
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0


def test_real_cv_visualize_results_accepts_cox_summary(tmp_path: Path) -> None:
    base_dir = tmp_path / "cv"
    for lambda_fuse, offset in [(0.1, 0.0), (1.0, 0.02)]:
        _write_result(base_dir, lambda_fuse, 0, 0.61 + offset, 0.66 + offset)
        _write_result(base_dir, lambda_fuse, 1, 0.63 + offset, 0.67 + offset)

    fold_df, summary_df = load_or_collect_cv_results(base_dir)
    cox_df = pd.DataFrame(
        {
            "dataset": ["support2"],
            "n_folds": [2],
            "c_td_test_cox_mean": [0.625],
            "c_td_test_cox_se": [0.004],
        }
    )

    output_dir = tmp_path / "plots_with_cox"
    outputs = create_all_plots(fold_df, summary_df, output_dir, cox_df=cox_df)

    assert {path.name for path in outputs} == {
        "cv_lambda_vs_c_td.png",
        "cv_train_test_c_td.png",
        "cv_fold_spaghetti.png",
        "cv_convergence_diagnostics.png",
    }
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0


def test_real_cv_visualize_results_accepts_aft_summary(tmp_path: Path) -> None:
    base_dir = tmp_path / "cv"
    for lambda_fuse, offset in [(0.1, 0.0), (1.0, 0.02)]:
        _write_result(base_dir, lambda_fuse, 0, 0.61 + offset, 0.66 + offset)
        _write_result(base_dir, lambda_fuse, 1, 0.63 + offset, 0.67 + offset)

    fold_df, summary_df = load_or_collect_cv_results(base_dir)
    aft_df = pd.DataFrame(
        {
            "dataset": ["support2", "support2"],
            "aft_model": ["weibull", "log_normal"],
            "n_folds": [2, 2],
            "c_td_test_aft_mean": [0.622, 0.618],
            "c_td_test_aft_se": [0.005, 0.006],
        }
    )

    output_dir = tmp_path / "plots_with_aft"
    outputs = create_all_plots(fold_df, summary_df, output_dir, aft_df=aft_df)

    assert {path.name for path in outputs} == {
        "cv_lambda_vs_c_td.png",
        "cv_train_test_c_td.png",
        "cv_fold_spaghetti.png",
        "cv_convergence_diagnostics.png",
        "cv_model_comparison.png",
    }
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0
