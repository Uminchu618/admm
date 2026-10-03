from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pilot.compare_fused_penalties import pair_methods, summarize_methods


def _records() -> pd.DataFrame:
    rows = []
    for method, c_td, rmise, detected, tp in (
        ("lasso", 0.60, 0.30, 4, 2),
        ("mcp", 0.61, 0.25, 3, 2),
    ):
        rows.append(
            {
                "method": method,
                "data_name": "small_seed_42",
                "scenario": "small",
                "seed": 42,
                "lambda_fuse": 0.1,
                "mcp_gamma": 3.0 if method == "mcp" else None,
                "cv_mean_c_td": c_td + 0.01,
                "c_td_train": c_td + 0.02,
                "c_td_test": c_td,
                "converged": True,
                "rmise": rmise,
                "true_positive": tp,
                "detected": detected,
                "truth": 3,
                "false_positive": detected - tp,
                "initialization_source": "default",
                "result_path": f"{method}.json",
            }
        )
    return pd.DataFrame(rows)


def test_penalty_comparison_reports_small_scenario_recovery() -> None:
    summary = summarize_methods(_records()).set_index("method")

    assert summary.loc["lasso", "precision"] == 0.5
    assert summary.loc["mcp", "precision"] == 2.0 / 3.0
    assert summary.loc["mcp", "recall"] == 2.0 / 3.0
    assert np.isclose(summary.loc["mcp", "c_td_test_mean"], 0.61)


def test_penalty_comparison_is_paired_by_scenario_and_seed() -> None:
    paired = pair_methods(_records())

    assert len(paired) == 1
    assert np.isclose(paired.loc[0, "mcp_minus_lasso_c_td_test"], 0.01)
    assert paired.loc[0, "mcp_minus_lasso_detected"] == -1
    assert paired.loc[0, "mcp_minus_lasso_false_positive"] == -1
