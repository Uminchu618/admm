from __future__ import annotations

import pandas as pd

from scripts.pilot.check_diagnostic import evaluate


def _summary() -> pd.DataFrame:
    rows = []
    for data_name in ("oracle_seed_42", "fine_grid_seed_42"):
        for lambda_fuse, n_change_points in ((0.0, 3), (0.01, 1)):
            rows.append(
                {
                    "data_name": data_name,
                    "lambda_fuse": lambda_fuse,
                    "converged": True,
                    "bic_eligible": True,
                    "bic": 100.0 + lambda_fuse,
                    "returned_iter": 4,
                    "returned_from": "converged_iterate",
                    "returned_primal_residual": 0.001,
                    "returned_dual_residual": 0.001,
                    "returned_primal_tolerance": 0.002,
                    "returned_dual_tolerance": 0.002,
                    "n_change_points": n_change_points,
                }
            )
    return pd.DataFrame(rows)


def test_diagnostic_gate_passes_complete_converged_path() -> None:
    report = evaluate(_summary(), expected_rows=4)

    assert report["passed"] is True
    assert report["converged_rows"] == 4


def test_diagnostic_gate_fails_nonconverged_row() -> None:
    summary = _summary()
    summary.loc[0, "converged"] = False

    report = evaluate(summary, expected_rows=4)

    assert report["passed"] is False
    assert report["checks"]["all_formally_converged"] is False
