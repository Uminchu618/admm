from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _write_long_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    rows = []
    for i in range(X.shape[0]):
        for k in range(X.shape[1]):
            rows.append(
                {
                    "id": i + 1,
                    "k": k,
                    "time": float(y[i, 0]),
                    "event": int(y[i, 1]),
                    "x1": float(X[i, k, 0]),
                    "x2": float(X[i, k, 1]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_bootstrap_parameter_ci_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    time_grid = [0.0, 1.0, 2.0]
    X = rng.normal(size=(8, 2, 2))
    y = np.array(
        [
            [0.2, 1],
            [0.5, 1],
            [0.8, 1],
            [1.0, 0],
            [1.2, 1],
            [1.5, 0],
            [1.7, 1],
            [1.9, 1],
        ],
        dtype=float,
    )

    data_path = tmp_path / "train.csv"
    config_path = tmp_path / "config.json"
    output_json = tmp_path / "bootstrap_ci.json"
    _write_long_csv(data_path, X, y)
    config_path.write_text(
        json.dumps(
            {
                "time_grid": time_grid,
                "baseline_basis": "bspline",
                "n_baseline_basis": 6,
                "baseline_knot_margin": 1.1,
                "quadrature": {"rule": "gauss_legendre", "Q": 3},
                "lambda_fuse": 0.1,
                "rho": 1.0,
                "max_admm_iter": 1,
                "admm_tol_primal": 0.0,
                "admm_tol_dual": 0.0,
                "newton_steps_per_admm": 1,
                "max_newton_iter": 1,
                "newton_tol": 0.0,
                "line_search_max_steps": 2,
                "line_search_shrink": 0.5,
                "line_search_c1": 1e-4,
                "return_best_iterate": True,
                "clip_eta": 5.0,
                "random_state": None,
            }
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "scripts/bootstrap_parameter_ci.py",
        "--config",
        str(config_path),
        "--data",
        str(data_path),
        "--n-bootstrap",
        "3",
        "--random-state",
        "11",
        "--output-json",
        str(output_json),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            "bootstrap parameter CI CLI failed:\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    coef_csv = Path(payload["coef_ci_csv"])
    gamma_csv = Path(payload["gamma_ci_csv"])

    assert payload["n_bootstrap_requested"] == 3
    assert payload["n_bootstrap_success"] >= 1
    assert payload["feature_cols"] == ["x1", "x2"]
    assert len(payload["coef_ci"]) == 4
    assert len(payload["gamma_ci"]) == 6
    assert coef_csv.exists()
    assert gamma_csv.exists()
    assert {"ci_lower", "ci_upper", "estimate"}.issubset(
        pd.read_csv(coef_csv).columns
    )
