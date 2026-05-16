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


def test_main_eval_data_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    time_grid = [0.0, 1.0, 2.0]

    X_train = rng.normal(size=(8, 2, 2))
    y_train = np.array(
        [
            [0.2, 1],
            [0.5, 0],
            [0.8, 1],
            [1.1, 1],
            [1.3, 0],
            [1.5, 1],
            [1.7, 0],
            [1.9, 1],
        ],
        dtype=float,
    )
    X_test = rng.normal(size=(5, 2, 2))
    y_test = np.array(
        [
            [0.3, 1],
            [0.7, 0],
            [1.0, 1],
            [1.4, 0],
            [1.8, 1],
        ],
        dtype=float,
    )

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "result.json"

    _write_long_csv(train_path, X_train, y_train)
    _write_long_csv(test_path, X_test, y_test)
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
                "random_state": 0,
            }
        ),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
        "--data",
        str(train_path),
        "--eval-data",
        str(test_path),
        "--output",
        str(output_path),
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
            "CLI execution with --eval-data failed:\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["eval_data_path"] == str(test_path)
    assert payload["n_samples"] == X_train.shape[0]
    assert payload["n_eval_samples"] == X_test.shape[0]
    assert payload["summary"]["c_td_train"] is not None
    assert payload["summary"]["c_td_test"] is not None
