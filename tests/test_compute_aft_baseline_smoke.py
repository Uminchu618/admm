from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _write_long_csv(path: Path, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
    rows = []
    for i in range(X.shape[0]):
        for k in range(3):
            rows.append(
                {
                    "id": i + 1,
                    "k": k,
                    "time": float(time[i]),
                    "event": int(event[i]),
                    "x1": float(X[i, 0]),
                    "x2": float(X[i, 1]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_aft_baseline_seed42_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(42)

    X = rng.normal(size=(90, 2))
    linear = 0.45 * X[:, 0] - 0.25 * X[:, 1]
    survival_time = np.exp(1.2 + linear + rng.normal(scale=0.25, size=X.shape[0]))
    censor_time = rng.uniform(2.0, 8.0, size=X.shape[0])
    observed_time = np.minimum(survival_time, censor_time)
    event = (survival_time <= censor_time).astype(int)

    base_dir = tmp_path / "real_cv" / "support2" / "aft_smoke" / "lambda_0" / "fold_00"
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True)
    _write_long_csv(data_dir / "train.csv", X[:70], observed_time[:70], event[:70])
    _write_long_csv(data_dir / "test.csv", X[70:], observed_time[70:], event[70:])

    fold_output = tmp_path / "aft_fold_results.csv"
    summary_output = tmp_path / "aft_summary.csv"
    command = [
        sys.executable,
        "scripts/real_cv/compute_aft_baseline.py",
        "--base-dir",
        str(tmp_path / "real_cv" / "support2" / "aft_smoke"),
        "--aft-models",
        "weibull",
        "--fold-output",
        str(fold_output),
        "--summary-output",
        str(summary_output),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            "compute_aft_baseline execution failed:\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    result = pd.read_csv(fold_output)
    summary = pd.read_csv(summary_output)
    expected_cols = {
        "aft_model",
        "n_train",
        "n_features",
        "c_td_test_aft",
        "c_index_harrell_test",
    }
    missing = expected_cols - set(result.columns)
    if missing:
        raise AssertionError(f"Missing columns in output CSV: {sorted(missing)}")

    assert len(result) == 1
    assert len(summary) == 1
    assert result.loc[0, "aft_model"] == "weibull"
    assert 0.0 <= float(result.loc[0, "c_td_test_aft"]) <= 1.0
    assert 0.0 <= float(result.loc[0, "c_index_harrell_test"]) <= 1.0
