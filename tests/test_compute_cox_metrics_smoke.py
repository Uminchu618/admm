from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_compute_cox_metrics_seed42_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data/extended_aft_step/extended_aft_step_seed_42.csv"
    if not data_path.exists():
        raise AssertionError(f"test data not found: {data_path}")

    output_path = tmp_path / "cox_seed42_summary.csv"

    command = [
        sys.executable,
        "scripts/compute_cox_metrics.py",
        "--data-file",
        str(data_path),
        "--output",
        str(output_path),
        "--skip-compare",
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
            "compute_cox_metrics execution failed:\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    if not output_path.exists():
        raise AssertionError("Cox summary CSV was not created")

    result = pd.read_csv(output_path)
    expected_cols = {
        "data_name",
        "n_samples",
        "n_features",
        "c_td_cox",
        "c_index_harrell",
    }
    missing = expected_cols - set(result.columns)
    if missing:
        raise AssertionError(f"Missing columns in output CSV: {sorted(missing)}")

    if len(result) != 1:
        raise AssertionError(f"Expected one row in Cox summary, got {len(result)}")

    c_td = float(result.loc[0, "c_td_cox"])
    c_harrell = float(result.loc[0, "c_index_harrell"])
    assert 0.0 <= c_td <= 1.0
    assert 0.0 <= c_harrell <= 1.0
