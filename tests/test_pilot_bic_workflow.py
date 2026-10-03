from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from scripts.pilot.aggregate_bic_selection import select_by_bic
from scripts.pilot.visualize_bic_selection import plot_summary, summarize_metrics


def test_bic_selection_uses_eligible_minimum_and_larger_lambda_on_tie() -> None:
    summary = pd.DataFrame(
        [
            {
                "data_name": "small_seed_42",
                "lambda_fuse": 0.01,
                "bic": 100.0,
                "bic_eligible": True,
                "converged": True,
                "result_path": "small/0.01/result.json",
            },
            {
                "data_name": "small_seed_42",
                "lambda_fuse": 0.03,
                "bic": 100.0,
                "bic_eligible": True,
                "converged": True,
                "result_path": "small/0.03/result.json",
            },
            {
                "data_name": "small_seed_42",
                "lambda_fuse": 0.1,
                "bic": 90.0,
                "bic_eligible": False,
                "converged": False,
                "result_path": "small/0.1/result.json",
            },
        ]
    )

    selected, audit = select_by_bic(
        summary, expected_lambdas=[0.01, 0.03, 0.1]
    )

    assert selected.iloc[0]["lambda_fuse"] == 0.03
    assert audit.iloc[0]["n_eligible"] == 2
    assert audit.iloc[0]["n_bic_ties"] == 2
    assert audit.iloc[0]["n_missing_lambdas"] == 0


def test_submit_bic_warm_path_uses_one_task_per_dataset(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    bin_dir = tmp_path / "bin"
    train_dir.mkdir()
    eval_dir.mkdir()
    bin_dir.mkdir()
    for index in range(3):
        name = f"oracle_seed_{42 + index}.csv"
        (train_dir / name).touch()
        (eval_dir / name).touch()

    capture = tmp_path / "qsub_args.txt"
    qsub = bin_dir / "qsub"
    qsub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"$QSUB_CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    qsub.chmod(0o755)
    grid = tmp_path / "lambda.json"
    grid.write_text(json.dumps({"lambda_values": [0.0, 0.1]}), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "QSUB_CAPTURE_PATH": str(capture),
            "PILOT_TRAIN_DIR": str(train_dir),
            "PILOT_EVAL_DIR": str(eval_dir),
            "PILOT_BIC_OUTPUT_DIR": str(tmp_path / "output"),
            "PILOT_LAMBDA_GRID": str(grid),
            "PILOT_EXPECTED_DATASETS": "3",
            "UV_BIN": "/usr/bin/true",
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts/pilot/submit_bic_warm_path.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args[args.index("-t") + 1] == "1-3:1"
    exported = args[args.index("-v") + 1]
    assert f"PILOT_BIC_OUTPUT_DIR={tmp_path / 'output'}" in exported
    assert f"PILOT_LAMBDA_GRID={grid}" in exported
    assert args[-1] == "qsub_pilot_bic_warm_path.sh"


def test_bic_truth_summary_reports_no_change_false_positives() -> None:
    metrics = pd.DataFrame(
        {
            "scenario": ["no_change", "no_change"],
            "lambda_fuse": [0.1, 0.03],
            "c_td_test": [0.6, 0.62],
            "rmise": [0.1, 0.2],
            "true_positive": [0, 0],
            "detected": [2, 1],
            "truth": [0, 0],
        }
    )

    summary = summarize_metrics(metrics)

    assert summary.iloc[0]["false_positive"] == 3
    assert summary.iloc[0]["change_points_mean"] == 1.5


def test_bic_performance_plot_smoke(tmp_path: Path) -> None:
    scenarios = ["oracle", "fine_grid", "off_grid", "small", "no_change"]
    metric_rows = []
    summary_rows = []
    for index, scenario in enumerate(scenarios):
        metric_rows.append(
            {
                "scenario": scenario,
                "lambda_fuse": 0.01 * index,
            }
        )
        summary_rows.append(
            {
                "scenario": scenario,
                "c_td_test_mean": 0.6 + 0.01 * index,
                "c_td_test_se": 0.01,
                "rmise_mean": 0.2 - 0.01 * index,
                "rmise_se": 0.01,
                "precision": 0.5,
                "recall": 0.6,
                "f1": 0.55,
                "false_positive": index,
            }
        )
    output = tmp_path / "bic.png"

    plot_summary(pd.DataFrame(metric_rows), pd.DataFrame(summary_rows), output)

    assert output.exists()
    assert output.stat().st_size > 0
