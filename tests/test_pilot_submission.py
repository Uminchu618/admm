from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def test_submit_uses_derived_4500_task_array_for_five_fold_cv(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    bin_dir = tmp_path / "bin"
    train_dir.mkdir()
    eval_dir.mkdir()
    bin_dir.mkdir()

    for index in range(100):
        (train_dir / f"dataset_{index:03d}.csv").touch()
        (eval_dir / f"dataset_{index:03d}.csv").touch()

    capture_path = tmp_path / "qsub_args.txt"
    qsub_path = bin_dir / "qsub"
    qsub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"$QSUB_CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    qsub_path.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "QSUB_CAPTURE_PATH": str(capture_path),
            "PILOT_TRAIN_DIR": str(train_dir),
            "PILOT_EVAL_DIR": str(eval_dir),
            "PILOT_OUTPUT_DIR": str(tmp_path / "output"),
            "PILOT_RUN_NAME": "test_run",
            "UV_BIN": "/usr/bin/true",
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts" / "pilot" / "submit.sh")],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    task_flag_index = args.index("-t")
    assert args[task_flag_index + 1] == "1-4500:1"
    exported = args[args.index("-v") + 1]
    assert "PILOT_CONFIG_TEMPLATE=" in exported
    assert "generation/pilot/diagnostic_config.toml" in exported
    assert "PILOT_LAMBDA_GRID=" in exported
    assert "generation/pilot/lambda_grid.json" in exported
    assert "PILOT_RUN_NAME=test_run" in exported
    assert "PILOT_N_FOLDS=5" in exported
    assert "PILOT_SPLIT_SEED=1234" in exported
    assert args[-1] == "qsub_pilot.sh"


def test_qsub_script_does_not_hardcode_array_size() -> None:
    root = Path(__file__).resolve().parents[1]
    qsub_script = (root / "qsub_pilot.sh").read_text(encoding="utf-8")
    assert not any(
        line.startswith("#$ -t ") for line in qsub_script.splitlines()
    )
    assert "1-1200" not in qsub_script


def test_submit_refined_cv_uses_generated_manifest_size(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "qsub_args.txt"
    qsub_path = bin_dir / "qsub"
    qsub_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"$QSUB_CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    qsub_path.chmod(0o755)

    selections_path = tmp_path / "cv_selections.csv"
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    pd.DataFrame(
        {
            "data_name": ["oracle_seed_42", "fine_grid_seed_42"],
            "selected_lambda": [0.03, 0.1],
        }
    ).to_csv(selections_path, index=False)
    (train_dir / "oracle_seed_42.csv").touch()
    (train_dir / "fine_grid_seed_42.csv").touch()
    uv_bin = shutil.which("uv")
    assert uv_bin is not None

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "QSUB_CAPTURE_PATH": str(capture_path),
            "PILOT_CV_SELECTIONS": str(selections_path),
            "PILOT_TRAIN_DIR": str(train_dir),
            "PILOT_OUTPUT_DIR": str(tmp_path / "coarse"),
            "PILOT_REFINED_OUTPUT_DIR": str(tmp_path / "refined"),
            "PILOT_REFINED_ADDITIONS_DIR": str(tmp_path / "additions"),
            "UV_BIN": uv_bin,
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts" / "pilot" / "submit_refined_cv.sh")],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    task_flag_index = args.index("-t")
    assert args[task_flag_index + 1] == "1-210:1"
    exported = args[args.index("-v") + 1]
    assert "PILOT_REFINED_MANIFEST=" in exported
    assert "PILOT_REFINED_ADDITIONS_DIR=" in exported
    assert args[-1] == "qsub_pilot_refined_cv.sh"


def test_refined_task_runner_skips_reusable_result(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    train_dir = tmp_path / "train"
    additions_dir = tmp_path / "additions"
    train_dir.mkdir()
    (train_dir / "oracle_seed_42.csv").touch()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "task_id": [1],
            "data_name": ["oracle_seed_42"],
            "coarse_selected_lambda": [0.03],
            "grid_index": [10],
            "lambda_fuse": [0.03],
            "fold": [0],
            "output_path": ["unused-by-runner"],
        }
    ).to_csv(manifest, index=False)
    result = (
        additions_dir
        / "oracle_seed_42"
        / "lambda_0.03"
        / "fold_00"
        / "result.json"
    )
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"summary": {"converged": True, "c_td_test": 0.7}}),
        encoding="utf-8",
    )
    uv_bin = shutil.which("uv")
    assert uv_bin is not None
    env = os.environ.copy()
    env.update(
        {
            "UV_BIN": uv_bin,
            "PILOT_TRAIN_DIR": str(train_dir),
            "PILOT_REFINED_ADDITIONS_DIR": str(additions_dir),
            "PILOT_REFINED_MANIFEST": str(manifest),
        }
    )

    completed = subprocess.run(
        ["bash", str(root / "scripts" / "pilot" / "run_refined_cv_task.sh"), "1"],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Skip reusable result" in completed.stdout
