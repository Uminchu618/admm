from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def _write_capture_uv(path: Path, *, delegate_inline_python: bool = False) -> None:
    delegate = (
        'if [ "$1" = "run" ] && [ "$2" = "python" ] && [ "$3" = "-" ]; then\n'
        '  exec "$REAL_UV" "$@"\n'
        "fi\n"
        if delegate_inline_python
        else ""
    )
    path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        + delegate
        + "printf 'CALL\\n' >> \"$CAPTURE_PATH\"\n"
        + "printf '%s\\n' \"$@\" >> \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_simulation_cv_task_maps_lambda_and_fold_without_independent_eval(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    (data_dir / "scenario_seed_42.csv").touch()
    config = tmp_path / "config.toml"
    config.write_text("lambda_fuse = 0.0\n", encoding="utf-8")
    lambda_grid = tmp_path / "lambda.json"
    lambda_grid.write_text(
        json.dumps({"lambda_values": [0.01, 0.1]}), encoding="utf-8"
    )
    capture = tmp_path / "capture.txt"
    fake_uv = tmp_path / "fake-uv"
    _write_capture_uv(fake_uv)

    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(data_dir),
            "OUTPUT_BASE_DIR": str(tmp_path / "output"),
            "CONFIG_TEMPLATE": str(config),
            "LAMBDA_GRID_FILE": str(lambda_grid),
            "N_FOLDS": "5",
            "SGE_TASK_ID": "7",
            "UV_BIN": str(fake_uv),
            "CAPTURE_PATH": str(capture),
        }
    )
    completed = subprocess.run(
        ["bash", "run_simulation_cv_experiment.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Lambda: 0.1" in completed.stdout
    assert "Fold: 1 / 5" in completed.stdout
    captured = capture.read_text(encoding="utf-8")
    assert "--eval-data" in captured
    assert "fold_01/data/test.csv" in captured
    assert "extended_aft_step_eval" not in captured


def test_real_full_fit_reads_cv_selected_lambda(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    selection = tmp_path / "selected_lambda.json"
    selection.write_text(
        json.dumps(
            {
                "selection_method": "five_fold_cv_mean_c_td",
                "selected_lambda": 0.03,
                "n_folds": 5,
            }
        ),
        encoding="utf-8",
    )
    raw_data = tmp_path / "support2.csv"
    raw_data.touch()
    config = tmp_path / "config.toml"
    config.touch()
    capture = tmp_path / "capture.txt"
    fake_uv = tmp_path / "fake-uv"
    _write_capture_uv(fake_uv, delegate_inline_python=True)
    real_uv = shutil.which("uv")
    assert real_uv is not None

    env = os.environ.copy()
    env.update(
        {
            "DATASETS": "support2",
            "SUPPORT2_INPUT": str(raw_data),
            "CONFIG_PATH": str(config),
            "CV_SELECTION_FILE": str(selection),
            "OUTPUT_BASE_DIR": str(tmp_path / "output"),
            "SGE_TASK_ID": "1",
            "UV_BIN": str(fake_uv),
            "REAL_UV": real_uv,
            "CAPTURE_PATH": str(capture),
        }
    )
    completed = subprocess.run(
        ["bash", "run_real_full_experiment.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Lambda selection: cv" in completed.stdout
    assert "Lambda: 0.03" in completed.stdout
    lines = capture.read_text(encoding="utf-8").splitlines()
    lambda_flag = lines.index("--lambda-fuse")
    assert lines[lambda_flag + 1] == "0.03"
    selection_flag = lines.index("--selection-file")
    assert lines[selection_flag + 1] == str(selection)
