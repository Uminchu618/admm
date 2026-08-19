from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_submit_uses_derived_900_task_array(tmp_path: Path) -> None:
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
    assert args[task_flag_index + 1] == "1-900:1"
    exported = args[args.index("-v") + 1]
    assert "PILOT_CONFIG_TEMPLATE=" in exported
    assert "generation/pilot/diagnostic_config.toml" in exported
    assert "PILOT_LAMBDA_GRID=" in exported
    assert "generation/pilot/lambda_grid.json" in exported
    assert "PILOT_RUN_NAME=test_run" in exported
    assert args[-1] == "qsub_pilot.sh"


def test_qsub_script_does_not_hardcode_array_size() -> None:
    root = Path(__file__).resolve().parents[1]
    qsub_script = (root / "qsub_pilot.sh").read_text(encoding="utf-8")
    assert not any(
        line.startswith("#$ -t ") for line in qsub_script.splitlines()
    )
    assert "1-1200" not in qsub_script
