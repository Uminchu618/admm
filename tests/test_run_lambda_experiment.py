from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_lambda_experiment_passes_matching_eval_data(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    output_dir = tmp_path / "output"
    train_dir.mkdir()
    eval_dir.mkdir()
    (train_dir / "scenario_seed_7.csv").touch()
    (eval_dir / "scenario_seed_7.csv").touch()

    config_path = tmp_path / "config.toml"
    config_path.write_text("lambda_fuse = 10.0\n", encoding="utf-8")
    lambda_path = tmp_path / "lambda.json"
    lambda_path.write_text(json.dumps({"lambda_values": [1.0]}), encoding="utf-8")

    capture_path = tmp_path / "uv-args.txt"
    fake_uv = tmp_path / "fake-uv"
    fake_uv.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(train_dir),
            "EVAL_DATA_DIR": str(eval_dir),
            "OUTPUT_BASE_DIR": str(output_dir),
            "CONFIG_TEMPLATE": str(config_path),
            "LAMBDA_GRID_FILE": str(lambda_path),
            "UV_BIN": str(fake_uv),
            "CAPTURE_PATH": str(capture_path),
            "SGE_TASK_ID": "1",
        }
    )

    completed = subprocess.run(
        ["bash", "run_lambda_experiment.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    eval_flag_index = args.index("--eval-data")
    assert args[eval_flag_index + 1] == str(eval_dir / "scenario_seed_7.csv")
