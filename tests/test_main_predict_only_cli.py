from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_main_predict_only_cli(tmp_path: Path) -> None:
    data_src = Path("data/simulated_data.csv")
    if not data_src.exists():
        raise AssertionError("test data not found: data/simulated_data.csv")

    result_src = Path("outputs/result.json")
    if not result_src.exists():
        raise AssertionError("result json not found: outputs/result.json")

    out_path = tmp_path / "predict_only_output.json"

    command = [
        sys.executable,
        "main.py",
        "--data",
        str(data_src),
        "--load-result",
        str(result_src),
        "--predict-times",
        "1.0,2.0,3.0",
        "--output",
        str(out_path),
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
            "CLI predict-only execution failed:\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    if not out_path.exists():
        raise AssertionError("predict-only output json was not created")

    with out_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["mode"] == "predict_only"
    assert payload["predict_times"] == [1.0, 2.0, 3.0]
    assert "summary" in payload
    assert "c_td" in payload["summary"]
    assert "survival" in payload
    assert "cumulative_hazard" in payload
    assert len(payload["survival"]) > 0
    assert len(payload["cumulative_hazard"]) > 0
