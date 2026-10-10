from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_main_predict_only_cli(tmp_path: Path) -> None:
    data_src = tmp_path / "data.csv"
    data_src.write_text(
        "id,k,time,event,x1\n"
        "1,0,0.5,1,0.2\n1,1,0.5,1,0.2\n1,2,0.5,1,0.2\n"
        "2,0,2.5,0,-0.3\n2,1,2.5,0,-0.3\n2,2,2.5,0,-0.3\n",
        encoding="utf-8",
    )
    # 過去の適応的 rho の設定を含む result.json も予測に利用できる。
    legacy_result = {
        "time_grid": [0.0, 1.0, 2.0, 3.0],
        "coef": [[0.1], [0.1], [0.1]],
        "gamma": [-1.0] * 6,
        "config": {"n_baseline_basis": 6},
    }
    legacy_result.setdefault("config", {}).update({
        "adaptive_rho": True,
        "rho_balance_mu": 10.0,
        "rho_increase_factor": 2.0,
        "rho_decrease_factor": 2.0,
        "rho_update_interval": 5,
        "rho_min": 1e-6,
        "rho_max": 1e6,
    })
    result_src = tmp_path / "legacy_result.json"
    result_src.write_text(json.dumps(legacy_result), encoding="utf-8")

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
