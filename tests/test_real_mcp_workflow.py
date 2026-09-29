from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import matplotlib.pyplot as plt

from scripts.real_cv.validate_mcp_results import validate
from scripts.real_cv.visualize_mcp_beta import main as beta_main
from scripts.real_cv.visualize_results import _set_lambda_axis

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["framingham", "support2"])
def test_parallel_submission_uses_nine_lambdas_and_five_folds(
    tmp_path: Path, dataset: str
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is unavailable")
    qsub = tmp_path / "qsub"
    qsub.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    qsub.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": f"{tmp_path}:{env['PATH']}", "UV_BIN": uv})

    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/real_cv/mcp_workflow.sh"), "submit", dataset],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = completed.stdout.splitlines()
    assert args[args.index("-t") + 1] == "1-45:1"
    assert args[-1] == "qsub_real_mcp_cv.sh"
    exported = args[args.index("-v") + 1]
    assert f"DATASET={dataset}" in exported
    assert "EXPERIMENT_NAME=mcp_5fold_seed1234" in exported
    assert "config_real_mcp.toml" in exported
    assert "generation/pilot/lambda_grid.json" in exported


def test_warm_submission_is_separate_five_task_experiment(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is unavailable")
    qsub = tmp_path / "qsub"
    qsub.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    qsub.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": f"{tmp_path}:{env['PATH']}", "UV_BIN": uv})
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/real_cv/mcp_workflow.sh"), "submit-warm", "support2"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = completed.stdout.splitlines()
    assert args[args.index("-t") + 1] == "1-5:1"
    assert args[-1] == "qsub_real_mcp_warm.sh"
    assert "EXPERIMENT_NAME=mcp_5fold_seed1234_warm" in args[args.index("-v") + 1]


def test_warm_runner_uses_previous_lambda_result_only_after_first_fit(tmp_path: Path) -> None:
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"lambda_values": [0.1, 0.25]}), encoding="utf-8")
    split = tmp_path / "splits.csv"
    split.write_text("id,fold\n1,0\n2,1\n", encoding="utf-8")
    trace = tmp_path / "trace.txt"
    stub = tmp_path / "uv-stub"
    stub.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        "if [ \"$2\" = python ]; then printf '0.25\\n0.1\\n'; exit 0; fi\n"
        "if [ \"$2\" = scripts/real_cv/prepare_fold.py ]; then exit 0; fi\n"
        "if [ \"$2\" = main.py ]; then\n"
        "  printf '%s\\n' \"$*\" >> \"$TRACE_FILE\"\n"
        "  while [ \"$#\" -gt 0 ]; do\n"
        "    if [ \"$1\" = --output ]; then shift; mkdir -p \"$(dirname \"$1\")\"; printf '{}' > \"$1\"; break; fi\n"
        "    shift\n"
        "  done\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "UV_BIN": str(stub),
            "LAMBDA_GRID": str(grid),
            "SPLITS_FILE": str(split),
            "OUTPUT_BASE_DIR": str(tmp_path / "outputs"),
            "TRACE_FILE": str(trace),
            "DATASET": "framingham",
            "N_FOLDS": "2",
        }
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts/real_cv/run_mcp_warm_path.sh"), "1"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "lambda_0.25/fold_00" in calls[0]
    assert "--init-result" not in calls[0]
    assert "lambda_0.1/fold_00" in calls[1]
    assert "--init-result" in calls[1]
    assert "lambda_0.25/fold_00/result.json" in calls[1]


def test_validation_finds_missing_or_wrong_penalty_results(tmp_path: Path) -> None:
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"lambda_values": [0, 0.1]}), encoding="utf-8")
    base = tmp_path / "cv"
    for value in [0, 0.1]:
        for fold in range(2):
            path = base / f"lambda_{value:.15g}" / f"fold_{fold:02d}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"config": {"fuse_penalty": "mcp", "lambda_fuse": value}}),
                encoding="utf-8",
            )
    assert validate(base, grid, 2) == 4
    (base / "lambda_0.1/fold_01/result.json").unlink()
    with pytest.raises(RuntimeError, match="1 missing"):
        validate(base, grid, 2)
    wrong = base / "lambda_0.1/fold_01/result.json"
    wrong.write_text(
        json.dumps({"config": {"fuse_penalty": "lasso", "lambda_fuse": 0.1}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="1 invalid"):
        validate(base, grid, 2)


def test_zero_and_small_lambdas_use_readable_axis() -> None:
    fig, ax = plt.subplots()
    _set_lambda_axis(ax, [0.0, 0.0001, 0.001, 0.01, 0.25])
    assert ax.get_xscale() == "symlog"
    assert "0" in [tick.get_text() for tick in ax.get_xticklabels()]
    plt.close(fig)


def test_selected_beta_plot_uses_cv_selected_lambda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "mcp"
    base.mkdir()
    (base / "selected_lambda.json").write_text(
        json.dumps({"selection_method": "five_fold_cv_mean_c_td", "selected_lambda": 0.1}),
        encoding="utf-8",
    )
    for value in [0.0, 0.1]:
        for fold in [0, 1]:
            path = base / f"lambda_{value:.15g}" / f"fold_{fold:02d}" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "coef": [[value + fold], [value + fold + 1]],
                        "time_grid": [0, 1, 2],
                        "feature_cols": ["age"],
                    }
                ),
                encoding="utf-8",
            )
    monkeypatch.setattr(
        "sys.argv",
        ["visualize_mcp_beta.py", "--dataset", "framingham", "--base-dir", str(base)],
    )
    beta_main()
    assert len(list((base / "plots/beta_by_lambda").glob("*.png"))) == 2
    assert len(list((base / "plots/selected_beta").glob("*.png"))) == 1
