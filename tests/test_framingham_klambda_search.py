from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.real_cv.klambda_search import (
    aggregate, prepare, read_json, refit, run_task, select_pair, submit,
    task_config, validate_grid, write_json,
)


@pytest.fixture
def search(tmp_path: Path) -> Path:
    raw = tmp_path / "framingham.csv"
    n = 12
    pd.DataFrame(dict(RANDID=range(1, n + 1), PERIOD=[1] * n,
                      AGE=np.arange(n) + 40, SEX=np.arange(n) % 2 + 1,
                      BMI=np.arange(n) * 0.3 + 22, SYSBP=np.arange(n) * 2 + 110,
                      DIABP=np.arange(n) * 1.5 + 70,
                      TIMEHYP=[8766, 1200, 2400, 3600, 8766, 4800] * 2)).to_csv(raw, index=False)
    splits = tmp_path / "splits.csv"
    pd.DataFrame(dict(id=range(1, n + 1), fold=np.arange(n) % 2)).to_csv(splits, index=False)
    config = tmp_path / "config.json"
    write_json(config, dict(time_grid=[0, 1, 2, 3, 4, 5, 6], random_state=1234,
                            n_baseline_basis=6, lambda_fuse=0.1, max_admm_iter=1,
                            newton_steps_per_admm=1, max_newton_iter=1, clip_eta=5,
                            quadrature=dict(rule="gauss_legendre", Q=3)))
    grid = tmp_path / "grid.json"
    write_json(grid, dict(time_range=[0, 6], k_values=[1, 6], lambda_values=[0, 0.1]))
    directory = tmp_path / "search"
    prepare(SimpleNamespace(search_dir=directory, input=raw, splits=splits,
                            config=config, grid=grid, n_folds=2))
    return directory


def _args(search: Path, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(search_dir=search, tie_tolerance=1e-12, **kwargs)


def _fake_result(search: Path, task: dict, score=0.7, converged=True) -> None:
    manifest = read_json(search / "manifest.json")
    config = task_config(manifest, task)
    meta = manifest["fold_summaries"][str(task["fold"])]
    output = search / task["directory"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "result.json", dict(config=config, time_grid=config["time_grid"],
                n_samples=meta["n_train"], n_eval_samples=meta["n_test"],
                feature_cols=manifest["feature_cols"], eval_data_path="test.csv",
                summary=dict(c_td_test=score, c_td_train=0.8, converged=converged,
                    objective_last=10.0, primal_residual_last=0.0, dual_residual_last=0.0)))


def test_prepare_shares_data_and_maps_all_tasks(search: Path) -> None:
    manifest = read_json(search / "manifest.json")
    tasks = manifest["tasks"]
    assert len(tasks) == 8
    assert [(t["K"], t["lambda_fuse"], t["fold"]) for t in tasks] == [
        (1, 0.0, 0), (1, 0.0, 1), (1, 0.1, 0), (1, 0.1, 1),
        (6, 0.0, 0), (6, 0.0, 1), (6, 0.1, 0), (6, 0.1, 1)]
    for fold in range(2):
        train = pd.read_csv(search / "prepared" / f"fold_{fold:02d}" / "train.csv")
        test = pd.read_csv(search / "prepared" / f"fold_{fold:02d}" / "test.csv")
        assert "k" not in train.columns
        assert len(train) == len(test) == 6
        assert set(train.id).isdisjoint(test.id)
        np.testing.assert_allclose(train[["AGE", "BMI", "SYSBP", "DIABP"]].mean(), 0, atol=1e-14)
        raw_times = dict(zip(range(1, 13), [8766, 1200, 2400, 3600, 8766, 4800] * 2))
        np.testing.assert_allclose(train.time, train.id.map(raw_times) * 6 / 8766)
        assert manifest["fold_summaries"][str(fold)]["time_scale_max_original"] == 8766
    assert not list(search.glob("K_*"))


@pytest.mark.parametrize("task_id", [1, 5])
def test_run_actual_task_k1_and_k6_and_resume(search: Path, task_id: int, monkeypatch) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENABLED", raising=False)
    run_task(_args(search, task_id=task_id, skip_existing=False))
    task = read_json(search / "manifest.json")["tasks"][task_id - 1]
    result = read_json(search / task["directory"] / "result.json")
    assert np.asarray(result["coef"]).shape == (task["K"], 5)
    assert result["n_samples"] == result["n_eval_samples"] == 6
    assert isinstance(result["summary"]["c_td_test"], float)
    assert (search / task["directory"] / "runtime.json").exists()
    assert not (search / task["directory"] / ".running").exists()
    run_task(_args(search, task_id=task_id, skip_existing=True))
    with pytest.raises(FileExistsError):
        run_task(_args(search, task_id=task_id, skip_existing=False))


def test_pending_final_selection_and_refit_config(search: Path, monkeypatch) -> None:
    tasks = read_json(search / "manifest.json")["tasks"]
    empty = aggregate(_args(search))
    assert empty["selection_method"] == "pending_cv_completion"
    for task in tasks[:2]:
        _fake_result(search, task)
    pending = aggregate(_args(search))
    assert pending["selected_K"] is None
    assert pending["provisional_K"] == 1
    with pytest.raises(ValueError, match="completed CV"):
        refit(_args(search))
    # 同点では小さいK、同じKでは大きいlambdaを選ぶ。
    # 高スコアでも非収束のK=6/lambda=0は除外する。
    for task in tasks[2:]:
        bad = task["K"] == 6 and task["lambda_fuse"] == 0
        _fake_result(search, task, score=0.99 if bad else 0.7, converged=not bad)
    final = aggregate(_args(search))
    assert final["selection_method"] == "joint_cv_mean_c_td"
    assert final["selected_K"] == 1
    assert final["selected_lambda"] == 0.1
    assert final["time_grid"] == [0.0, 6.0]
    assert (search / "K_lambda_vs_c_td.png").exists()
    calls = []
    monkeypatch.setattr("scripts.real_cv.klambda_search.run_fit", lambda *args: calls.append(args))
    refit(_args(search))
    assert calls[0][1]["time_grid"] == [0.0, 6.0]
    assert calls[0][1]["lambda_fuse"] == 0.1
    assert calls[0][2] == search / "prepared/all.csv"
    assert calls[0][3] is None


def test_complete_but_no_eligible_candidate(search: Path) -> None:
    for task in read_json(search / "manifest.json")["tasks"]:
        _fake_result(search, task, score=None)
    result = aggregate(_args(search))
    assert result["selection_method"] == "no_eligible_candidate"
    assert result["selected_K"] is None
    with pytest.raises(ValueError):
        refit(_args(search))


def test_modified_data_and_mismatched_results_rejected(search: Path) -> None:
    tasks = read_json(search / "manifest.json")["tasks"]
    _fake_result(search, tasks[0])
    path = search / tasks[0]["directory"] / "result.json"
    result = read_json(path)
    result["config"]["lambda_fuse"] = 10
    write_json(path, result)
    with pytest.raises(ValueError, match="config/time_grid mismatch"):
        aggregate(_args(search))
    data = search / "prepared/fold_00/train.csv"
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="Prepared data changed"):
        run_task(_args(search, task_id=1, skip_existing=True))


def test_submit_computes_task_count(search: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("scripts.real_cv.klambda_search.subprocess.run", lambda *a, **kw: calls.append((a, kw)))
    submit(_args(search, refit=False, max_concurrent=3))
    command = calls[0][0][0]
    assert command[command.index("-t") + 1] == "1-8"
    assert command[command.index("-tc") + 1] == "3"


@pytest.mark.parametrize("update", [dict(k_values=[0]), dict(k_values=[6, 6]),
    dict(k_values=[3.5]), dict(lambda_values=[float("nan")]), dict(time_range=[1, 6])])
def test_invalid_grid_rejected(update: dict) -> None:
    grid = dict(k_values=[6], lambda_values=[0.1], time_range=[0, 6])
    grid.update(update)
    with pytest.raises(ValueError):
        validate_grid(grid)


def test_nonfinite_and_ineligible_scores_cannot_win() -> None:
    summary = pd.DataFrame(dict(K=[3, 6, 12], lambda_fuse=[0.1] * 3,
        cv_eligible=[True, True, False], c_td_test_mean=[0.6, np.inf, 0.99]))
    selected = select_pair(summary)
    assert selected.loc[selected.selected, "K"].tolist() == [3]
