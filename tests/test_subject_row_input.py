from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admm.model import ADMMHazardAFT
from main import _load_dataset, main
from scripts.real_cv.common import to_long_format


def _config() -> dict:
    return {
        "time_grid": list(np.linspace(0.0, 6.0, 7)),
        "n_baseline_basis": 6,
        "quadrature": {"rule": "gauss_legendre", "Q": 3},
        "lambda_fuse": 0.01,
        "max_admm_iter": 4,
        "newton_steps_per_admm": 1,
        "max_newton_iter": 1,
        "clip_eta": 5.0,
        "random_state": 1234,
    }


def _subjects() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(12, 5)), columns=["AGE", "SEX", "BMI", "SYSBP", "DIABP"])
    frame["SEX"] = rng.integers(0, 2, 12)
    frame.insert(0, "event", [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    frame.insert(0, "time", np.linspace(0.2, 6.0, 12))
    frame.insert(0, "id", np.arange(12))
    return frame


def test_k6_fit_predict_score_equivalence(tmp_path: Path) -> None:
    subjects = _subjects()
    feature_cols = list(subjects.columns[3:])
    row_path = tmp_path / "subjects.csv"
    long_path = tmp_path / "long.csv"
    subjects.sample(frac=1, random_state=1).to_csv(row_path, index=False)
    to_long_format(subjects, 6, feature_cols).sample(frac=1, random_state=2).to_csv(long_path, index=False)
    rows = _load_dataset(row_path)
    long = _load_dataset(long_path)
    assert rows.X.shape == (12, 5)
    assert long.X.shape == (12, 6, 5)
    np.testing.assert_array_equal(rows.y, long.y)
    np.testing.assert_array_equal(np.repeat(rows.X[:, None, :], 6, axis=1), long.X)

    row_model = ADMMHazardAFT.from_config(_config()).fit(rows.X, rows.y)
    long_model = ADMMHazardAFT.from_config(_config()).fit(long.X, long.y)
    assert row_model.coef_.shape == (6, 5)
    assert np.any(row_model.coef_ != 0)  # 初期値だけを比較しない
    for attribute in ("coef_", "gamma_", "z_", "u_"):
        np.testing.assert_allclose(getattr(row_model, attribute), getattr(long_model, attribute), rtol=1e-10, atol=1e-12)
    times = [0.0, 0.3, 1.0, 2.5, 4.7, 6.0]
    for method in ("predict_survival_function", "predict_cumulative_hazard"):
        np.testing.assert_allclose(
            getattr(row_model, method)(rows.X, times=times),
            getattr(long_model, method)(long.X, times=times),
            rtol=1e-10, atol=1e-12,
        )
    assert np.isfinite(row_model.score(rows.X, rows.y))
    assert row_model.score(rows.X, rows.y) == long_model.score(long.X, long.y)
    for key in ("objective", "neg_loglik", "primal_residual", "dual_residual"):
        np.testing.assert_allclose(row_model.history_[key], long_model.history_[key], rtol=1e-10, atol=1e-12)
    for key in ("stopping_reason", "n_admm_iter"):
        assert row_model.history_[key] == long_model.history_[key]


@pytest.mark.parametrize("mixed_eval", [False, True])
def test_k6_cli_equivalence(tmp_path: Path, monkeypatch, mixed_eval: bool) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENABLED", raising=False)
    subjects = _subjects()
    paths = {}
    for kind, frame in (
        ("row", subjects),
        ("long", to_long_format(subjects, 6, list(subjects.columns[3:]))),
    ):
        paths[kind] = tmp_path / f"{kind}.csv"
        frame.to_csv(paths[kind], index=False)
    # 新形式ではデータの旧 time_grid が実験設定を上書きしない。
    Path(f"{paths['row']}.meta.json").write_text(json.dumps({"time_grid": [0.0, 6.0]}))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()))
    results = {}
    predictions = {}
    for kind in ("row", "long"):
        eval_kind = ("long" if kind == "row" else "row") if mixed_eval else kind
        output = tmp_path / f"{kind}_result.json"
        main(["--config", str(config_path), "--data", str(paths[kind]),
              "--eval-data", str(paths[eval_kind]), "--output", str(output)])
        results[kind] = json.loads(output.read_text())
        prediction = tmp_path / f"{kind}_prediction.json"
        main(["--config", str(config_path), "--data", str(paths[kind]),
              "--load-result", str(output), "--predict-times", "0.3,1,2.5,6",
              "--output", str(prediction)])
        predictions[kind] = json.loads(prediction.read_text())
    for key in ("coef", "gamma", "z_last"):
        np.testing.assert_allclose(results["row"][key], results["long"][key], rtol=1e-10, atol=1e-12)
    for key in ("time_grid",):
        assert results["row"][key] == results["long"][key]
    for key in ("objective", "neg_loglik", "primal_residual", "dual_residual"):
        np.testing.assert_allclose(results["row"]["history"][key], results["long"]["history"][key], rtol=1e-10, atol=1e-12)
    for key in ("stopping_reason", "n_admm_iter"):
        assert results["row"]["history"][key] == results["long"]["history"][key]
    for key in ("c_td", "c_td_train", "c_td_test"):
        assert results["row"]["summary"][key] is not None
        assert results["row"]["summary"][key] == results["long"]["summary"][key]
    for key in ("survival", "cumulative_hazard"):
        np.testing.assert_allclose(predictions["row"][key], predictions["long"][key], rtol=1e-10, atol=1e-12)
    for key in ("summary", "n_features", "n_samples"):
        assert predictions["row"][key] == predictions["long"][key]
    assert results["row"]["n_features"] == 5
    assert results["row"]["n_samples"] == 12


@pytest.mark.parametrize("k", [1, 3, 6, 12])
def test_subject_input_uses_configured_k(k: int) -> None:
    X = np.ones((2, 5))
    model = ADMMHazardAFT(time_grid=np.linspace(0.0, 6.0, k + 1))
    prepared, _, _, grid = model._validate_inputs(X, [[1.0, 1], [6.0, 0]])
    assert prepared.shape == (2, k, 5)
    assert len(grid) == k + 1


@pytest.mark.parametrize("k", [1, 5, 7])
def test_long_input_k_mismatch_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="K.*time_grid"):
        ADMMHazardAFT.from_config(_config()).fit(np.ones((2, k, 5)), [[1, 1], [6, 0]])


def test_subject_csv_duplicate_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    pd.DataFrame({"id": [1, 1], "time": [1, 2], "event": [1, 0], "x": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique ids"):
        _load_dataset(path)
