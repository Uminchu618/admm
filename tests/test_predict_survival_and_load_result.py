from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admm.model import ADMMHazardAFT


def test_predict_survival_and_load_result_json(tmp_path: Path) -> None:
    time_grid = [0.0, 1.0, 2.0, 3.0]
    K = len(time_grid) - 1
    p = 2

    model = ADMMHazardAFT(
        time_grid=time_grid,
        baseline_basis="bspline",
        n_baseline_basis=8,
        quadrature={"rule": "gauss_legendre", "Q": 5},
        clip_eta=5.0,
    )

    coef = np.array(
        [
            [0.05, -0.02],
            [0.03, 0.01],
            [0.01, 0.00],
        ],
        dtype=float,
    )
    gamma = np.array([-2.0, -1.0, -0.5, 0.0, -0.2, 0.1, -0.4, -1.5], dtype=float)

    result_path = tmp_path / "result.json"
    result_payload = {
        "time_grid": time_grid,
        "coef": coef.tolist(),
        "gamma": gamma.tolist(),
        "z_last": np.zeros((p, K - 1), dtype=float).tolist(),
        "history": {"objective": [123.0]},
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, ensure_ascii=False)

    model.load_params_from_result_json(result_path)

    X_static = np.array(
        [
            [0.1, -0.3],
            [0.5, 0.2],
        ],
        dtype=float,
    )
    times = [0.5, 1.5, 2.5]

    survival = model.predict_survival_function(X_static, times=times)
    cumulative = model.predict_cumulative_hazard(X_static, times=times)
    y_eval = np.array(
        [
            [0.5, 1],
            [2.5, 0],
        ],
        dtype=float,
    )
    c_td = model.score(X_static, y_eval)

    assert survival.shape == (2, 3)
    assert cumulative.shape == (2, 3)
    assert np.all(np.isfinite(survival))
    assert np.all(np.isfinite(cumulative))
    assert np.all((survival >= 0.0) & (survival <= 1.0))

    # 累積ハザードは時間とともに単調非減少になる。
    assert np.all(np.diff(cumulative, axis=1) >= -1e-10)

    # S(t) = exp(-Lambda(t)) の一致性を確認する。
    assert np.allclose(survival, np.exp(-cumulative), atol=1e-12, rtol=1e-10)
    assert np.isfinite(c_td)
    assert 0.0 <= c_td <= 1.0
