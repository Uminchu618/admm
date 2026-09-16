from __future__ import annotations

import numpy as np
import pytest

from admm.model import ADMMHazardAFT


def _model() -> ADMMHazardAFT:
    return ADMMHazardAFT(
        time_grid=[0.0, 1.0, 2.0],
        n_baseline_basis=5,
        quadrature={"rule": "gauss_legendre", "Q": 2},
        lambda_fuse=0.01,
        rho=1.0,
        max_admm_iter=1,
        admm_tol_primal=0.0,
        admm_tol_dual=0.0,
        admm_tol_rel=0.0,
        newton_steps_per_admm=1,
        max_newton_iter=1,
        newton_tol=0.0,
        clip_eta=5.0,
        random_state=None,
    )


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(6, 2, 1))
    y = np.array(
        [[0.2, 1], [0.5, 0], [0.8, 1], [1.2, 1], [1.5, 0], [1.8, 1]],
        dtype=float,
    )
    return X, y


def test_fit_records_provided_warm_start() -> None:
    X, y = _data()
    model = _model().fit(
        X,
        y,
        beta0=np.zeros((2, 1), dtype=float),
        gamma0=np.zeros(5, dtype=float),
    )

    assert model.history_["initialization"] == {
        "beta": "provided",
        "gamma": "provided",
    }


def test_fit_rejects_warm_start_shape_mismatch() -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="beta0 の形状"):
        _model().fit(X, y, beta0=np.zeros((3, 1), dtype=float))


def test_hazard_aft_model_runs_with_mcp_penalty() -> None:
    X, y = _data()
    model = _model()
    model.fuse_penalty = "mcp"
    model.mcp_gamma = 3.0

    fitted = model.fit(X, y)

    assert fitted.history_["fuse_penalty"] == "mcp"
    assert fitted.history_["mcp_gamma"] == 3.0
    assert len(fitted.history_["penalty"]) == 1
    assert np.allclose(fitted.history_["mcp_convexity_margin"], [2.0 / 3.0])
