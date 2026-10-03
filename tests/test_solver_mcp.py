from __future__ import annotations

import numpy as np
import pytest

from admm.solver import FusedLassoADMMSolver, firm_threshold, mcp_penalty


class _FlatObjective:
    def value(self, beta, gamma, X, T, delta) -> float:
        return 0.0

    def grad_gamma(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.zeros_like(np.asarray(gamma, dtype=float))

    def hess_gamma(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.eye(np.asarray(gamma).size, dtype=float)

    def grad_beta(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.zeros_like(np.asarray(beta, dtype=float))

    def hess_beta(self, beta, gamma, X, T, delta) -> np.ndarray:
        beta_array = np.asarray(beta, dtype=float)
        K, p = beta_array.shape
        return np.repeat(np.eye(p, dtype=float)[None, :, :], K, axis=0)


def _solver(**overrides) -> FusedLassoADMMSolver:
    params = {
        "objective": _FlatObjective(),
        "lambda_fuse": 1.0,
        "rho": 2.0,
        "max_admm_iter": 1,
        "admm_tol_primal": 0.0,
        "admm_tol_dual": 0.0,
        "admm_tol_rel": 0.0,
        "admm_stagnation_tol": 0.0,
        "admm_stagnation_patience": 2,
        "newton_steps_per_admm": 1,
        "max_newton_iter": 1,
        "newton_tol": 0.0,
        "line_search_max_steps": 1,
        "line_search_shrink": 0.5,
        "line_search_c1": 1e-4,
        "return_best_iterate": False,
        "random_state": None,
        "fuse_penalty": "mcp",
        "mcp_gamma": 3.0,
    }
    params.update(overrides)
    return FusedLassoADMMSolver(**params)


def _inputs(n_samples: int = 1):
    beta0 = np.array([[0.0], [10.0]], dtype=float)
    gamma0 = np.zeros(1, dtype=float)
    X = np.zeros((n_samples, 2, 1), dtype=float)
    T = np.ones(n_samples, dtype=float)
    delta = np.zeros(n_samples, dtype=int)
    return beta0, gamma0, X, T, delta


def test_mcp_penalty_boundaries_are_continuous() -> None:
    values = np.array([0.0, 1.0, 6.0, 7.0])
    actual = mcp_penalty(values, lambda_fuse=2.0, gamma=3.0)
    expected = np.array([0.0, 2.0 - 1.0 / 6.0, 6.0, 6.0])
    assert np.allclose(actual, expected)


def test_firm_threshold_covers_both_boundaries_and_signs() -> None:
    # lambda/rho=1, gamma*lambda=6, denominator=5/6
    values = np.array([-7.0, -6.0, -1.5, -1.0, 0.0, 1.0, 1.5, 6.0, 7.0])
    actual = firm_threshold(values, lambda_fuse=2.0, gamma=3.0, rho=2.0)
    expected = np.array([-7.0, -6.0, -0.6, 0.0, 0.0, 0.0, 0.6, 6.0, 7.0])
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    ("gamma", "rho"),
    [(1.0, 2.0), (3.0, 1.0 / 3.0), (3.0, 0.0)],
)
def test_firm_threshold_rejects_invalid_curvature(gamma: float, rho: float) -> None:
    with pytest.raises(ValueError):
        firm_threshold([1.0], lambda_fuse=1.0, gamma=gamma, rho=rho)


def test_solver_uses_sample_scaled_lambda_for_mcp_z_step_and_objective() -> None:
    n_samples = 2
    solver = _solver(lambda_fuse=1.0, rho=2.0, mcp_gamma=3.0)

    _, _, z, _, history = solver.solve(*_inputs(n_samples))

    # effective lambda=2, upper MCP boundary=6; beta difference 10 is unshrunk.
    assert np.allclose(z, [[10.0]])
    assert history["penalty"][-1] == 6.0
    assert history["objective"][-1] == 6.0
    assert history["fuse_penalty"] == "mcp"
    assert history["mcp_gamma"] == 3.0
    assert history["mcp_convexity_margin"] == [2.0 - 1.0 / 3.0]


def test_solver_rejects_mcp_when_gamma_rho_is_not_greater_than_one() -> None:
    solver = _solver(rho=0.25, mcp_gamma=4.0)
    with pytest.raises(ValueError, match=r"mcp_gamma \* rho > 1"):
        solver.solve(*_inputs())


def test_adaptive_rho_never_crosses_mcp_curvature_boundary() -> None:
    solver = _solver(
        lambda_fuse=0.0,
        rho=1.0,
        mcp_gamma=2.0,
        adaptive_rho=True,
        rho_balance_mu=2.0,
        rho_decrease_factor=4.0,
        rho_update_interval=1,
        rho_min=1e-6,
        max_admm_iter=2,
    )

    _, _, _, _, history = solver.solve(*_inputs())

    assert all(rho > 0.5 for rho in history["rho"])
    assert history["rho_final"] > 0.5
