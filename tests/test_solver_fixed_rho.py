from __future__ import annotations

import numpy as np
import pytest

from admm.model import ADMMHazardAFT
from admm.solver import FusedLassoADMMSolver


class _QuadraticBetaObjective:
    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)

    def value(self, beta, gamma, X, T, delta) -> float:
        diff = np.asarray(beta, dtype=float) - self.target
        return 0.5 * float(np.sum(diff * diff))

    def grad_gamma(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.zeros_like(np.asarray(gamma, dtype=float))

    def hess_gamma(self, beta, gamma, X, T, delta) -> np.ndarray:
        size = np.asarray(gamma, dtype=float).size
        return np.eye(size, dtype=float)

    def grad_beta(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.asarray(beta, dtype=float) - self.target

    def hess_beta(self, beta, gamma, X, T, delta) -> np.ndarray:
        beta_array = np.asarray(beta, dtype=float)
        K, p = beta_array.shape
        return np.repeat(np.eye(p, dtype=float)[None, :, :], K, axis=0)


def _solver(*, max_admm_iter: int) -> FusedLassoADMMSolver:
    return FusedLassoADMMSolver(
        objective=_QuadraticBetaObjective(np.array([[0.0], [10.0]])),
        lambda_fuse=100.0,
        rho=1.0,
        max_admm_iter=max_admm_iter,
        admm_tol_primal=0.0,
        admm_tol_dual=0.0,
        admm_tol_rel=0.0,
        admm_stagnation_tol=0.0,
        admm_stagnation_patience=10,
        newton_steps_per_admm=1,
        max_newton_iter=1,
        newton_tol=0.0,
        line_search_max_steps=4,
        line_search_shrink=0.5,
        line_search_c1=1e-4,
        return_best_iterate=True,
        random_state=None,
    )


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta0 = np.zeros((2, 1), dtype=float)
    gamma0 = np.zeros(1, dtype=float)
    X = np.zeros((1, 2, 1), dtype=float)
    T = np.ones(1, dtype=float)
    delta = np.zeros(1, dtype=int)
    return beta0, gamma0, X, T, delta


def test_initial_best_is_not_returned_as_estimate() -> None:
    beta, _, _, _, history = _solver(max_admm_iter=1).solve(*_inputs())

    assert history["best_iter"] is None
    assert history["returned_iter"] == 0
    assert history["returned_from"] == "last_iterate"
    assert history["used_best_iterate"] is False
    assert not np.allclose(beta, 0.0)
    assert history["converged"] is False
    assert history["bic_eligible"] is False
    assert history["returned_neg_loglik"] == history["neg_loglik"][0]


def test_converged_terminal_iterate_is_returned_with_matching_metrics() -> None:
    solver = _solver(max_admm_iter=2)
    solver.lambda_fuse = 0.0
    beta0, gamma0, X, T, delta = _inputs()
    beta0 = np.array([[0.0], [10.0]], dtype=float)

    beta, _, z, _, history = solver.solve(beta0, gamma0, X, T, delta)

    assert history["stopping_reason"] == "residual_converged"
    assert history["returned_from"] == "converged_iterate"
    assert history["returned_iter"] == history["n_admm_iter"] - 1
    assert history["used_best_iterate"] is False
    assert history["converged"] is True
    assert history["bic_eligible"] is True
    assert history["returned_primal_residual"] == history["primal_residual"][-1]
    assert np.allclose(z, np.diff(beta, axis=0).T)


@pytest.mark.parametrize("fuse_penalty", ["lasso", "mcp"])
def test_rho_stays_fixed_when_primal_residual_dominates(fuse_penalty: str) -> None:
    solver = _solver(max_admm_iter=5)
    solver.fuse_penalty = fuse_penalty

    _, _, _, _, history = solver.solve(*_inputs())

    assert history["n_admm_iter"] == 5
    assert history["stopping_reason"] == "max_iter"
    assert all(value > 0.0 for value in history["primal_residual"])
    assert history["dual_residual"] == [0.0] * 5
    assert history["rho"] == [solver.rho] * 5
    assert history["rho_final"] == solver.rho
    assert history["returned_rho"] == solver.rho


def test_stagnation_stops_without_changing_rho() -> None:
    solver = _solver(max_admm_iter=8)
    solver.admm_tol_primal = 0.5
    solver.admm_tol_dual = 0.5
    solver.admm_stagnation_tol = 1.0
    solver.admm_stagnation_patience = 1
    solver.newton_tol = 100.0

    _, _, _, _, history = solver.solve(*_inputs())

    assert history["stopping_reason"] == "stagnated"
    assert history["n_admm_iter"] == 2
    assert history["stagnation_count"][-1] == 1
    assert history["rho"] == [solver.rho] * 2


@pytest.mark.parametrize("adaptive_rho", [False, True])
def test_legacy_config_uses_fixed_rho_without_mutating_input(adaptive_rho: bool) -> None:
    config = {
        "time_grid": [0.0, 1.0, 2.0],
        "rho": 2.0,
        "max_admm_iter": 2,
        "adaptive_rho": adaptive_rho,
        "rho_balance_mu": 10.0,
        "rho_increase_factor": 2.0,
        "rho_decrease_factor": 2.0,
        "rho_update_interval": 1,
        "rho_min": 1e-6,
        "rho_max": 1e6,
    }
    original_config = dict(config)
    model = ADMMHazardAFT.from_config(config)
    model.fit(np.zeros((3, 1)), np.array([[0.5, 1], [1.0, 0], [1.5, 1]]))

    assert config == original_config
    assert model.rho_ == model.rho == 2.0
    assert model.history_["rho"] == [2.0] * model.history_["n_admm_iter"]
    assert model.history_["returned_rho"] == model.rho_
    assert not hasattr(model, "adaptive_rho")


def test_from_config_still_rejects_unknown_keys() -> None:
    with pytest.raises(TypeError, match="unknown_setting"):
        ADMMHazardAFT.from_config({"unknown_setting": True})
