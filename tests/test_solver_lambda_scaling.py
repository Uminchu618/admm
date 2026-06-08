from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admm.solver import FusedLassoADMMSolver


class _FlatObjective:
    def value(self, beta, gamma, X, T, delta) -> float:
        return 0.0

    def grad_gamma(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.zeros_like(np.asarray(gamma, dtype=float))

    def hess_gamma(self, beta, gamma, X, T, delta) -> np.ndarray:
        gamma_arr = np.asarray(gamma, dtype=float).reshape(-1)
        return np.eye(gamma_arr.size, dtype=float)

    def grad_beta(self, beta, gamma, X, T, delta) -> np.ndarray:
        return np.zeros_like(np.asarray(beta, dtype=float))

    def hess_beta(self, beta, gamma, X, T, delta) -> np.ndarray:
        beta_arr = np.asarray(beta, dtype=float)
        K, p = beta_arr.shape
        return np.repeat(np.eye(p, dtype=float)[None, :, :], K, axis=0)


def test_lambda_fuse_is_scaled_by_sample_size_in_z_update() -> None:
    n_samples = 5
    lambda_fuse = 1.0
    rho = 2.0
    beta0 = np.array([[0.0], [10.0]], dtype=float)
    gamma0 = np.array([0.0], dtype=float)
    X = np.zeros((n_samples, 2, 1), dtype=float)
    T = np.ones(n_samples, dtype=float)
    delta = np.zeros(n_samples, dtype=int)

    solver = FusedLassoADMMSolver(
        objective=_FlatObjective(),
        lambda_fuse=lambda_fuse,
        rho=rho,
        max_admm_iter=1,
        admm_tol_primal=0.0,
        admm_tol_dual=0.0,
        admm_tol_rel=0.0,
        admm_stagnation_tol=0.0,
        admm_stagnation_patience=2,
        newton_steps_per_admm=1,
        max_newton_iter=1,
        newton_tol=0.0,
        line_search_max_steps=1,
        line_search_shrink=0.5,
        line_search_c1=1e-4,
        return_best_iterate=False,
        random_state=None,
    )

    _, _, z, _, history = solver.solve(beta0, gamma0, X, T, delta)

    expected_threshold = n_samples * lambda_fuse / rho
    assert np.allclose(z, [[10.0 - expected_threshold]])
    assert history["lambda_fuse_scale"] == n_samples
    assert history["lambda_fuse_effective"] == n_samples * lambda_fuse
    assert history["objective"][-1] == n_samples * lambda_fuse * 10.0
