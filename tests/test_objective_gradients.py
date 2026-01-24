from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admm.baseline import BSplineBaseline
from admm.objective import HazardAFTObjective
from admm.quadrature import QuadratureRule
from admm.time_partition import TimePartition


def open_uniform_knots(
    n_basis: int, degree: int, x_min: float, x_max: float
) -> np.ndarray:
    if x_max <= x_min:
        raise ValueError("x_max must be > x_min")
    n_internal = n_basis - (degree + 1)
    if n_internal < 0:
        raise ValueError("n_basis must be at least degree + 1")
    if n_internal > 0:
        internal = np.linspace(x_min, x_max, n_internal + 2, dtype=float)[1:-1]
    else:
        internal = np.array([], dtype=float)
    left = np.full(degree + 1, x_min, dtype=float)
    right = np.full(degree + 1, x_max, dtype=float)
    return np.concatenate([left, internal, right])


def assert_allclose(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, name: str
) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.max(np.abs(a - b))
    denom = np.max(np.abs(b))
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f"{name}: contains NaN/inf")
    if diff > atol + rtol * denom:
        raise AssertionError(
            f"{name}: not close (max|a-b|={diff:.3e}, max|b|={denom:.3e}, atol={atol:.1e}, rtol={rtol:.1e})"
        )


def main() -> None:
    rng = np.random.default_rng(0)

    # --- 時間分割のスモークテスト（slides15.md の k(i) 定義） ---
    tp = TimePartition([0.0, 1.0, 2.0])
    T_test = np.array([0.0, 0.2, 0.999, 1.0, 1.5, 1.999, 2.0], dtype=float)
    k = tp.interval_index(T_test)
    # 定義: t_{k-1} <= T < t_k (1-based)
    # T==1.0 は第2区間、T==2.0 は境界（実装ではKに丸めて許容）
    if not np.array_equal(k, np.array([1, 1, 1, 2, 2, 2, 2], dtype=int)):
        raise AssertionError(f"interval_index unexpected: {k}")

    # --- Objective の数値微分テスト ---
    # 小さな次元で、finite difference で grad と Hessian(方向微分) を検証する。
    n = 6
    p = 2
    K = 2
    time_grid = [0.0, 1.0, 2.0]

    X = rng.normal(size=(n, p))
    T = rng.uniform(0.05, 1.95, size=n)
    delta = rng.integers(0, 2, size=n).astype(int)

    # beta は intercept 込み (K, p+1)
    beta = rng.normal(scale=0.1, size=(K, p + 1))

    # baseline の knots は x=exp(eta)*t の範囲をカバーするよう広めに取る
    # ここでは eta を小さくしているので exp(eta) は概ね ~[0.7, 1.4] 程度。
    x_min, x_max = 0.0, 4.0
    n_basis = 8
    degree = 3
    knots = open_uniform_knots(n_basis, degree, x_min, x_max).tolist()
    baseline = BSplineBaseline(
        n_basis=n_basis,
        degree=degree,
        knots=knots,
        extrapolate=False,
    )

    gamma = rng.normal(scale=0.1, size=n_basis)

    quad = QuadratureRule({"rule": "gauss_legendre", "Q": 15})
    time_partition = TimePartition(time_grid)
    obj = HazardAFTObjective(
        baseline=baseline,
        time_partition=time_partition,
        quadrature=quad,
        clip_eta=10.0,
    )

    eps = 1e-5

    # --- grad_beta ---
    g_beta = obj.grad_beta(beta, gamma, X, T, delta)
    g_beta_fd = np.zeros_like(g_beta)
    for k0 in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            step = np.zeros_like(beta)
            step[k0, j] = eps
            f_plus = obj.value(beta + step, gamma, X, T, delta)
            f_minus = obj.value(beta - step, gamma, X, T, delta)
            g_beta_fd[k0, j] = (f_plus - f_minus) / (2 * eps)

    assert_allclose(g_beta, g_beta_fd, atol=2e-4, rtol=2e-4, name="grad_beta")

    # --- grad_gamma ---
    g_gamma = obj.grad_gamma(beta, gamma, X, T, delta)
    g_gamma_fd = np.zeros_like(g_gamma)
    for m in range(gamma.size):
        step = np.zeros_like(gamma)
        step[m] = eps
        f_plus = obj.value(beta, gamma + step, X, T, delta)
        f_minus = obj.value(beta, gamma - step, X, T, delta)
        g_gamma_fd[m] = (f_plus - f_minus) / (2 * eps)

    assert_allclose(g_gamma, g_gamma_fd, atol=2e-4, rtol=2e-4, name="grad_gamma")

    # --- Hessian の方向微分チェック（gamma） ---
    H_gg = obj.hess_gamma(beta, gamma, X, T, delta)
    if H_gg.shape != (gamma.size, gamma.size):
        raise AssertionError(f"H_gg shape mismatch: {H_gg.shape}")
    assert_allclose(H_gg, H_gg.T, atol=1e-10, rtol=1e-10, name="H_gg symmetry")

    v = rng.normal(size=gamma.size)
    hv = H_gg @ v
    g_plus = obj.grad_gamma(beta, gamma + eps * v, X, T, delta)
    g_minus = obj.grad_gamma(beta, gamma - eps * v, X, T, delta)
    hv_fd = (g_plus - g_minus) / (2 * eps)
    assert_allclose(hv, hv_fd, atol=5e-4, rtol=5e-4, name="H_gg @ v")

    # --- Hessian の方向微分チェック（beta, block diagonal） ---
    H_bb = obj.hess_beta(beta, gamma, X, T, delta)
    if H_bb.shape != (K, p + 1, p + 1):
        raise AssertionError(f"H_bb shape mismatch: {H_bb.shape}")

    d = rng.normal(size=beta.shape)
    Hd = np.zeros_like(beta)
    for k0 in range(K):
        Hd[k0] = H_bb[k0] @ d[k0]

    g_plus = obj.grad_beta(beta + eps * d, gamma, X, T, delta)
    g_minus = obj.grad_beta(beta - eps * d, gamma, X, T, delta)
    Hd_fd = (g_plus - g_minus) / (2 * eps)
    assert_allclose(Hd, Hd_fd, atol=8e-4, rtol=8e-4, name="H_bb @ d")

    print("OK: objective gradients/hessians finite-difference checks passed")


if __name__ == "__main__":
    main()
