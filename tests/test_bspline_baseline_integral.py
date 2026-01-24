from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admm.baseline import BSplineBaseline
from admm.quadrature import QuadratureRule


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


def fit_sin_baseline(
    n_basis: int, degree: int, x_min: float, x_max: float, n_points: int
):
    x = np.linspace(x_min, x_max, n_points, dtype=float, endpoint=False)
    y = np.sin(x)

    knots = open_uniform_knots(n_basis, degree, x_min, x_max)
    baseline = BSplineBaseline(
        n_basis=n_basis,
        degree=degree,
        knots=knots,
        extrapolate=False,
    )

    basis = baseline.basis(x)
    gamma, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return baseline, gamma


def quad_integral(
    baseline: BSplineBaseline,
    gamma: np.ndarray,
    a: float,
    b: float,
    *,
    rule: str,
    q: int,
) -> float:
    quad = QuadratureRule({"rule": rule, "Q": q})
    v, w = quad.nodes_weights(a, b)
    return float((baseline.basis(v) @ gamma) @ w)


def main() -> None:
    # 「ベースラインの積分」を QuadratureRule で評価し、解析解と突き合わせるスモークテスト。
    # ここでは spline 近似誤差も入るため、閾値は過度に厳しくしない。
    n_basis = 12
    degree = 3
    x_min = -3.0
    x_max = 3.0
    n_points = 400

    baseline, gamma = fit_sin_baseline(n_basis, degree, x_min, x_max, n_points)

    true_integral = float(np.cos(x_min) - np.cos(x_max))

    # Gauss-Legendre
    approx_gl = quad_integral(
        baseline, gamma, x_min, x_max, rule="gauss_legendre", q=40
    )
    abs_err_gl = abs(true_integral - approx_gl)

    # Simpson（Q は奇数）
    approx_sp = quad_integral(baseline, gamma, x_min, x_max, rule="simpson", q=51)
    abs_err_sp = abs(true_integral - approx_sp)

    # 近似としてそれなりに合っていればOK（将来、高速化や基底変更で多少変動しても壊れにくい閾値）
    tol = 5e-3
    if abs_err_gl > tol:
        raise AssertionError(
            f"Gauss-Legendre integral error too large: abs_err={abs_err_gl:.6e} > {tol:.2e}"
        )
    if abs_err_sp > tol:
        raise AssertionError(
            f"Simpson integral error too large: abs_err={abs_err_sp:.6e} > {tol:.2e}"
        )

    print(
        "OK: baseline integral via QuadratureRule "
        f"(abs_err_gl={abs_err_gl:.3e}, abs_err_sp={abs_err_sp:.3e}, true={true_integral:.6e})"
    )


if __name__ == "__main__":
    main()
