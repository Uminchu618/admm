from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admm.model import ADMMHazardAFT


def main() -> None:
    rng = np.random.default_rng(0)

    n = 12
    p = 2

    X = rng.normal(size=(n, p))
    T = rng.uniform(0.05, 1.95, size=n)
    delta = rng.integers(0, 2, size=n).astype(int)
    y = np.column_stack([T, delta])

    time_grid = [0.0, 1.0, 2.0]

    model = ADMMHazardAFT(
        time_grid=time_grid,
        baseline_basis="bspline",
        n_baseline_basis=8,
        quadrature={"rule": "gauss_legendre", "Q": 5},
        lambda_fuse=0.1,
        rho=1.0,
        max_admm_iter=2,
        admm_tol_primal=0.0,
        admm_tol_dual=0.0,
        newton_steps_per_admm=1,
        max_newton_iter=1,
        newton_tol=0.0,
        clip_eta=5.0,
        random_state=None,
    )

    fitted = model.fit(X, y)

    if not hasattr(fitted, "coef_"):
        raise AssertionError("coef_ not set")
    if not hasattr(fitted, "gamma_"):
        raise AssertionError("gamma_ not set")
    if fitted.coef_.shape != (len(time_grid) - 1, p):
        raise AssertionError(f"coef_ shape mismatch: {fitted.coef_.shape}")
    if fitted.gamma_.shape != (8,):
        raise AssertionError(f"gamma_ shape mismatch: {fitted.gamma_.shape}")
    if "objective" not in fitted.history_:
        raise AssertionError("history_ missing objective")
    if len(fitted.history_["objective"]) == 0:
        raise AssertionError("history_ objective empty")

    print("OK: ADMMHazardAFT.fit smoke test passed")


if __name__ == "__main__":
    main()
