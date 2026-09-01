"""モデル選択で用いる自由度と BIC の共通計算。"""

from __future__ import annotations

import math
from typing import Any


def count_change_points(z: Any, tol: float = 1e-8) -> int | None:
    """補助変数 ``z = D beta`` に含まれる非ゼロ差分の個数を返す。"""

    if z is None:
        return None
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    try:
        return sum(
            abs(float(value)) > tol
            for row in z
            for value in row
        )
    except (TypeError, ValueError):
        return None


def effective_degrees_of_freedom(
    *,
    n_baseline_basis: Any,
    n_features: Any,
    z: Any,
    z_tol: float = 1e-8,
) -> int | None:
    """``M + p + #change-points`` による実効自由度を返す。"""

    n_change_points = count_change_points(z, z_tol)
    if (
        n_baseline_basis is None
        or n_features is None
        or n_change_points is None
    ):
        return None

    try:
        m = int(n_baseline_basis)
        p = int(n_features)
    except (TypeError, ValueError):
        return None
    if m < 0 or p < 0:
        return None
    return m + p + n_change_points


def compute_bic(
    *, neg_loglik: Any, n_samples: Any, degrees_of_freedom: Any
) -> float | None:
    """負の対数尤度から ``2*NLL + df*log(n)`` を計算する。"""

    if neg_loglik is None or n_samples is None or degrees_of_freedom is None:
        return None
    try:
        n = int(n_samples)
        if n <= 0:
            return None
        return 2.0 * float(neg_loglik) + float(degrees_of_freedom) * math.log(n)
    except (TypeError, ValueError):
        return None
