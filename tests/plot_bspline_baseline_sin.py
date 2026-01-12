from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for visualization
    plt = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admm.baseline import BSplineBaseline


def open_uniform_knots(
    n_basis: int, degree: int, knot_range: Tuple[float, float]
) -> np.ndarray:
    x_min, x_max = knot_range
    if x_max <= x_min:
        raise ValueError("knot_range must satisfy x_max > x_min")
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run(
    n_basis: int,
    degree: int,
    x_min: float,
    x_max: float,
    n_points: int,
    output: Optional[Path],
    show: bool,
) -> None:
    x = np.linspace(x_min, x_max, n_points, dtype=float, endpoint=False)
    y = np.sin(x)
    dy = np.cos(x)
    ddy = -np.sin(x)

    knots = open_uniform_knots(n_basis, degree, (x_min, x_max))
    baseline = BSplineBaseline(
        n_basis=n_basis,
        degree=degree,
        knots=knots,
        extrapolate=False,
    )

    basis = baseline.basis(x)
    gamma, *_ = np.linalg.lstsq(basis, y, rcond=None)

    y_hat = basis @ gamma
    dy_hat = baseline.basis_deriv(x) @ gamma
    ddy_hat = baseline.basis_second_deriv(x) @ gamma

    print(f"RMSE sin(x): {rmse(y, y_hat):.6e}")
    print(f"RMSE cos(x): {rmse(dy, dy_hat):.6e}")
    print(f"RMSE -sin(x): {rmse(ddy, ddy_hat):.6e}")

    if plt is None:
        print("matplotlib is not available. Skip plotting.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    _add_knot_guides(axes, knots, x_min, x_max, degree)
    axes[0].plot(x, y, label="sin(x)", color="black")
    axes[0].plot(x, y_hat, label="bspline", color="C1")
    axes[0].set_ylabel("value")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(x, dy, label="cos(x)", color="black")
    axes[1].plot(x, dy_hat, label="bspline deriv", color="C1")
    axes[1].set_ylabel("first deriv")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].plot(x, ddy, label="-sin(x)", color="black")
    axes[2].plot(x, ddy_hat, label="bspline second deriv", color="C1")
    axes[2].set_ylabel("second deriv")
    axes[2].set_xlabel("x")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if output is not None:
        if output.is_dir():
            output = output / "bspline_sin.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"Saved plot to {output}")

    if show:
        plt.show()


def _add_knot_guides(
    axes: np.ndarray, knots: np.ndarray, x_min: float, x_max: float, degree: int
) -> None:
    unique_knots = np.unique(knots)
    internal = unique_knots[(unique_knots > x_min) & (unique_knots < x_max)]
    endpoints = np.array([x_min, x_max], dtype=float)

    for i, ax in enumerate(axes):
        label = "knots" if i == 0 else None
        if internal.size > 0:
            ax.vlines(
                internal,
                0,
                1,
                transform=ax.get_xaxis_transform(),
                color="0.6",
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
                label=label,
            )
        ax.vlines(
            endpoints,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            color="0.4",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.text(
            0.99,
            0.02,
            f"degree={degree}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.4",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit sin(x) with BSplineBaseline and visualize derivatives."
    )
    parser.add_argument("--n-basis", type=int, default=12)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=-3.0)
    parser.add_argument("--x-max", type=float, default=3.0)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bspline_sin.png"),
        help="Output image path.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()
    output = None if args.no_save else args.output

    run(
        n_basis=args.n_basis,
        degree=args.degree,
        x_min=args.x_min,
        x_max=args.x_max,
        n_points=args.n_points,
        output=output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
