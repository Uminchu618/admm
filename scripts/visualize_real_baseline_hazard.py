#!/usr/bin/env python3
"""Visualize estimated baseline hazard curves for real-data CV results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admm.baseline import BSplineBaseline


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_lambda_label(lambda_value: float) -> str:
    return f"{lambda_value:.12g}"


def format_lambda_filename(lambda_value: float) -> str:
    return format_lambda_label(lambda_value).replace(".", "p").replace("-", "m")


def collect_cv_results(cv_dir: Path) -> dict[float, list[tuple[int, Path, dict[str, Any]]]]:
    by_lambda: dict[float, list[tuple[int, Path, dict[str, Any]]]] = {}
    for result_path in sorted(cv_dir.glob("lambda_*/fold_*/result.json")):
        lambda_dir = result_path.parents[1]
        fold_dir = result_path.parent
        try:
            lambda_value = float(lambda_dir.name.removeprefix("lambda_"))
            fold_idx = int(fold_dir.name.removeprefix("fold_"))
        except ValueError as exc:
            raise ValueError(f"Unexpected CV result path: {result_path}") from exc
        by_lambda.setdefault(lambda_value, []).append(
            (fold_idx, result_path, load_result(result_path))
        )

    for lambda_value in by_lambda:
        by_lambda[lambda_value].sort(key=lambda item: item[0])
    return dict(sorted(by_lambda.items(), key=lambda item: item[0]))


def build_baseline(result: dict[str, Any], result_path: Path) -> tuple[BSplineBaseline, np.ndarray, np.ndarray, float]:
    gamma = np.asarray(result.get("gamma"), dtype=float).reshape(-1)
    time_grid = np.asarray(result.get("time_grid"), dtype=float)
    config = result.get("config")
    if gamma.ndim != 1 or gamma.size == 0:
        raise ValueError(f"gamma must be a non-empty 1D array in {result_path}")
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError(f"time_grid must be a 1D array with at least two points in {result_path}")
    if not isinstance(config, dict):
        raise ValueError(f"config must be present in {result_path}")

    n_basis = int(config.get("n_baseline_basis", gamma.size))
    if n_basis != gamma.size:
        raise ValueError(
            f"n_baseline_basis and gamma length mismatch in {result_path}: "
            f"{n_basis} vs {gamma.size}"
        )

    knot_margin = float(config.get("baseline_knot_margin", 1.1))
    clip_eta = float(config.get("clip_eta", 20.0))
    if knot_margin <= 0.0:
        raise ValueError(f"baseline_knot_margin must be positive in {result_path}")

    baseline = BSplineBaseline(
        n_basis=n_basis,
        degree=3,
        knots=None,
        knot_range=(0.0, float(time_grid[-1]) * knot_margin),
        extrapolate=False,
    )
    return baseline, gamma, time_grid, clip_eta


def evaluate_baseline_hazard(
    result: dict[str, Any],
    result_path: Path,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline, gamma, time_grid, clip_eta = build_baseline(result, result_path)
    t_min = max(float(time_grid[0]), 0.0)
    t_max = float(time_grid[-1])
    t = np.linspace(t_min, t_max, n_points, dtype=float)
    log_hazard = np.asarray(baseline.basis(t), dtype=float) @ gamma
    log_hazard = np.clip(log_hazard, -clip_eta, clip_eta)
    hazard = np.exp(log_hazard)
    return t, hazard, time_grid


def setup_axis(
    ax: plt.Axes,
    time_grid: np.ndarray,
    yscale: str,
    ymax: float | None = None,
) -> None:
    for boundary in time_grid:
        ax.axvline(boundary, color="#d9d9d9", linewidth=0.7, alpha=0.55)
    ax.set_yscale(yscale)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.set_xlabel("time")
    ax.set_ylabel("baseline hazard")
    ax.grid(True, linestyle=":", alpha=0.55)


def plot_lambda_curves(
    dataset_name: str,
    lambda_value: float,
    fold_results: list[tuple[int, Path, dict[str, Any]]],
    output_dir: Path,
    n_points: int,
    yscale: str,
    ymax: float | None,
) -> Path:
    first_t: np.ndarray | None = None
    first_time_grid: np.ndarray | None = None

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cmap = plt.get_cmap("tab10")
    for fold_idx, result_path, result in fold_results:
        t, hazard, time_grid = evaluate_baseline_hazard(result, result_path, n_points)
        if first_t is None:
            first_t = t
            first_time_grid = time_grid
        else:
            if not np.allclose(t, first_t):
                raise ValueError(f"time evaluation grid mismatch in {result_path}")
            if not np.allclose(time_grid, first_time_grid):
                raise ValueError(f"time_grid mismatch in {result_path}")

        ax.plot(
            t,
            hazard,
            linewidth=1.6,
            alpha=0.88,
            color=cmap(fold_idx % 10),
            label=f"fold {fold_idx:02d}",
        )

    assert first_time_grid is not None
    setup_axis(ax, first_time_grid, yscale, ymax)
    ax.set_title(
        f"{dataset_name} baseline hazard by fold, lambda={format_lambda_label(lambda_value)}"
    )
    ax.legend(loc="best", ncol=1)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{dataset_name}_lambda_{format_lambda_filename(lambda_value)}_baseline_hazard.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overview(
    dataset_name: str,
    by_lambda: dict[float, list[tuple[int, Path, dict[str, Any]]]],
    output_path: Path,
    n_points: int,
    yscale: str,
    ymax: float | None,
) -> Path:
    n_lambdas = len(by_lambda)
    n_cols = min(5, n_lambdas)
    n_rows = int(math.ceil(n_lambdas / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.7 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    cmap = plt.get_cmap("tab10")

    for panel_idx, (lambda_value, fold_results) in enumerate(by_lambda.items()):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = axes[row, col]
        first_time_grid: np.ndarray | None = None
        for fold_idx, result_path, result in fold_results:
            t, hazard, time_grid = evaluate_baseline_hazard(result, result_path, n_points)
            if first_time_grid is None:
                first_time_grid = time_grid
            elif not np.allclose(time_grid, first_time_grid):
                raise ValueError(f"time_grid mismatch in {result_path}")
            ax.plot(
                t,
                hazard,
                linewidth=1.15,
                alpha=0.82,
                color=cmap(fold_idx % 10),
                label=f"fold {fold_idx:02d}" if panel_idx == 0 else None,
            )
        assert first_time_grid is not None
        setup_axis(ax, first_time_grid, yscale, ymax)
        ax.set_title(f"lambda={format_lambda_label(lambda_value)}")

    for panel_idx in range(n_lambdas, n_rows * n_cols):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        axes[row, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle(f"{dataset_name} baseline hazard by lambda and fold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dataset(
    dataset_name: str,
    cv_dir: Path,
    output_dir: Path,
    n_points: int,
    yscale: str,
    ymax: float | None,
) -> list[Path]:
    by_lambda = collect_cv_results(cv_dir)
    if not by_lambda:
        raise ValueError(f"No CV result.json files found under {cv_dir}")

    paths: list[Path] = []
    dataset_dir = output_dir / "cv_by_lambda" / dataset_name
    for lambda_value, fold_results in by_lambda.items():
        paths.append(
            plot_lambda_curves(
                dataset_name=dataset_name,
                lambda_value=lambda_value,
                fold_results=fold_results,
                output_dir=dataset_dir,
                n_points=n_points,
                yscale=yscale,
                ymax=ymax,
            )
        )

    paths.append(
        plot_overview(
            dataset_name=dataset_name,
            by_lambda=by_lambda,
            output_path=output_dir / f"{dataset_name}_baseline_hazard_overview.png",
            n_points=n_points,
            yscale=yscale,
            ymax=ymax,
        )
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize real-data CV baseline hazard curves by lambda and fold."
    )
    parser.add_argument(
        "--framingham-cv-dir",
        type=Path,
        default=Path("outputs")
        / "real_cv"
        / "framingham"
        / "framingham_5fold_seed1234",
        help="Framingham CV experiment directory",
    )
    parser.add_argument(
        "--support-cv-dir",
        type=Path,
        default=Path("outputs") / "real_cv" / "support2" / "support2_5fold_seed1234",
        help="Support2 CV experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "real_visualizations" / "baseline_hazard",
        help="Output directory",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=400,
        help="Number of time points used to draw each curve",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale for baseline hazard",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional upper limit for the baseline hazard y-axis",
    )
    args = parser.parse_args()

    outputs = []
    outputs.extend(
        plot_dataset(
            dataset_name="framingham",
            cv_dir=args.framingham_cv_dir,
            output_dir=args.output_dir,
            n_points=args.n_points,
            yscale=args.yscale,
            ymax=args.ymax,
        )
    )
    outputs.extend(
        plot_dataset(
            dataset_name="support2",
            cv_dir=args.support_cv_dir,
            output_dir=args.output_dir,
            n_points=args.n_points,
            yscale=args.yscale,
            ymax=args.ymax,
        )
    )

    for path in outputs:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
