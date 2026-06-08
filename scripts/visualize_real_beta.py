#!/usr/bin/env python3
"""Framingham と Support2 の推定 beta を可視化するスクリプト。"""

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


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_feature_names(result: dict[str, Any], n_features: int) -> list[str]:
    feature_cols = result.get("feature_cols")
    if isinstance(feature_cols, list) and len(feature_cols) == n_features:
        return [str(name) for name in feature_cols]
    return [f"x{i + 1}" for i in range(n_features)]


def format_lambda_label(lambda_value: float) -> str:
    return f"{lambda_value:.12g}"


def format_lambda_filename(lambda_value: float) -> str:
    return format_lambda_label(lambda_value).replace(".", "p").replace("-", "m")


def validate_coef_result(result: dict[str, Any], result_path: Path) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(result.get("coef"), dtype=float)
    time_grid = np.asarray(result.get("time_grid"), dtype=float)
    if coef.ndim != 2:
        raise ValueError(f"coef must be 2D in {result_path}")
    if time_grid.ndim != 1:
        raise ValueError(f"time_grid must be 1D in {result_path}")
    if time_grid.size != coef.shape[0] + 1:
        raise ValueError(
            f"time_grid length must be coef rows + 1 in {result_path}: "
            f"{time_grid.size} vs {coef.shape[0]}"
        )
    return coef, time_grid


def plot_beta_trajectories(result_path: Path, output_path: Path) -> Path:
    result = load_result(result_path)

    coef, time_grid = validate_coef_result(result, result_path)
    feature_names = resolve_feature_names(result, coef.shape[1])
    time_mid = (time_grid[:-1] + time_grid[1:]) / 2.0

    n_features = coef.shape[1]
    n_cols = min(4, n_features)
    n_rows = int(math.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 2.8 * n_rows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    dataset_name = result_path.stem.replace("_result", "")
    for idx, feature_name in enumerate(feature_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.plot(
            time_mid,
            coef[:, idx],
            marker="o",
            linewidth=1.8,
            color="#2c7fb8",
        )
        for boundary in time_grid:
            ax.axvline(boundary, color="#d9d9d9", linewidth=0.7, alpha=0.65)
        ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.7)
        ax.set_title(feature_name)
        ax.grid(True, linestyle=":", alpha=0.5)

    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("time midpoint")
    for row_axes in axes[:, 0]:
        row_axes.set_ylabel("beta")

    fig.suptitle(f"Estimated beta trajectories: {dataset_name}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
        result = load_result(result_path)
        by_lambda.setdefault(lambda_value, []).append((fold_idx, result_path, result))

    for lambda_value in by_lambda:
        by_lambda[lambda_value].sort(key=lambda item: item[0])
    return dict(sorted(by_lambda.items(), key=lambda item: item[0]))


def plot_cv_beta_trajectories_by_lambda(
    dataset_name: str,
    cv_dir: Path,
    output_dir: Path,
) -> list[Path]:
    by_lambda = collect_cv_results(cv_dir)
    if not by_lambda:
        raise ValueError(f"No CV result.json files found under {cv_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for lambda_value, fold_results in by_lambda.items():
        first_fold, first_path, first_result = fold_results[0]
        first_coef, time_grid = validate_coef_result(first_result, first_path)
        feature_names = resolve_feature_names(first_result, first_coef.shape[1])
        time_mid = (time_grid[:-1] + time_grid[1:]) / 2.0

        n_features = first_coef.shape[1]
        n_cols = min(4, n_features)
        n_rows = int(math.ceil(n_features / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.4 * n_cols, 2.8 * n_rows),
            sharex=True,
        )
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

        cmap = plt.get_cmap("tab10")
        for fold_idx, result_path, result in fold_results:
            coef, fold_time_grid = validate_coef_result(result, result_path)
            fold_feature_names = resolve_feature_names(result, coef.shape[1])
            if coef.shape != first_coef.shape:
                raise ValueError(
                    f"coef shape mismatch in {result_path}: "
                    f"{coef.shape} vs {first_coef.shape}"
                )
            if not np.allclose(fold_time_grid, time_grid):
                raise ValueError(f"time_grid mismatch in {result_path}")
            if fold_feature_names != feature_names:
                raise ValueError(f"feature_cols mismatch in {result_path}")

            color = cmap(fold_idx % 10)
            for feature_idx, feature_name in enumerate(feature_names):
                row = feature_idx // n_cols
                col = feature_idx % n_cols
                ax = axes[row, col]
                ax.plot(
                    time_mid,
                    coef[:, feature_idx],
                    marker="o",
                    linewidth=1.2,
                    markersize=3.0,
                    alpha=0.85,
                    color=color,
                    label=f"fold {fold_idx:02d}" if feature_idx == 0 else None,
                )
                ax.set_title(feature_name)
                ax.grid(True, linestyle=":", alpha=0.5)

        for feature_idx in range(n_features):
            row = feature_idx // n_cols
            col = feature_idx % n_cols
            ax = axes[row, col]
            for boundary in time_grid:
                ax.axvline(boundary, color="#d9d9d9", linewidth=0.7, alpha=0.45)
            ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.7)

        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("time midpoint")
        for row_axes in axes[:, 0]:
            row_axes.set_ylabel("beta")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(handles),
                bbox_to_anchor=(0.5, 0.985),
            )
        fig.suptitle(
            f"{dataset_name} beta trajectories by fold, lambda={format_lambda_label(lambda_value)}",
            y=1.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        output_path = (
            output_dir
            / f"{dataset_name}_lambda_{format_lambda_filename(lambda_value)}_beta_trajectories.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framingham と Support2 の推定 beta を可視化する。"
    )
    parser.add_argument(
        "--framingham-result",
        type=Path,
        default=Path("outputs") / "framingham_result.json",
        help="Framingham の result.json",
    )
    parser.add_argument(
        "--support-result",
        type=Path,
        default=Path("outputs") / "support2_result.json",
        help="Support2 の result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "real_visualizations" / "beta",
        help="画像の保存先ディレクトリ",
    )
    parser.add_argument(
        "--framingham-cv-dir",
        type=Path,
        default=Path("outputs")
        / "real_cv"
        / "framingham"
        / "framingham_5fold_seed1234",
        help="Framingham CV の実験ディレクトリ",
    )
    parser.add_argument(
        "--support-cv-dir",
        type=Path,
        default=Path("outputs") / "real_cv" / "support2" / "support2_5fold_seed1234",
        help="Support2 CV の実験ディレクトリ",
    )
    parser.add_argument(
        "--cv-by-lambda",
        action="store_true",
        help="lambda ごとに fold 別 beta trajectory を重ねた図も生成する",
    )
    args = parser.parse_args()

    outputs = [
        plot_beta_trajectories(
            args.framingham_result,
            args.output_dir / "framingham_beta_trajectories.png",
        ),
        plot_beta_trajectories(
            args.support_result,
            args.output_dir / "support2_beta_trajectories.png",
        ),
    ]
    if args.cv_by_lambda:
        outputs.extend(
            plot_cv_beta_trajectories_by_lambda(
                "framingham",
                args.framingham_cv_dir,
                args.output_dir / "cv_by_lambda" / "framingham",
            )
        )
        outputs.extend(
            plot_cv_beta_trajectories_by_lambda(
                "support2",
                args.support_cv_dir,
                args.output_dir / "cv_by_lambda" / "support2",
            )
        )

    for path in outputs:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
