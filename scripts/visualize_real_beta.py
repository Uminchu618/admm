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


def plot_beta_trajectories(result_path: Path, output_path: Path) -> Path:
    result = load_result(result_path)

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

    for path in outputs:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
