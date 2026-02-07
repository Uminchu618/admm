#!/usr/bin/env python3
"""Lambda実験の可視化スクリプト

目的:
    aggregate_lambda_results.py で作成した summary.csv を読み込み、
    lambda値と目的関数の関係を可視化する。

使い方:
    python scripts/visualize_lambda_results.py --summary outputs/lambda_summary.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_lambda_vs_objective(df: pd.DataFrame, output_dir: Path) -> None:
    """Lambda値と目的関数の関係をプロット

    Args:
        df: 集計結果のDataFrame
        output_dir: プロット保存先ディレクトリ
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 各データファイルごとに線を引く（代表例のみ）
    data_names = df["data_name"].unique()
    if len(data_names) > 10:
        # 多すぎる場合は最初の10個のみ
        data_names = data_names[:10]

    for data_name in data_names:
        subset = df[df["data_name"] == data_name].sort_values("lambda_fuse")
        ax.plot(
            subset["lambda_fuse"],
            subset["objective_last"],
            marker="o",
            label=data_name,
            alpha=0.7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("lambda_fuse (log scale)")
    ax.set_ylabel("Objective (last)")
    ax.set_title("Objective vs Lambda")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = output_dir / "lambda_vs_objective.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close(fig)


def load_beta_results(
    results_dir: Path,
) -> Tuple[Dict[float, List[np.ndarray]], np.ndarray]:
    """結果JSONからbetaを読み込み、lambdaごとに集約する。

    Returns:
        (beta_by_lambda, time_grid)
    """
    beta_by_lambda: Dict[float, List[np.ndarray]] = {}
    time_grid_ref: np.ndarray | None = None

    for result_path in results_dir.rglob("result.json"):
        lambda_dir = result_path.parent.name
        if not lambda_dir.startswith("lambda_"):
            continue
        try:
            lambda_val = float(lambda_dir.replace("lambda_", ""))
        except ValueError:
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        coef = np.asarray(payload.get("coef"), dtype=float)
        time_grid = np.asarray(payload.get("time_grid"), dtype=float)
        if coef.ndim != 2 or time_grid.ndim != 1:
            continue

        if time_grid_ref is None:
            time_grid_ref = time_grid
        elif time_grid_ref.shape != time_grid.shape or not np.allclose(
            time_grid_ref, time_grid
        ):
            continue

        beta_by_lambda.setdefault(lambda_val, []).append(coef)

    if time_grid_ref is None:
        raise FileNotFoundError("No valid result.json found in results_dir.")

    return beta_by_lambda, time_grid_ref


def compute_true_beta(time_points: np.ndarray, config_path: Path) -> np.ndarray:
    """生成設定から真のbeta(t)を計算する。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "stepwise_beta" in cfg:
        step_cfg = cfg["stepwise_beta"]
        time_grid = np.asarray(step_cfg["time_grid"], dtype=float)

        def piecewise_beta(levels: list[float]) -> np.ndarray:
            idx = np.searchsorted(time_grid[1:], time_points, side="right")
            idx = np.clip(idx, 0, len(levels) - 1)
            return np.asarray(levels, dtype=float)[idx]

        beta1 = piecewise_beta(step_cfg["beta1_levels"])
        beta2 = piecewise_beta(step_cfg["beta2_levels"])
        beta3 = piecewise_beta(step_cfg["beta3_levels"])
        return np.vstack([beta1, beta2, beta3]).T

    td = cfg["time_dependence"]
    scenario = cfg["scenario"]

    beta1 = td["b11"] * np.exp(-td["c1"] * time_points)
    beta2 = td["b21"] * np.log1p(td["c2"] * time_points)
    if scenario == 1:
        beta3 = td["b31"] * (time_points - td["t0"]) ** 2
    else:
        beta3 = np.full_like(time_points, td["b30"], dtype=float)

    return np.vstack([beta1, beta2, beta3]).T


def plot_lambda_distribution(
    output_dir: Path,
    results_dir: Path,
    generator_config: Path,
) -> None:
    """Lambda値ごとのβ分布をlambda×共変量グリッドで表示

    Args:
        df: 集計結果のDataFrame
        output_dir: プロット保存先ディレクトリ
        results_dir: result.json を含むディレクトリ
        generator_config: 真値β(t)の計算用設定
    """
    beta_by_lambda, time_grid = load_beta_results(results_dir)
    lambdas = sorted(beta_by_lambda.keys())

    time_mid = (time_grid[:-1] + time_grid[1:]) / 2.0
    true_beta = compute_true_beta(time_mid, generator_config)

    n_rows = len(lambdas)
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.5, n_rows * 2.4),
        sharex=True,
    )

    if n_rows == 1:
        axes = np.array([axes])

    time_spacing = time_mid[1] - time_mid[0] if len(time_mid) > 1 else 0.2
    box_width = time_spacing * 0.6

    for i, lambda_val in enumerate(lambdas):
        coef_list = beta_by_lambda[lambda_val]
        coef_stack = np.stack(coef_list, axis=0)
        if coef_stack.shape[1] != len(time_mid) or coef_stack.shape[2] < n_cols:
            continue

        for j in range(n_cols):
            ax = axes[i, j]
            box_data = [coef_stack[:, k, j] for k in range(len(time_mid))]
            ax.boxplot(
                box_data,
                positions=time_mid,
                widths=box_width,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#9ecae1", "alpha": 0.7},
                medianprops={"color": "#08306b", "linewidth": 1.2},
            )
            ax.plot(
                time_mid,
                true_beta[:, j],
                color="#e31a1c",
                linewidth=1.6,
                label="true",
            )
            if j == 0:
                ax.set_ylim(-1, 2)
            elif j == 1:
                ax.set_ylim(-1, 2)
            elif j == 2:
                ax.set_ylim(-1, 2)
            if i == 0:
                ax.set_title(f"x{j+1}")
            if j == 0:
                ax.set_ylabel(f"lambda={lambda_val:.4g}\nβ")
            if i == n_rows - 1:
                ax.set_xlabel("time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("β(t) distribution by lambda and covariate", y=1.02)
    fig.tight_layout()
    output_path = output_dir / "lambda_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close(fig)


def plot_convergence_vs_lambda(df: pd.DataFrame, output_dir: Path) -> None:
    """Lambda値と収束状況（残差）の関係をプロット

    Args:
        df: 集計結果のDataFrame
        output_dir: プロット保存先ディレクトリ
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Primal residual
    for data_name in df["data_name"].unique()[:10]:
        subset = df[df["data_name"] == data_name].sort_values("lambda_fuse")
        axes[0].plot(
            subset["lambda_fuse"],
            subset["primal_residual_last"],
            marker="o",
            alpha=0.5,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("lambda_fuse")
    axes[0].set_ylabel("Primal residual (last)")
    axes[0].set_title("Convergence: Primal residual")
    axes[0].grid(True, alpha=0.3)

    # Dual residual
    for data_name in df["data_name"].unique()[:10]:
        subset = df[df["data_name"] == data_name].sort_values("lambda_fuse")
        axes[1].plot(
            subset["lambda_fuse"],
            subset["dual_residual_last"],
            marker="o",
            alpha=0.5,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("lambda_fuse")
    axes[1].set_ylabel("Dual residual (last)")
    axes[1].set_title("Convergence: Dual residual")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = output_dir / "lambda_vs_convergence.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda実験結果の可視化")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/lambda_summary.csv"),
        help="集計結果のCSVパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lambda_plots"),
        help="プロット保存先ディレクトリ",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/lambda_experiments"),
        help="result.json を含む実験結果ディレクトリ",
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=Path("generation/extended_aft_step_generator.config.json"),
        help="真値β(t)計算用の生成設定",
    )

    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Error: Summary file not found: {args.summary}")
        return

    df = pd.read_csv(args.summary)
    print(f"Loaded {len(df)} results from {args.summary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_lambda_vs_objective(df, args.output_dir)
    plot_lambda_distribution(args.output_dir, args.results_dir, args.generator_config)
    plot_convergence_vs_lambda(df, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
