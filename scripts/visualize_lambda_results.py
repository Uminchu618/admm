#!/usr/bin/env python3
"""Lambda実験の可視化スクリプト

目的:
    aggregate_lambda_results.py で作成した summary.csv を読み込み、
    lambda値と目的関数の関係を可視化する。

使い方:
    python scripts/visualize_lambda_results.py --summary outputs/lambda_summary.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def plot_lambda_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Lambda値ごとの目的関数分布を箱ひげ図で表示

    Args:
        df: 集計結果のDataFrame
        output_dir: プロット保存先ディレクトリ
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # lambda値を文字列に変換してソート
    df_plot = df.copy()
    df_plot["lambda_str"] = df_plot["lambda_fuse"].apply(lambda x: f"{x:.4f}")
    df_plot = df_plot.sort_values("lambda_fuse")

    sns.boxplot(data=df_plot, x="lambda_str", y="objective_last", ax=ax)
    ax.set_xlabel("lambda_fuse")
    ax.set_ylabel("Objective (last)")
    ax.set_title("Objective distribution by lambda")
    ax.tick_params(axis="x", rotation=45)

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

    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Error: Summary file not found: {args.summary}")
        return

    df = pd.read_csv(args.summary)
    print(f"Loaded {len(df)} results from {args.summary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_lambda_vs_objective(df, args.output_dir)
    plot_lambda_distribution(df, args.output_dir)
    plot_convergence_vs_lambda(df, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
