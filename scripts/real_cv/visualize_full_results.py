#!/usr/bin/env python3
"""全データ real fit の BIC vs lambda を可視化する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_lambda_vs_bic(df: pd.DataFrame, output_dir: Path, dpi: int = 150) -> Path:
    """dataset ごとに lambda と BIC の関係を描く。"""

    required = {"dataset", "lambda_fuse", "bic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    valid = df.dropna(subset=["dataset", "lambda_fuse", "bic"]).copy()
    if valid.empty:
        raise ValueError("No valid BIC rows found for plotting.")

    valid["lambda_fuse"] = pd.to_numeric(valid["lambda_fuse"], errors="coerce")
    valid["bic"] = pd.to_numeric(valid["bic"], errors="coerce")
    valid = valid.dropna(subset=["lambda_fuse", "bic"])
    if valid.empty:
        raise ValueError("No numeric BIC rows found for plotting.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset, subset in valid.groupby("dataset"):
        subset = subset.sort_values("lambda_fuse")
        ax.plot(
            subset["lambda_fuse"],
            subset["bic"],
            marker="o",
            linewidth=2.0,
            label=str(dataset),
        )

        best = subset.loc[subset["bic"].idxmin()]
        ax.scatter(
            [best["lambda_fuse"]],
            [best["bic"]],
            s=90,
            marker="*",
            zorder=5,
        )
        ax.annotate(
            f"{dataset}: {float(best['lambda_fuse']):.4g}",
            xy=(float(best["lambda_fuse"]), float(best["bic"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    if (valid["lambda_fuse"] > 0).all():
        ax.set_xscale("log")
        ax.set_xlabel("lambda_fuse (log scale)")
    else:
        ax.set_xlabel("lambda_fuse")
    ax.set_ylabel("BIC")
    ax.set_title("Full-data BIC by lambda")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lambda_vs_bic.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize full real-data results")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/real_full/full_summary.csv"),
        help="CSV made by aggregate_full_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/real_full/plots"),
        help="Directory for PNG plots.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution.")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    output = plot_lambda_vs_bic(df, args.output_dir, dpi=args.dpi)
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
