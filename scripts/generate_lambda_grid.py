#!/usr/bin/env python3
"""Lambda grid の生成スクリプト

目的:
    実験用のlambda値を対数スケールで生成し、lambda_grid.json に保存する。

使い方:
    # デフォルト（0 + 1～10の対数9点、合計10点）
    python scripts/generate_lambda_grid.py

    # カスタム範囲
    python scripts/generate_lambda_grid.py --min 0.001 --max 100 --n-points 20
"""

import argparse
import json
from pathlib import Path

import numpy as np


def generate_lambda_grid(
    min_value: float,
    max_value: float,
    n_points: int,
    scale: str = "log",
    include_zero: bool = True,
) -> list[float]:
    """Lambda値のグリッドを生成

    Args:
        min_value: 最小値
        max_value: 最大値
        n_points: 点数。include_zero=True の場合は 0 を含む合計点数
        scale: "log" または "linear"
        include_zero: 先頭に 0.0 を含めるか

    Returns:
        lambda値のリスト
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    positive_n_points = n_points - 1 if include_zero else n_points
    if positive_n_points <= 0:
        raise ValueError("n_points must be at least 2 when include_zero=True")

    if scale == "log":
        if min_value <= 0 or max_value <= 0:
            raise ValueError("min_value and max_value must be positive for log scale")
        values = np.logspace(
            np.log10(min_value),
            np.log10(max_value),
            positive_n_points,
        )
    elif scale == "linear":
        values = np.linspace(min_value, max_value, positive_n_points)
    else:
        raise ValueError(f"Unknown scale: {scale}")

    lambda_values = values.tolist()
    if include_zero:
        lambda_values = [0.0, *lambda_values]
    return lambda_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda gridの生成")
    parser.add_argument(
        "--min",
        type=float,
        default=1.0,
        help="最小値（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--max",
        type=float,
        default=10.0,
        help="最大値（デフォルト: 10.0）",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10,
        help="点数。--no-zero 未指定時は 0 を含む合計点数（デフォルト: 10）",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="スケール（デフォルト: log）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lambda_grid.json"),
        help="出力パス（デフォルト: lambda_grid.json）",
    )
    parser.add_argument(
        "--no-zero",
        action="store_true",
        help="先頭の 0.0 を含めない",
    )

    args = parser.parse_args()

    # Lambda値を生成
    lambda_values = generate_lambda_grid(
        args.min,
        args.max,
        args.n_points,
        args.scale,
        include_zero=not args.no_zero,
    )

    # JSON形式で保存
    data = {
        "description": (
            f"Lambda values for parallel experiments "
            f"({'0 + ' if not args.no_zero else ''}{args.scale} scale: "
            f"{args.min} to {args.max}, {args.n_points} points)"
        ),
        "lambda_values": lambda_values,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(lambda_values)} lambda values:")
    print(f"  Range: {min(lambda_values):.6f} to {max(lambda_values):.6f}")
    print(f"  Scale: {args.scale}")
    print(f"Saved to: {args.output}")

    # 値を表示
    print("\nValues:")
    for i, val in enumerate(lambda_values, 1):
        print(f"  {i:2d}. {val:.6f}")


if __name__ == "__main__":
    main()
