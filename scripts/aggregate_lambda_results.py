#!/usr/bin/env python3
"""Lambda並列実験の結果を集計するスクリプト

目的:
    outputs/lambda_experiments/ 以下の全結果JSONを読み込み、
    lambda値と評価指標を横並びで比較可能な形（CSV/DataFrame）にまとめる。

使い方:
    python scripts/aggregate_lambda_results.py --output summary.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def collect_results(base_dir: Path) -> List[Dict[str, Any]]:
    """結果JSONをすべて収集する

    Args:
        base_dir: outputs/lambda_experiments/ のパス

    Returns:
        各runの情報を含む辞書のリスト
    """
    results = []

    # outputs/lambda_experiments/{data_name}/lambda_{value}/result.json を探索
    for result_path in base_dir.rglob("result.json"):
        try:
            with result_path.open("r", encoding="utf-8") as f:
                result = json.load(f)

            # ディレクトリ構造からdata_nameとlambda値を抽出
            lambda_dir = result_path.parent.name  # lambda_{value}
            data_dir = result_path.parent.parent.name

            # lambda値を抽出（lambda_0.01 → 0.01）
            lambda_str = lambda_dir.replace("lambda_", "")

            row = {
                "data_name": data_dir,
                "lambda_fuse": float(lambda_str),
                "n_samples": result.get("n_samples"),
                "n_features": result.get("n_features"),
                "objective_last": result.get("summary", {}).get("objective_last"),
                "primal_residual_last": result.get("summary", {}).get(
                    "primal_residual_last"
                ),
                "dual_residual_last": result.get("summary", {}).get(
                    "dual_residual_last"
                ),
                "result_path": str(result_path.relative_to(base_dir.parent.parent)),
            }

            # configから主要なハイパーパラメータも記録
            config = result.get("config", {})
            row["rho"] = config.get("rho")
            row["max_admm_iter"] = config.get("max_admm_iter")
            row["clip_eta"] = config.get("clip_eta")

            results.append(row)

        except Exception as e:
            print(f"Warning: Failed to load {result_path}: {e}")
            continue

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda並列実験の結果を集計")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/lambda_experiments"),
        help="実験結果のベースディレクトリ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lambda_summary.csv"),
        help="集計結果の出力パス（CSV）",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="objective_last",
        help="ソートに使う列名（デフォルト: objective_last）",
    )

    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"Error: Base directory not found: {args.base_dir}")
        return

    print(f"Collecting results from: {args.base_dir}")
    results = collect_results(args.base_dir)

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)

    # lambda値でグループ化して統計を表示
    print(f"\n=== Collected {len(df)} results ===")
    print(f"Data files: {df['data_name'].nunique()}")
    print(f"Lambda values: {sorted(df['lambda_fuse'].unique())}")

    # ソート
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=True)

    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved summary to: {args.output}")

    # 上位5件を表示
    print("\n=== Top 5 results (by objective) ===")
    print(
        df[["data_name", "lambda_fuse", "objective_last", "primal_residual_last"]].head(
            5
        )
    )


if __name__ == "__main__":
    main()
