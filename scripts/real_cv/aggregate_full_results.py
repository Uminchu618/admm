#!/usr/bin/env python3
"""全データ real fit の lambda 別 result.json を集計する。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def count_nonzero_z(z_last: Any, tol: float) -> int | None:
    """z_last のうち |z| > tol の要素数を数える。"""

    if z_last is None:
        return None

    try:
        total = 0
        for row in z_last:
            for value in row:
                if abs(float(value)) > tol:
                    total += 1
        return total
    except (TypeError, ValueError):
        return None


def _parse_path_metadata(result_path: Path) -> tuple[str | None, float | None]:
    """outputs/real_full/{dataset}/.../lambda_x/result.json から情報を読む。"""

    dataset = None
    lambda_value = None
    parts = list(result_path.parts)
    if "real_full" in parts:
        idx = parts.index("real_full")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]

    for part in parts:
        if part.startswith("lambda_"):
            try:
                lambda_value = float(part.replace("lambda_", "", 1))
            except ValueError:
                pass
    return dataset, lambda_value


def collect_results(base_dir: Path, z_tol: float = 1e-8) -> pd.DataFrame:
    """base_dir 以下の result.json を DataFrame にする。"""

    rows: list[dict[str, Any]] = []
    for result_path in sorted(base_dir.rglob("result.json")):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except Exception as exc:
            print(f"Warning: failed to load {result_path}: {exc}", file=sys.stderr)
            continue

        path_dataset, path_lambda = _parse_path_metadata(result_path)
        summary = result.get("summary", {})
        config = result.get("config", {})
        n_samples = result.get("n_samples")
        lambda_fuse = config.get("lambda_fuse", path_lambda)
        lambda_fuse_effective = summary.get(
            "lambda_fuse_effective",
            result.get("history", {}).get("lambda_fuse_effective"),
        )
        if (
            lambda_fuse_effective is None
            and lambda_fuse is not None
            and n_samples is not None
        ):
            lambda_fuse_effective = float(n_samples) * float(lambda_fuse)
        n_params = count_nonzero_z(result.get("z_last"), z_tol)
        neg_loglik_last = summary.get("neg_loglik_last")

        if (
            neg_loglik_last is not None
            and n_params is not None
            and n_samples is not None
        ):
            bic = 2.0 * float(neg_loglik_last) + float(n_params) * math.log(
                float(n_samples)
            )
        else:
            bic = None

        rows.append(
            {
                "dataset": result.get("dataset", path_dataset),
                "lambda_fuse": lambda_fuse,
                "lambda_fuse_effective": lambda_fuse_effective,
                "n_samples": n_samples,
                "n_features": result.get("n_features"),
                "objective_last": summary.get("objective_last"),
                "neg_loglik_last": neg_loglik_last,
                "primal_residual_last": summary.get("primal_residual_last"),
                "dual_residual_last": summary.get("dual_residual_last"),
                "stopping_reason": summary.get("stopping_reason"),
                "n_admm_iter": summary.get("n_admm_iter"),
                "c_td": summary.get("c_td"),
                "c_td_train": summary.get("c_td_train"),
                "n_params": n_params,
                "bic": bic,
                "rho": config.get("rho"),
                "max_admm_iter": config.get("max_admm_iter"),
                "clip_eta": config.get("clip_eta"),
                "result_path": str(result_path),
            }
        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for column in ["lambda_fuse", "bic", "c_td", "c_td_train"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["dataset", "lambda_fuse"], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate full real-data results")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/real_full"),
        help="Directory containing dataset/experiment/lambda_*/result.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/real_full/full_summary.csv"),
        help="Path to write summary CSV.",
    )
    parser.add_argument(
        "--z-tol",
        type=float,
        default=1e-8,
        help="|z|>tol を非ゼロパラメータとして数える閾値。",
    )
    args = parser.parse_args()

    df = collect_results(args.base_dir, z_tol=args.z_tol)
    if df.empty:
        print(f"No result.json files found under {args.base_dir}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved summary to: {args.output}")

    best = df.dropna(subset=["bic"]).sort_values(["dataset", "bic"])
    if not best.empty:
        print("\n=== Best lambda by BIC ===")
        columns = ["dataset", "lambda_fuse", "bic", "n_params", "c_td"]
        print(best.groupby("dataset", as_index=False).head(1)[columns])


if __name__ == "__main__":
    main()
