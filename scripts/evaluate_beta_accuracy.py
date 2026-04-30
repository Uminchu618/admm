#!/usr/bin/env python3
"""推定された時間変動係数 beta(t) の真値誤差を集計・可視化する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_lambda_from_path(path: Path) -> float | None:
    """result.json の親ディレクトリ名 lambda_{value} から lambda を取り出す。"""
    for part in reversed(path.parts):
        if part.startswith("lambda_") and part != "lambda_experiments":
            try:
                return float(part.removeprefix("lambda_"))
            except ValueError:
                return None
    return None


def extract_data_name_from_path(path: Path) -> str:
    """outputs/lambda_experiments/{data_name}/lambda_x/result.json を想定して名前を返す。"""
    parent = path.parent
    if parent.name.startswith("lambda_"):
        return parent.parent.name
    return parent.name


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stepwise_true_beta(time_grid: np.ndarray, step_cfg: dict[str, Any]) -> np.ndarray:
    """生成設定の stepwise_beta から、推定区間ごとの真値 beta を作る。"""
    true_grid = np.asarray(step_cfg["time_grid"], dtype=float)
    levels_by_feature = [
        np.asarray(step_cfg[key], dtype=float)
        for key in sorted(step_cfg)
        if key.startswith("beta") and key.endswith("_levels")
    ]

    if not levels_by_feature:
        raise ValueError("stepwise_beta に beta*_levels がありません。")

    expected_len = len(true_grid) - 1
    for levels in levels_by_feature:
        if len(levels) != expected_len:
            raise ValueError("stepwise_beta の time_grid と levels の長さが一致しません。")

    mids = 0.5 * (time_grid[:-1] + time_grid[1:])
    idx = np.searchsorted(true_grid[1:], mids, side="right")
    idx = np.clip(idx, 0, expected_len - 1)
    return np.vstack([levels[idx] for levels in levels_by_feature]).T


def _smooth_true_beta_midpoint(time_grid: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """extended_aft_generator.config.json 形式の連続 beta を区間中点で近似する。"""
    td = cfg["time_dependence"]
    scenario = int(cfg["scenario"])
    t_mid = 0.5 * (time_grid[:-1] + time_grid[1:])

    beta1 = float(td["b11"]) * np.exp(-float(td["c1"]) * t_mid)
    beta2 = float(td["b21"]) * np.log1p(float(td["c2"]) * t_mid)
    if scenario == 1:
        beta3 = float(td["b31"]) * (t_mid - float(td["t0"])) ** 2
    else:
        beta3 = np.full_like(t_mid, float(td["b30"]), dtype=float)

    return np.vstack([beta1, beta2, beta3]).T


def compute_true_beta_by_interval(
    time_grid: np.ndarray, generator_config: Path
) -> np.ndarray:
    """生成設定から推定区間ごとの真の beta_{jk} を返す。"""
    cfg = load_json(generator_config)
    if "stepwise_beta" in cfg:
        return _stepwise_true_beta(time_grid, cfg["stepwise_beta"])
    if "time_dependence" in cfg and "scenario" in cfg:
        return _smooth_true_beta_midpoint(time_grid, cfg)
    raise ValueError(
        "真値 beta を計算できる設定ではありません。stepwise_beta または "
        "time_dependence/scenario が必要です。"
    )


def compute_integrated_errors(
    coef: np.ndarray, true_beta: np.ndarray, time_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """特徴量ごとの integral absolute error と integral squared error を返す。"""
    if coef.shape != true_beta.shape:
        raise ValueError(
            f"coef と true_beta の形状が一致しません: {coef.shape} != {true_beta.shape}"
        )

    widths = np.diff(time_grid).astype(float)
    if len(widths) != coef.shape[0]:
        raise ValueError("time_grid の区間数と coef の行数が一致しません。")
    if np.any(widths <= 0.0):
        raise ValueError("time_grid は狭義単調増加である必要があります。")

    diff = coef - true_beta
    iae = np.sum(widths[:, None] * np.abs(diff), axis=0)
    ise = np.sum(widths[:, None] * diff**2, axis=0)
    return iae, ise


def collect_beta_accuracy(results_dir: Path, generator_config: Path) -> pd.DataFrame:
    """result.json を走査し、beta(t) の積分誤差を run ごとに集計する。"""
    rows: list[dict[str, Any]] = []
    for result_path in sorted(results_dir.rglob("result.json")):
        payload = load_json(result_path)
        if "coef" not in payload or "time_grid" not in payload:
            continue

        lambda_fuse = extract_lambda_from_path(result_path)
        if lambda_fuse is None:
            continue

        coef = np.asarray(payload["coef"], dtype=float)
        time_grid = np.asarray(payload["time_grid"], dtype=float)
        if coef.ndim != 2 or time_grid.ndim != 1:
            continue

        true_beta = compute_true_beta_by_interval(time_grid, generator_config)
        if true_beta.shape[1] < coef.shape[1]:
            raise ValueError(
                f"真値 beta の特徴量数が不足しています: {result_path} "
                f"true={true_beta.shape[1]}, coef={coef.shape[1]}"
            )
        true_beta = true_beta[:, : coef.shape[1]]

        iae, ise = compute_integrated_errors(coef, true_beta, time_grid)
        row: dict[str, Any] = {
            "data_name": extract_data_name_from_path(result_path),
            "lambda_fuse": lambda_fuse,
            "result_path": str(result_path),
            "iae_mean": float(np.mean(iae)),
            "ise_mean": float(np.mean(ise)),
            "iae_sum": float(np.sum(iae)),
            "ise_sum": float(np.sum(ise)),
        }
        for j, value in enumerate(iae, start=1):
            row[f"iae_x{j}"] = float(value)
        for j, value in enumerate(ise, start=1):
            row[f"ise_x{j}"] = float(value)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"有効な result.json が見つかりません: {results_dir}")

    return pd.DataFrame(rows).sort_values(["lambda_fuse", "data_name"])


def _box_data_by_lambda(df: pd.DataFrame, column: str) -> tuple[np.ndarray, list[np.ndarray]]:
    lambdas = np.array(sorted(df["lambda_fuse"].dropna().unique()), dtype=float)
    values = [
        df.loc[df["lambda_fuse"] == lambda_val, column].to_numpy(dtype=float)
        for lambda_val in lambdas
    ]
    return lambdas, values


def plot_beta_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    """lambda ごとの平均 IAE/ISE 分布を箱ひげ図で保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    plot_specs = [
        ("iae_mean", "Mean integrated absolute error"),
        ("ise_mean", "Mean integrated squared error"),
    ]
    colors = ["#9ecae1", "#fdae6b"]
    edge_colors = ["#08306b", "#7f2704"]

    for ax, (column, ylabel), color, edge_color in zip(
        axes, plot_specs, colors, edge_colors, strict=True
    ):
        lambdas, values = _box_data_by_lambda(df, column)
        widths = np.maximum(lambdas * 0.12, np.min(lambdas) * 0.08)
        ax.boxplot(
            values,
            positions=lambdas,
            widths=widths,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.75},
            medianprops={"color": edge_color, "linewidth": 1.4},
            whiskerprops={"color": edge_color, "alpha": 0.8},
            capprops={"color": edge_color, "alpha": 0.8},
        )
        medians = [float(np.median(v)) for v in values]
        ax.plot(lambdas, medians, color=edge_color, marker="o", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("lambda_fuse (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(lambdas, labels=[f"{val:g}" for val in lambdas])
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Accuracy of estimated time-varying coefficients")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_by_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """lambda ごとに中央値と四分位をまとめる。"""
    grouped = df.groupby("lambda_fuse", as_index=False).agg(
        n_runs=("data_name", "count"),
        iae_mean_median=("iae_mean", "median"),
        iae_mean_q1=("iae_mean", lambda s: float(s.quantile(0.25))),
        iae_mean_q3=("iae_mean", lambda s: float(s.quantile(0.75))),
        ise_mean_median=("ise_mean", "median"),
        ise_mean_q1=("ise_mean", lambda s: float(s.quantile(0.25))),
        ise_mean_q3=("ise_mean", lambda s: float(s.quantile(0.75))),
    )
    return grouped.sort_values("lambda_fuse")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="beta(t) 推定精度の積分誤差を計算し可視化する。"
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
        help="真値 beta(t) を含む生成設定JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/beta_accuracy_summary.csv"),
        help="run単位の誤差集計CSV",
    )
    parser.add_argument(
        "--output-lambda-csv",
        type=Path,
        default=Path("outputs/beta_accuracy_by_lambda.csv"),
        help="lambda単位の要約CSV",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("outputs/lambda_plots/beta_accuracy_by_lambda.png"),
        help="可視化PNGの保存先",
    )
    args = parser.parse_args()

    df = collect_beta_accuracy(args.results_dir, args.generator_config)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    lambda_summary = summarize_by_lambda(df)
    args.output_lambda_csv.parent.mkdir(parents=True, exist_ok=True)
    lambda_summary.to_csv(args.output_lambda_csv, index=False, encoding="utf-8")

    plot_beta_accuracy(df, args.output_plot)

    print(f"Saved run-level summary to: {args.output_csv}")
    print(f"Saved lambda summary to: {args.output_lambda_csv}")
    print(f"Saved plot to: {args.output_plot}")
    print("\n=== Best lambda by median IAE ===")
    print(lambda_summary.sort_values("iae_mean_median").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
