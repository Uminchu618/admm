#!/usr/bin/env python3
"""実データ CV 結果を lambda/fold 単位で可視化する。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.aggregate_results import collect_results, summarize_by_lambda


def _as_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """存在する列だけ数値化した DataFrame を返す。"""

    copied = df.copy()
    for column in columns:
        if column in copied.columns:
            copied[column] = pd.to_numeric(copied[column], errors="coerce")
    return copied


def _prepare_fold_df(fold_df: pd.DataFrame) -> pd.DataFrame:
    """可視化で使う列を数値化し、lambda/fold 順に並べる。"""

    numeric_columns = [
        "lambda_fuse",
        "fold",
        "c_td_train",
        "c_td_test",
        "n_admm_iter",
        "primal_residual_last",
        "dual_residual_last",
    ]
    prepared = _as_numeric(fold_df, numeric_columns)
    return prepared.sort_values(["lambda_fuse", "fold"], na_position="last")


def _lambda_summary_for_plots(fold_df: pd.DataFrame) -> pd.DataFrame:
    """平均・標準偏差・標準誤差を train/test c_td と収束指標で集計する。"""

    prepared = fold_df.copy()
    for column in [
        "c_td_test",
        "c_td_train",
        "n_admm_iter",
        "primal_residual_last",
        "dual_residual_last",
    ]:
        if column not in prepared.columns:
            prepared[column] = np.nan

    grouped = (
        prepared.groupby("lambda_fuse", dropna=False)
        .agg(
            n_folds=("fold", "count"),
            c_td_test_mean=("c_td_test", "mean"),
            c_td_test_std=("c_td_test", "std"),
            c_td_train_mean=("c_td_train", "mean"),
            c_td_train_std=("c_td_train", "std"),
            n_admm_iter_mean=("n_admm_iter", "mean"),
            n_admm_iter_std=("n_admm_iter", "std"),
            primal_residual_last_mean=("primal_residual_last", "mean"),
            primal_residual_last_std=("primal_residual_last", "std"),
            dual_residual_last_mean=("dual_residual_last", "mean"),
            dual_residual_last_std=("dual_residual_last", "std"),
        )
        .reset_index()
        .sort_values("lambda_fuse")
    )

    for metric in [
        "c_td_test",
        "c_td_train",
        "n_admm_iter",
        "primal_residual_last",
        "dual_residual_last",
    ]:
        grouped[f"{metric}_se"] = grouped.apply(
            lambda row: (
                float(row[f"{metric}_std"]) / math.sqrt(float(row["n_folds"]))
                if row["n_folds"] and pd.notna(row[f"{metric}_std"])
                else np.nan
            ),
            axis=1,
        )
    return grouped


def _metric_error(summary_df: pd.DataFrame, metric: str, error: str) -> np.ndarray | None:
    if error == "none":
        return None
    column = f"{metric}_{error}"
    if column not in summary_df.columns:
        return None
    return summary_df[column].to_numpy(dtype=float)


def _cox_test_summary(
    cox_df: pd.DataFrame | None,
    *,
    error: str,
) -> tuple[float, float | None] | None:
    """Cox CV summary/fold CSV から test c_td の平均と誤差幅を読む。"""

    if cox_df is None or cox_df.empty:
        return None

    if "c_td_test_cox_mean" in cox_df.columns:
        mean_values = pd.to_numeric(
            cox_df["c_td_test_cox_mean"], errors="coerce"
        ).dropna()
        if mean_values.empty:
            return None
        mean = float(mean_values.iloc[0])
        err = None
        error_column = f"c_td_test_cox_{error}"
        if error != "none" and error_column in cox_df.columns:
            err_values = pd.to_numeric(cox_df[error_column], errors="coerce").dropna()
            if not err_values.empty:
                err = float(err_values.iloc[0])
        return mean, err

    if "c_td_test_cox" not in cox_df.columns:
        return None
    values = pd.to_numeric(cox_df["c_td_test_cox"], errors="coerce").dropna()
    if values.empty:
        return None
    mean = float(values.mean())
    if values.shape[0] <= 1 or error == "none":
        return mean, None
    std = float(values.std(ddof=1))
    err = std if error == "std" else std / math.sqrt(values.shape[0])
    return mean, err


def _add_cox_test_reference(
    ax: plt.Axes,
    cox_df: pd.DataFrame | None,
    *,
    error: str,
) -> None:
    """現在の軸に Cox test c_td の基準線を重ねる。"""

    summary = _cox_test_summary(cox_df, error=error)
    if summary is None:
        return

    mean, se = summary
    label = f"Cox test c_td = {mean:.3f}"
    ax.axhline(
        mean,
        color="#111827",
        linestyle=":",
        linewidth=2.0,
        label=label,
        zorder=1,
    )
    if error != "none" and se is not None and np.isfinite(se):
        ax.axhspan(
            mean - se,
            mean + se,
            color="#111827",
            alpha=0.08,
            label=f"Cox +/- {error}",
            zorder=0,
        )


def _aft_test_summaries(
    aft_df: pd.DataFrame | None,
    *,
    error: str,
) -> list[tuple[str, float, float | None]]:
    """AFT CV summary/fold CSV から model 別 test c_td 平均と誤差幅を読む。"""

    if aft_df is None or aft_df.empty:
        return []

    model_column = "aft_model" if "aft_model" in aft_df.columns else None
    if "c_td_test_aft_mean" in aft_df.columns:
        rows = []
        for idx, row in aft_df.iterrows():
            mean = pd.to_numeric(
                pd.Series([row.get("c_td_test_aft_mean")]), errors="coerce"
            ).iloc[0]
            if pd.isna(mean):
                continue
            err = None
            error_column = f"c_td_test_aft_{error}"
            if error != "none" and error_column in aft_df.columns:
                err_value = pd.to_numeric(
                    pd.Series([row.get(error_column)]), errors="coerce"
                ).iloc[0]
                if pd.notna(err_value):
                    err = float(err_value)
            label = str(row.get(model_column, f"AFT {idx + 1}")) if model_column else "AFT"
            rows.append((label, float(mean), err))
        return rows

    if "c_td_test_aft" not in aft_df.columns:
        return []

    grouped = (
        aft_df.groupby(model_column, dropna=False)
        if model_column is not None
        else [(None, aft_df)]
    )
    rows = []
    for model, subset in grouped:
        values = pd.to_numeric(subset["c_td_test_aft"], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        err = None
        if values.shape[0] > 1 and error != "none":
            std = float(values.std(ddof=1))
            err = std if error == "std" else std / math.sqrt(values.shape[0])
        label = "AFT" if model is None or pd.isna(model) else str(model)
        rows.append((label, mean, err))
    return rows


def _add_aft_test_references(
    ax: plt.Axes,
    aft_df: pd.DataFrame | None,
    *,
    error: str,
) -> None:
    """現在の軸に AFT test c_td の基準線を重ねる。"""

    summaries = _aft_test_summaries(aft_df, error=error)
    if not summaries:
        return

    colors = ["#8c564b", "#9467bd", "#e377c2", "#17becf", "#bcbd22"]
    for idx, (model, mean, err) in enumerate(summaries):
        color = colors[idx % len(colors)]
        ax.axhline(
            mean,
            color=color,
            linestyle="-.",
            linewidth=1.8,
            label=f"{model} AFT test c_td = {mean:.3f}",
            zorder=1,
        )
        if error != "none" and err is not None and np.isfinite(err):
            ax.axhspan(
                mean - err,
                mean + err,
                color=color,
                alpha=0.06,
                label=f"{model} AFT +/- {error}",
                zorder=0,
            )


def _set_lambda_axis(ax: plt.Axes, lambdas: pd.Series | np.ndarray) -> None:
    values = np.asarray(lambdas, dtype=float)
    if np.all(values > 0):
        ax.set_xscale("log")
        ax.set_xlabel("lambda_fuse (log scale)")
    else:
        ax.set_xlabel("lambda_fuse")
    ax.grid(True, alpha=0.3)


def _positive_for_log(values: pd.Series | np.ndarray) -> tuple[np.ndarray, float]:
    """log 表示用に 0 以下を小さな正値へ寄せる。"""

    array = np.asarray(values, dtype=float)
    positive = array[np.isfinite(array) & (array > 0)]
    floor = float(np.min(positive) * 0.1) if positive.size else 1e-12
    return np.where(array > 0, array, floor), floor


def plot_lambda_vs_c_td(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    cox_df: pd.DataFrame | None = None,
    aft_df: pd.DataFrame | None = None,
    error: str = "se",
    dpi: int = 150,
) -> Path:
    """lambda ごとの test c_td を fold 点・平均線・エラーバーで描く。"""

    valid_fold = fold_df.dropna(subset=["lambda_fuse", "c_td_test"])
    valid_summary = summary_df.dropna(subset=["lambda_fuse", "c_td_test_mean"])
    if valid_fold.empty or valid_summary.empty:
        raise ValueError("No valid c_td_test values found for plotting.")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    lambdas = valid_summary["lambda_fuse"].to_numpy(dtype=float)

    for fold, subset in valid_fold.groupby("fold", dropna=False):
        subset = subset.sort_values("lambda_fuse")
        ax.scatter(
            subset["lambda_fuse"],
            subset["c_td_test"],
            s=34,
            alpha=0.58,
            label=f"fold {int(fold):02d}" if pd.notna(fold) else "fold NA",
        )

    ax.errorbar(
        lambdas,
        valid_summary["c_td_test_mean"].to_numpy(dtype=float),
        yerr=_metric_error(valid_summary, "c_td_test", error),
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        capsize=4,
        label=f"mean +/- {error}" if error != "none" else "mean",
    )
    _add_cox_test_reference(ax, cox_df, error=error)
    _add_aft_test_references(ax, aft_df, error=error)

    best_idx = valid_summary["c_td_test_mean"].idxmax()
    best_lambda = float(valid_summary.loc[best_idx, "lambda_fuse"])
    best_score = float(valid_summary.loc[best_idx, "c_td_test_mean"])
    ax.axvline(best_lambda, color="#d62728", linestyle="--", linewidth=1.3)
    ax.annotate(
        f"best lambda={best_lambda:.4g}\nmean c_td={best_score:.3f}",
        xy=(best_lambda, best_score),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#7f1d1d",
    )

    _set_lambda_axis(ax, lambdas)
    ax.set_ylabel("test c_td")
    ax.set_title("CV test c_td by lambda")
    ax.legend(loc="best", fontsize="small", ncols=2)
    fig.tight_layout()

    output_path = output_dir / "cv_lambda_vs_c_td.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_train_test_c_td(
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    cox_df: pd.DataFrame | None = None,
    aft_df: pd.DataFrame | None = None,
    error: str = "se",
    dpi: int = 150,
) -> Path:
    """train/test の mean c_td を lambda に対して比較する。"""

    valid = summary_df.dropna(subset=["lambda_fuse", "c_td_test_mean"])
    if valid.empty:
        raise ValueError("No valid c_td summary values found for plotting.")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    lambdas = valid["lambda_fuse"].to_numpy(dtype=float)

    if "c_td_train_mean" in valid.columns and valid["c_td_train_mean"].notna().any():
        ax.errorbar(
            lambdas,
            valid["c_td_train_mean"].to_numpy(dtype=float),
            yerr=_metric_error(valid, "c_td_train", error),
            color="#2ca02c",
            marker="s",
            linewidth=1.8,
            capsize=4,
            label=f"train mean +/- {error}" if error != "none" else "train mean",
        )

    ax.errorbar(
        lambdas,
        valid["c_td_test_mean"].to_numpy(dtype=float),
        yerr=_metric_error(valid, "c_td_test", error),
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        capsize=4,
        label=f"test mean +/- {error}" if error != "none" else "test mean",
    )
    _add_cox_test_reference(ax, cox_df, error="none")
    _add_aft_test_references(ax, aft_df, error="none")

    _set_lambda_axis(ax, lambdas)
    ax.set_ylabel("c_td")
    ax.set_title("Train/test c_td by lambda")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = output_dir / "cv_train_test_c_td.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fold_spaghetti(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    cox_df: pd.DataFrame | None = None,
    aft_df: pd.DataFrame | None = None,
    dpi: int = 150,
) -> Path:
    """fold ごとの test c_td 軌跡と平均線を描く。"""

    valid_fold = fold_df.dropna(subset=["lambda_fuse", "fold", "c_td_test"])
    valid_summary = summary_df.dropna(subset=["lambda_fuse", "c_td_test_mean"])
    if valid_fold.empty or valid_summary.empty:
        raise ValueError("No valid fold-level c_td_test values found for plotting.")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for fold, subset in valid_fold.groupby("fold"):
        subset = subset.sort_values("lambda_fuse")
        ax.plot(
            subset["lambda_fuse"],
            subset["c_td_test"],
            marker="o",
            linewidth=1.2,
            alpha=0.58,
            label=f"fold {int(fold):02d}",
        )

    valid_summary = valid_summary.sort_values("lambda_fuse")
    lambdas = valid_summary["lambda_fuse"].to_numpy(dtype=float)
    ax.plot(
        lambdas,
        valid_summary["c_td_test_mean"].to_numpy(dtype=float),
        color="#111827",
        marker="o",
        linewidth=2.6,
        label="mean",
    )
    _add_cox_test_reference(ax, cox_df, error="none")
    _add_aft_test_references(ax, aft_df, error="none")

    _set_lambda_axis(ax, lambdas)
    ax.set_ylabel("test c_td")
    ax.set_title("Fold-wise test c_td trajectories")
    ax.legend(loc="best", fontsize="small", ncols=2)
    fig.tight_layout()

    output_path = output_dir / "cv_fold_spaghetti.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_convergence_diagnostics(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    error: str = "se",
    dpi: int = 150,
) -> Path:
    """ADMM iteration・残差・停止理由を 2x2 パネルで描く。"""

    valid = summary_df.dropna(subset=["lambda_fuse"]).sort_values("lambda_fuse")
    if valid.empty:
        raise ValueError("No valid lambda values found for convergence plotting.")

    lambdas = valid["lambda_fuse"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    ax_iter, ax_primal, ax_dual, ax_stop = axes.ravel()

    if "n_admm_iter_mean" in valid.columns and valid["n_admm_iter_mean"].notna().any():
        ax_iter.errorbar(
            lambdas,
            valid["n_admm_iter_mean"].to_numpy(dtype=float),
            yerr=_metric_error(valid, "n_admm_iter", error),
            color="#9467bd",
            marker="o",
            linewidth=1.9,
            capsize=4,
        )
    _set_lambda_axis(ax_iter, lambdas)
    ax_iter.set_ylabel("ADMM iterations")
    ax_iter.set_title("ADMM iterations")

    if (
        "primal_residual_last_mean" in valid.columns
        and valid["primal_residual_last_mean"].notna().any()
    ):
        y, floor = _positive_for_log(valid["primal_residual_last_mean"])
        yerr = _metric_error(valid, "primal_residual_last", error)
        ax_primal.errorbar(
            lambdas,
            y,
            yerr=yerr,
            color="#ff7f0e",
            marker="o",
            linewidth=1.9,
            capsize=4,
        )
        ax_primal.set_yscale("log")
        if floor > 0:
            ax_primal.set_ylim(bottom=floor * 0.5)
    _set_lambda_axis(ax_primal, lambdas)
    ax_primal.set_ylabel("primal residual last")
    ax_primal.set_title("Primal residual")

    if (
        "dual_residual_last_mean" in valid.columns
        and valid["dual_residual_last_mean"].notna().any()
    ):
        y, floor = _positive_for_log(valid["dual_residual_last_mean"])
        yerr = _metric_error(valid, "dual_residual_last", error)
        ax_dual.errorbar(
            lambdas,
            y,
            yerr=yerr,
            color="#8c564b",
            marker="o",
            linewidth=1.9,
            capsize=4,
        )
        ax_dual.set_yscale("log")
        if floor > 0:
            ax_dual.set_ylim(bottom=floor * 0.5)
    _set_lambda_axis(ax_dual, lambdas)
    ax_dual.set_ylabel("dual residual last")
    ax_dual.set_title("Dual residual")

    if "stopping_reason" in fold_df.columns and fold_df["stopping_reason"].notna().any():
        stop_counts = (
            fold_df.dropna(subset=["lambda_fuse", "stopping_reason"])
            .assign(stopping_reason=lambda df: df["stopping_reason"].astype(str))
            .groupby(["lambda_fuse", "stopping_reason"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        bottoms = np.zeros(len(stop_counts), dtype=float)
        x = stop_counts.index.to_numpy(dtype=float)
        for reason in stop_counts.columns:
            values = stop_counts[reason].to_numpy(dtype=float)
            ax_stop.bar(x, values, bottom=bottoms, label=reason, width=x * 0.08)
            bottoms += values
        _set_lambda_axis(ax_stop, x)
        ax_stop.set_ylabel("fold count")
        ax_stop.legend(loc="best", fontsize="small")
    else:
        ax_stop.text(0.5, 0.5, "stopping_reason unavailable", ha="center", va="center")
        ax_stop.set_axis_off()
    ax_stop.set_title("Stopping reasons")

    fig.suptitle("CV convergence diagnostics", y=1.01)
    fig.tight_layout()

    output_path = output_dir / "cv_convergence_diagnostics.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_model_comparison_table(
    summary_df: pd.DataFrame,
    *,
    cox_df: pd.DataFrame | None = None,
    aft_df: pd.DataFrame | None = None,
    error: str = "se",
) -> pd.DataFrame:
    """ADMM best、Cox、AFT の test c_td 比較表を作る。"""

    rows: list[dict[str, float | str | None]] = []

    admm = summary_df.dropna(subset=["c_td_test_mean"])
    if not admm.empty:
        best = admm.sort_values("c_td_test_mean", ascending=False).iloc[0]
        err_col = f"c_td_test_{error}"
        rows.append(
            {
                "model": "ADMM best lambda",
                "model_family": "admm",
                "lambda_fuse": float(best["lambda_fuse"])
                if pd.notna(best.get("lambda_fuse"))
                else None,
                "c_td_test_mean": float(best["c_td_test_mean"]),
                "c_td_test_error": float(best[err_col])
                if error != "none" and err_col in best and pd.notna(best[err_col])
                else None,
            }
        )

    cox_summary = _cox_test_summary(cox_df, error=error)
    if cox_summary is not None:
        mean, err = cox_summary
        rows.append(
            {
                "model": "CoxPH",
                "model_family": "cox",
                "lambda_fuse": None,
                "c_td_test_mean": mean,
                "c_td_test_error": err,
            }
        )

    for model, mean, err in _aft_test_summaries(aft_df, error=error):
        rows.append(
            {
                "model": f"{model} AFT",
                "model_family": "aft",
                "lambda_fuse": None,
                "c_td_test_mean": mean,
                "c_td_test_error": err,
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("c_td_test_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    *,
    error: str = "se",
    dpi: int = 150,
) -> Path:
    """model 別の test c_td 平均を横棒グラフで描く。"""

    valid = comparison_df.dropna(subset=["model", "c_td_test_mean"])
    if valid.empty:
        raise ValueError("No valid model comparison rows found for plotting.")

    valid = valid.sort_values("c_td_test_mean", ascending=True)
    colors = valid["model_family"].map(
        {"admm": "#1f77b4", "cox": "#111827", "aft": "#9467bd"}
    ).fillna("#6b7280")
    xerr = None
    if error != "none" and "c_td_test_error" in valid.columns:
        xerr = pd.to_numeric(valid["c_td_test_error"], errors="coerce").to_numpy(
            dtype=float
        )
    centers = valid["c_td_test_mean"].to_numpy(dtype=float)
    if xerr is not None:
        finite_error = np.where(np.isfinite(xerr), xerr, 0.0)
    else:
        finite_error = np.zeros_like(centers)
    x_min = float(np.min(centers - finite_error))
    x_max = float(np.max(centers + finite_error))
    span = max(x_max - x_min, 0.02)
    padding = max(span * 0.25, 0.005)

    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.55 * len(valid) + 1.8)))
    ax.barh(
        valid["model"],
        valid["c_td_test_mean"],
        xerr=xerr,
        color=colors,
        alpha=0.88,
        capsize=4,
    )
    ax.set_xlabel("test c_td")
    ax.set_title("Model comparison by CV test c_td")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(left=max(0.0, x_min - padding), right=min(1.0, x_max + padding))
    fig.tight_layout()

    output_path = output_dir / "cv_model_comparison.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_or_collect_cv_results(
    base_dir: Path,
    fold_results_path: Path | None = None,
    summary_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """既存 CSV を読むか、result.json から集計する。"""

    if fold_results_path is not None and fold_results_path.exists():
        fold_df = pd.read_csv(fold_results_path)
    else:
        fold_df = collect_results(base_dir)

    if fold_df.empty:
        raise FileNotFoundError(f"No result.json files found under {base_dir}")

    fold_df = _prepare_fold_df(fold_df)

    if summary_path is not None and summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = _as_numeric(
            summary_df,
            [
                "lambda_fuse",
                "n_folds",
                "c_td_test_mean",
                "c_td_test_std",
                "c_td_test_se",
                "c_td_train_mean",
                "c_td_train_std",
                "c_td_train_se",
                "primal_residual_last_mean",
                "dual_residual_last_mean",
            ],
        ).sort_values("lambda_fuse")
        extended_summary = _lambda_summary_for_plots(fold_df)
        for column in extended_summary.columns:
            if column not in summary_df.columns:
                summary_df = summary_df.merge(
                    extended_summary[["lambda_fuse", column]],
                    on="lambda_fuse",
                    how="left",
                )
    else:
        summary_df = _lambda_summary_for_plots(fold_df)

    return fold_df, summary_df


def write_cv_tables(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fold_output: Path,
    summary_output: Path,
) -> None:
    """可視化に使った集計 CSV を保存する。"""

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(fold_output, index=False, encoding="utf-8")
    # 既存 aggregate_results.py と同じ列を先頭に残しつつ、可視化用列も保存する。
    base_summary = summarize_by_lambda(fold_df)
    extra_columns = [
        column
        for column in summary_df.columns
        if column not in base_summary.columns and column != "lambda_fuse"
    ]
    if extra_columns:
        base_summary = base_summary.merge(
            summary_df[["lambda_fuse", *extra_columns]], on="lambda_fuse", how="left"
        )
    base_summary.to_csv(summary_output, index=False, encoding="utf-8")


def create_all_plots(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    cox_df: pd.DataFrame | None = None,
    aft_df: pd.DataFrame | None = None,
    error: str = "se",
    dpi: int = 150,
) -> list[Path]:
    """1〜4 の CV 可視化をまとめて作成する。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_lambda_vs_c_td(
            fold_df,
            summary_df,
            output_dir,
            cox_df=cox_df,
            aft_df=aft_df,
            error=error,
            dpi=dpi,
        ),
        plot_train_test_c_td(
            summary_df,
            output_dir,
            cox_df=cox_df,
            aft_df=aft_df,
            error=error,
            dpi=dpi,
        ),
        plot_fold_spaghetti(
            fold_df, summary_df, output_dir, cox_df=cox_df, aft_df=aft_df, dpi=dpi
        ),
        plot_convergence_diagnostics(
            fold_df, summary_df, output_dir, error=error, dpi=dpi
        ),
    ]
    comparison_df = build_model_comparison_table(
        summary_df, cox_df=cox_df, aft_df=aft_df, error=error
    )
    if not comparison_df.empty and aft_df is not None:
        outputs.append(
            plot_model_comparison(comparison_df, output_dir, error=error, dpi=dpi)
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize real-data CV results")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/real_cv/support2/support2_5fold_seed1234"),
        help="Directory containing lambda_*/fold_*/result.json.",
    )
    parser.add_argument(
        "--fold-results",
        type=Path,
        default=None,
        help="Optional fold-level CSV. If absent, result.json files are collected.",
    )
    parser.add_argument(
        "--summary-by-lambda",
        type=Path,
        default=None,
        help="Optional lambda-level summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to <base-dir>/plots.",
    )
    parser.add_argument(
        "--cox-summary",
        type=Path,
        default=None,
        help=(
            "Optional Cox CV CSV. Accepts either cox_summary.csv "
            "or cox_fold_results.csv from compute_cox_baseline.py."
        ),
    )
    parser.add_argument(
        "--aft-summary",
        type=Path,
        default=None,
        help=(
            "Optional AFT CV CSV. Accepts either aft_summary.csv "
            "or aft_fold_results.csv from compute_aft_baseline.py."
        ),
    )
    parser.add_argument(
        "--error",
        choices=["se", "std", "none"],
        default="se",
        help="Error bars for mean lines.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution.")
    parser.add_argument(
        "--no-write-csv",
        action="store_true",
        help="Do not write fold_results.csv and summary_by_lambda.csv.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or (args.base_dir / "plots")
    fold_output = args.fold_results or (args.base_dir / "fold_results.csv")
    summary_output = args.summary_by_lambda or (args.base_dir / "summary_by_lambda.csv")

    fold_df, summary_df = load_or_collect_cv_results(
        args.base_dir,
        fold_results_path=args.fold_results,
        summary_path=args.summary_by_lambda,
    )

    if not args.no_write_csv:
        write_cv_tables(fold_df, summary_df, fold_output, summary_output)
        print(f"Saved fold results to: {fold_output}")
        print(f"Saved lambda summary to: {summary_output}")

    cox_df = pd.read_csv(args.cox_summary) if args.cox_summary is not None else None
    aft_df = pd.read_csv(args.aft_summary) if args.aft_summary is not None else None
    comparison_df = build_model_comparison_table(
        summary_df, cox_df=cox_df, aft_df=aft_df, error=args.error
    )
    if aft_df is not None and not comparison_df.empty and not args.no_write_csv:
        comparison_output = args.base_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_output, index=False, encoding="utf-8")
        print(f"Saved model comparison to: {comparison_output}")

    outputs = create_all_plots(
        fold_df,
        summary_df,
        output_dir,
        cox_df=cox_df,
        aft_df=aft_df,
        error=args.error,
        dpi=args.dpi,
    )
    for output in outputs:
        print(f"Saved plot to: {output}")

    best = summary_df.dropna(subset=["c_td_test_mean"]).sort_values(
        "c_td_test_mean", ascending=False
    )
    if not best.empty:
        row = best.iloc[0]
        print(
            "Best lambda by mean test c_td: "
            f"{float(row['lambda_fuse']):.6g} "
            f"(mean={float(row['c_td_test_mean']):.4f})"
        )


if __name__ == "__main__":
    main()
