#!/usr/bin/env python3
"""Pilot simulation summary の診断表と図を生成する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


SCENARIO_ORDER = ["oracle", "fine_grid", "off_grid", "small", "no_change"]
SCENARIO_LABELS = {
    "oracle": "Oracle",
    "fine_grid": "Fine-grid",
    "off_grid": "Off-grid",
    "small": "Small",
    "no_change": "No-change",
}
SCENARIO_COLORS = {
    "oracle": "#3568A8",
    "fine_grid": "#2A9D8F",
    "off_grid": "#E76F51",
    "small": "#8E63B0",
    "no_change": "#6B7280",
}


def load_summary(path: Path, residual_threshold: float) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "data_name",
        "lambda_fuse",
        "primal_residual_last",
        "dual_residual_last",
        "c_td_train",
        "c_td_test",
        "n_change_points",
        "bic",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    extracted = data["data_name"].str.extract(
        r"^(oracle|fine_grid|off_grid|small|no_change)_seed_(\d+)$"
    )
    data["scenario"] = extracted[0]
    data["seed"] = pd.to_numeric(extracted[1], errors="coerce")
    if data[["scenario", "seed"]].isna().any().any():
        invalid = data.loc[data["scenario"].isna(), "data_name"].unique().tolist()
        raise ValueError(f"Could not parse scenario/seed from data_name: {invalid[:5]}")
    data["seed"] = data["seed"].astype(int)
    data["scenario"] = pd.Categorical(
        data["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    data["c_td_gap"] = data["c_td_train"] - data["c_td_test"]
    data["residual_screen_pass"] = (
        (data["primal_residual_last"] <= residual_threshold)
        & (data["dual_residual_last"] <= residual_threshold)
    )
    data["delta_bic"] = data["bic"] - data.groupby("data_name")["bic"].transform(
        "min"
    )
    return data


def aggregate_lambda(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["scenario", "lambda_fuse"], observed=True)
    result = grouped.agg(
        n=("seed", "count"),
        c_td_test_mean=("c_td_test", "mean"),
        c_td_test_sd=("c_td_test", "std"),
        c_td_train_mean=("c_td_train", "mean"),
        c_td_gap_mean=("c_td_gap", "mean"),
        change_points_mean=("n_change_points", "mean"),
        change_points_sd=("n_change_points", "std"),
        bic_mean=("bic", "mean"),
        delta_bic_mean=("delta_bic", "mean"),
        delta_bic_median=("delta_bic", "median"),
        primal_residual_median=("primal_residual_last", "median"),
        primal_residual_q25=("primal_residual_last", lambda x: x.quantile(0.25)),
        primal_residual_q75=("primal_residual_last", lambda x: x.quantile(0.75)),
        dual_residual_median=("dual_residual_last", "median"),
        residual_screen_pass_rate=("residual_screen_pass", "mean"),
    ).reset_index()
    result["c_td_test_se"] = result["c_td_test_sd"] / np.sqrt(result["n"])
    result["change_points_se"] = result["change_points_sd"] / np.sqrt(result["n"])
    result["scenario"] = result["scenario"].astype(str)
    return result


def select_by_bic(data: pd.DataFrame) -> pd.DataFrame:
    eligible = data.dropna(subset=["bic"]).copy()
    if "bic_eligible" in eligible.columns:
        eligible = eligible.loc[eligible["bic_eligible"].fillna(False)]
    if eligible.empty:
        return data.iloc[0:0].copy()
    selected = eligible.loc[eligible.groupby("data_name")["bic"].idxmin()].copy()
    selected["scenario"] = selected["scenario"].astype(str)
    return selected.sort_values(["scenario", "seed"])


def aggregate_selected(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "n",
                "lambda_mean",
                "lambda_median",
                "lambda_zero_rate",
                "c_td_test_mean",
                "c_td_test_sd",
                "c_td_gap_mean",
                "change_points_mean",
                "change_points_sd",
                "primal_residual_median",
                "residual_screen_pass_rate",
            ]
        )
    result = selected.groupby("scenario").agg(
        n=("seed", "count"),
        lambda_mean=("lambda_fuse", "mean"),
        lambda_median=("lambda_fuse", "median"),
        lambda_zero_rate=("lambda_fuse", lambda x: (x == 0).mean()),
        c_td_test_mean=("c_td_test", "mean"),
        c_td_test_sd=("c_td_test", "std"),
        c_td_gap_mean=("c_td_gap", "mean"),
        change_points_mean=("n_change_points", "mean"),
        change_points_sd=("n_change_points", "std"),
        primal_residual_median=("primal_residual_last", "median"),
        residual_screen_pass_rate=("residual_screen_pass", "mean"),
    )
    result = result.reindex(SCENARIO_ORDER).reset_index()
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def set_lambda_axis(ax: plt.Axes, lambdas: list[float]) -> None:
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{value:g}" for value in lambdas], rotation=45, ha="right")
    ax.set_xlabel("lambda (mean-loss scale)")


def plot_lambda_diagnostics(
    summary: pd.DataFrame,
    output_path: Path,
    residual_threshold: float,
) -> None:
    lambdas = sorted(summary["lambda_fuse"].unique().tolist())
    x = np.arange(len(lambdas))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for scenario in SCENARIO_ORDER:
        subset = summary[summary["scenario"] == scenario].set_index("lambda_fuse").reindex(
            lambdas
        )
        label = SCENARIO_LABELS[scenario]
        color = SCENARIO_COLORS[scenario]

        axes[0, 0].plot(
            x,
            subset["residual_screen_pass_rate"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        residual = subset["primal_residual_median"].clip(lower=1e-6)
        axes[0, 1].plot(x, residual, marker="o", linewidth=2, color=color, label=label)

        mean = subset["c_td_test_mean"].to_numpy(dtype=float)
        ci = 1.96 * subset["c_td_test_se"].to_numpy(dtype=float)
        axes[1, 0].plot(x, mean, marker="o", linewidth=2, color=color, label=label)
        axes[1, 0].fill_between(x, mean - ci, mean + ci, color=color, alpha=0.12)

        cp_mean = subset["change_points_mean"].to_numpy(dtype=float)
        cp_ci = 1.96 * subset["change_points_se"].to_numpy(dtype=float)
        axes[1, 1].plot(x, cp_mean, marker="o", linewidth=2, color=color, label=label)
        axes[1, 1].fill_between(
            x, np.maximum(cp_mean - cp_ci, 0), cp_mean + cp_ci, color=color, alpha=0.12
        )

    axes[0, 0].set_title("A. Residual-screen pass rate")
    axes[0, 0].set_ylabel("Pass rate")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[0, 1].set_title("B. Median primal residual")
    axes[0, 1].set_ylabel("Primal residual (log scale)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(
        residual_threshold,
        color="#B91C1C",
        linestyle="--",
        linewidth=1.5,
        label=f"screen threshold = {residual_threshold:g}",
    )

    axes[1, 0].set_title("C. Independent-test time-dependent C-index")
    axes[1, 0].set_ylabel("Test Ctd (mean and 95% MC CI)")
    axes[1, 0].axhline(0.5, color="#6B7280", linestyle=":", linewidth=1.3)
    axes[1, 0].set_ylim(0.48, 0.73)

    axes[1, 1].set_title("D. Estimated number of change points")
    axes[1, 1].set_ylabel("Mean count and 95% MC CI")
    axes[1, 1].set_ylim(bottom=-0.5)

    for ax in axes.flat:
        set_lambda_axis(ax, lambdas)
        style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Pilot lambda-path diagnostics\nResidual screen is diagnostic only, not the solver's formal stopping rule",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, hspace=0.38, wspace=0.22)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_boxplot(
    ax: plt.Axes,
    selected: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    rng: np.random.Generator,
) -> None:
    values = [
        selected.loc[selected["scenario"] == scenario, column].to_numpy(dtype=float)
        for scenario in SCENARIO_ORDER
    ]
    positions = np.arange(1, len(SCENARIO_ORDER) + 1)
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.7},
        whiskerprops={"color": "#6B7280"},
        capprops={"color": "#6B7280"},
    )
    for patch, scenario in zip(boxes["boxes"], SCENARIO_ORDER):
        patch.set_facecolor(SCENARIO_COLORS[scenario])
        patch.set_alpha(0.65)
        patch.set_edgecolor("#374151")
    for position, scenario_values, scenario in zip(positions, values, SCENARIO_ORDER):
        jitter = rng.normal(0.0, 0.055, size=len(scenario_values))
        ax.scatter(
            position + jitter,
            scenario_values,
            s=18,
            alpha=0.65,
            color=SCENARIO_COLORS[scenario],
            edgecolors="white",
            linewidths=0.3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([SCENARIO_LABELS[value] for value in SCENARIO_ORDER], rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)


def plot_bic_diagnostics(
    selected: pd.DataFrame,
    output_path: Path,
    residual_threshold: float,
) -> None:
    if selected.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No formally converged BIC-eligible fits",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rng = np.random.default_rng(20260818)

    selection = pd.crosstab(selected["scenario"], selected["lambda_fuse"], normalize="index")
    selection = selection.reindex(SCENARIO_ORDER).fillna(0.0)
    bottom = np.zeros(len(SCENARIO_ORDER))
    lambda_colors = ["#3568A8", "#F4A261", "#E76F51", "#8E63B0"]
    for idx, lambda_value in enumerate(selection.columns):
        values = selection[lambda_value].to_numpy(dtype=float)
        axes[0, 0].bar(
            np.arange(len(SCENARIO_ORDER)),
            values,
            bottom=bottom,
            width=0.68,
            color=lambda_colors[idx % len(lambda_colors)],
            label=f"lambda={lambda_value:g}",
        )
        bottom += values
    axes[0, 0].set_xticks(np.arange(len(SCENARIO_ORDER)))
    axes[0, 0].set_xticklabels(
        [SCENARIO_LABELS[value] for value in SCENARIO_ORDER], rotation=20
    )
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylabel("Share of 20 replications")
    axes[0, 0].set_title("A. Lambda selected by minimum BIC")
    axes[0, 0].legend(frameon=False, loc="upper left")
    style_axis(axes[0, 0])

    add_boxplot(
        axes[0, 1],
        selected,
        "primal_residual_last",
        "Primal residual",
        "B. Primal residual of BIC-selected fit",
        rng,
    )
    axes[0, 1].axhline(
        residual_threshold,
        color="#B91C1C",
        linestyle="--",
        linewidth=1.5,
        label=f"screen threshold = {residual_threshold:g}",
    )
    axes[0, 1].legend(frameon=False, loc="upper left")

    add_boxplot(
        axes[1, 0],
        selected,
        "c_td_test",
        "Independent-test Ctd",
        "C. Predictive performance of BIC-selected fit",
        rng,
    )
    axes[1, 0].axhline(0.5, color="#6B7280", linestyle=":", linewidth=1.3)

    add_boxplot(
        axes[1, 1],
        selected,
        "n_change_points",
        "Estimated count",
        "D. Change-point count of BIC-selected fit",
        rng,
    )
    axes[1, 1].set_ylim(bottom=-1)

    fig.suptitle(
        "BIC-selection diagnostics\nInterpret positive-lambda selections only after convergence is verified",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.10, hspace=0.42, wspace=0.20)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    source_path: Path,
    data: pd.DataFrame,
    selected: pd.DataFrame,
    selected_summary: pd.DataFrame,
    residual_threshold: float,
) -> None:
    positive = data[data["lambda_fuse"] > 0]
    positive_pass = int(positive["residual_screen_pass"].sum())
    selected_positive = selected[selected["lambda_fuse"] > 0]
    selected_pass = int(selected_positive["residual_screen_pass"].sum())
    lines = [
        "# Pilot simulation visualization report",
        "",
        f"- Source: `{source_path}`",
        f"- Rows: {len(data):,}",
        f"- Datasets: {data['data_name'].nunique():,}",
        f"- Lambda candidates: {data['lambda_fuse'].nunique():,}",
        f"- Residual screening rule: primal and dual residual <= {residual_threshold:g}",
        "- This screen is deliberately conservative and is not a replacement for the solver's recorded stopping rule.",
        "",
        "## Main diagnostic finding",
        "",
        f"- Positive-lambda fits passing the residual screen: {positive_pass}/{len(positive)}.",
        f"- BIC-selected positive-lambda fits passing the residual screen: {selected_pass}/{len(selected_positive)}.",
        f"- BIC selected a positive lambda in {len(selected_positive)}/{len(selected)} datasets.",
        "- Therefore, positive-lambda BIC results should not be interpreted as valid fused-lasso estimates until convergence is fixed or formally verified from result.json.",
        "",
        "## BIC-selected descriptive summary",
        "",
        "| Scenario | n | Median lambda | Mean test Ctd | Mean change points | Median primal residual | Screen pass rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if selected.empty:
        lines.extend(
            [
                "| — | 0 | — | — | — | — | — |",
                "",
                "No dataset had a formally converged BIC-eligible candidate.",
            ]
        )
    for row in selected_summary.itertuples(index=False):
        lines.append(
            f"| {SCENARIO_LABELS[row.scenario]} | {int(row.n)} | "
            f"{row.lambda_median:.3g} | {row.c_td_test_mean:.4f} | "
            f"{row.change_points_mean:.2f} | {row.primal_residual_median:.4f} | "
            f"{row.residual_screen_pass_rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Scope limitation",
            "",
            "The summary CSV does not contain coefficient-function error, matched change-point precision/recall, or localization error. Those require the per-fit result.json files and the scenario truth definitions.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/pilot_summary.csv"),
        help="Path to pilot_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pilot_visualizations"),
        help="Directory for PNG, CSV, and Markdown outputs.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=1e-2,
        help="Diagnostic residual-screen threshold; not a formal convergence rule.",
    )
    args = parser.parse_args()
    if args.residual_threshold <= 0:
        raise ValueError("--residual-threshold must be positive")

    data = load_summary(args.summary, args.residual_threshold)
    lambda_summary = aggregate_lambda(data)
    selected = select_by_bic(data)
    selected_summary = aggregate_selected(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambda_summary.to_csv(args.output_dir / "lambda_summary_by_scenario.csv", index=False)
    selected.to_csv(args.output_dir / "bic_selected_records.csv", index=False)
    selected_summary.to_csv(
        args.output_dir / "bic_selected_summary_by_scenario.csv", index=False
    )
    plot_lambda_diagnostics(
        lambda_summary,
        args.output_dir / "pilot_lambda_diagnostics.png",
        args.residual_threshold,
    )
    plot_bic_diagnostics(
        selected,
        args.output_dir / "pilot_bic_selected_diagnostics.png",
        args.residual_threshold,
    )
    write_report(
        args.output_dir / "pilot_visualization_report.md",
        args.summary,
        data,
        selected,
        selected_summary,
        args.residual_threshold,
    )
    print(f"Saved pilot visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
