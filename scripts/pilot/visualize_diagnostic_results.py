#!/usr/bin/env python3
"""Visualize convergence behavior in the 54-fit pilot diagnostic run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STOP_COLORS = {
    "residual_converged": "#2A9D8F",
    "stagnated": "#E9C46A",
    "max_iter": "#E76F51",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_summary(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "data_name",
        "lambda_fuse",
        "converged",
        "stopping_reason",
        "n_admm_iter",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["converged"] = _as_bool(data["converged"])
    data["scenario"] = data["data_name"].str.replace(
        r"_seed_\d+$", "", regex=True
    )
    return data


def plot_diagnostics(data: pd.DataFrame, output_path: Path) -> None:
    lambdas = sorted(data["lambda_fuse"].unique())
    labels = [f"{value:g}" for value in lambdas]
    x = range(len(lambdas))

    by_lambda = data.groupby("lambda_fuse", sort=True)
    total = by_lambda.size().reindex(lambdas)
    converged = by_lambda["converged"].sum().reindex(lambdas)
    failed = total - converged

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    fig.suptitle(
        "54-fit diagnostic after the first solver fix\n"
        "Oracle and Fine-grid, seeds 42–44",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.bar(x, converged, color=STOP_COLORS["residual_converged"], label="Converged")
    ax.bar(x, failed, bottom=converged, color="#D9D9D9", label="Failed")
    for i, (ok, n) in enumerate(zip(converged, total, strict=True)):
        ax.text(i, n + 0.12, f"{int(ok)}/{int(n)}", ha="center", fontsize=9)
    ax.set_title("A. Formal convergence by lambda")
    ax.set_ylabel("Fits")
    ax.set_xticks(list(x), labels, rotation=40, ha="right")
    ax.set_ylim(0, max(total) + 1.0)
    ax.legend(frameon=False, ncols=2)

    ax = axes[0, 1]
    scenario_order = [name for name in ["oracle", "fine_grid"] if name in set(data["scenario"])]
    scenario_total = data.groupby("scenario").size().reindex(scenario_order)
    scenario_ok = data.groupby("scenario")["converged"].sum().reindex(scenario_order)
    scenario_fail = scenario_total - scenario_ok
    y = range(len(scenario_order))
    ax.barh(y, scenario_ok, color=STOP_COLORS["residual_converged"], label="Converged")
    ax.barh(y, scenario_fail, left=scenario_ok, color="#D9D9D9", label="Failed")
    for i, (ok, n) in enumerate(zip(scenario_ok, scenario_total, strict=True)):
        ax.text(n + 0.35, i, f"{int(ok)}/{int(n)} ({ok / n:.1%})", va="center", fontsize=10)
    ax.set_title("B. Formal convergence by scenario")
    ax.set_xlabel("Fits")
    ax.set_yticks(list(y), [name.replace("_", " ").title() for name in scenario_order])
    ax.set_xlim(0, max(scenario_total) + 7)

    ax = axes[1, 0]
    stop_counts = (
        data.groupby(["lambda_fuse", "stopping_reason"])
        .size()
        .unstack(fill_value=0)
        .reindex(lambdas, fill_value=0)
    )
    bottom = pd.Series(0, index=lambdas, dtype=float)
    for reason in ["residual_converged", "stagnated", "max_iter"]:
        values = stop_counts.get(reason, pd.Series(0, index=lambdas))
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=STOP_COLORS[reason],
            label=reason.replace("_", " "),
        )
        bottom = bottom + values
    ax.set_title("C. Stopping reason by lambda")
    ax.set_ylabel("Fits")
    ax.set_xticks(list(x), labels, rotation=40, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    scenario_colors = {"oracle": "#457B9D", "fine_grid": "#F4A261"}
    for scenario in scenario_order:
        medians = (
            data.loc[data["scenario"] == scenario]
            .groupby("lambda_fuse")["n_admm_iter"]
            .median()
            .reindex(lambdas)
        )
        ax.plot(
            list(x),
            medians,
            marker="o",
            linewidth=2,
            color=scenario_colors[scenario],
            label=scenario.replace("_", " ").title(),
        )
    ax.axhline(1000, color="#555555", linestyle="--", linewidth=1, label="max_iter = 1000")
    ax.set_yscale("log")
    ax.set_title("D. Median ADMM iterations (all fits)")
    ax.set_ylabel("Iterations, log scale")
    ax.set_xticks(list(x), labels, rotation=40, ha="right")
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
    for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("lambda")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/pilot_diagnostic/adaptive_rho_newton5_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/pilot_diagnostic_adaptive_rho_newton5.png"),
    )
    args = parser.parse_args()
    plot_diagnostics(load_summary(args.summary), args.output)
    print(f"Saved diagnostic visualization to: {args.output}")


if __name__ == "__main__":
    main()
