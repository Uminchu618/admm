#!/usr/bin/env python3
"""BIC選択モデルの係数関数を生成時の真値と比較する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


DEFAULT_RUN = "adaptive_rho_normalized_stagnation_escape_newton5"
SCENARIOS = ["oracle", "fine_grid", "off_grid", "small", "no_change"]
LABELS = {
    "oracle": "Oracle",
    "fine_grid": "Fine-grid",
    "off_grid": "Off-grid",
    "small": "Small",
    "no_change": "No-change",
}
COLORS = {
    "oracle": "#3568A8",
    "fine_grid": "#2A9D8F",
    "off_grid": "#E76F51",
    "small": "#8E63B0",
    "no_change": "#6B7280",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def parse_scenarios(value: str) -> list[str]:
    scenarios = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(scenarios) - set(SCENARIOS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown scenarios: {invalid}")
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return scenarios


def scenario_and_seed(data_name: str) -> tuple[str, int]:
    for scenario in SCENARIOS:
        prefix = f"{scenario}_seed_"
        if data_name.startswith(prefix):
            return scenario, int(data_name.removeprefix(prefix))
    raise ValueError(f"unknown pilot data name: {data_name}")


def select_bic_fits(
    summary: pd.DataFrame, scenarios: Iterable[str], seeds: Iterable[int]
) -> pd.DataFrame:
    required = {"data_name", "lambda_fuse", "bic", "bic_eligible", "result_path"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    parsed = summary["data_name"].map(scenario_and_seed)
    data = summary.copy()
    data["scenario"] = parsed.map(lambda item: item[0])
    data["seed"] = parsed.map(lambda item: item[1])
    eligible = data.loc[
        _as_bool(data["bic_eligible"])
        & data["bic"].notna()
        & data["scenario"].isin(scenarios)
        & data["seed"].isin(seeds)
    ].copy()
    if eligible.empty:
        raise ValueError("no BIC-eligible fits matched the requested scenarios and seeds")

    selected = eligible.loc[eligible.groupby("data_name")["bic"].idxmin()].copy()
    selected = selected.sort_values(["scenario", "seed"]).reset_index(drop=True)
    expected = {(scenario, seed) for scenario in scenarios for seed in seeds}
    observed = set(zip(selected["scenario"], selected["seed"]))
    if observed != expected:
        missing_pairs = sorted(expected - observed)
        raise ValueError(f"no BIC-eligible fit for scenario/seed pairs: {missing_pairs}")
    return selected


def load_truth(config_path: Path) -> tuple[np.ndarray, np.ndarray]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stepwise = config["stepwise_beta"]
    grid = np.asarray(stepwise["true_time_grid"], dtype=float)
    levels = np.asarray(
        [stepwise[f"beta{index}_levels"] for index in range(1, 4)], dtype=float
    ).T
    if levels.shape != (len(grid) - 1, 3):
        raise ValueError(f"truth shape does not match true_time_grid in {config_path}")
    return grid, levels


def coefficient_rmise(
    analysis_grid: np.ndarray,
    estimate: np.ndarray,
    true_grid: np.ndarray,
    truth: np.ndarray,
) -> float:
    union_grid = np.unique(np.concatenate([analysis_grid, true_grid]))
    integrated_error = 0.0
    for left, right in zip(union_grid[:-1], union_grid[1:]):
        midpoint = (left + right) / 2.0
        estimate_index = int(
            np.clip(
                np.searchsorted(analysis_grid, midpoint, side="right") - 1,
                0,
                len(analysis_grid) - 2,
            )
        )
        truth_index = int(
            np.clip(
                np.searchsorted(true_grid, midpoint, side="right") - 1,
                0,
                len(true_grid) - 2,
            )
        )
        integrated_error += (right - left) * np.sum(
            (estimate[estimate_index] - truth[truth_index]) ** 2
        )
    duration = union_grid[-1] - union_grid[0]
    return float(np.sqrt(integrated_error / (estimate.shape[1] * duration)))


def step_values(levels: np.ndarray) -> np.ndarray:
    """区間ごとの値をstep描画用に右端まで延長する。"""

    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("levels must be a non-empty one-dimensional array")
    return np.append(levels, levels[-1])


def resolve_result_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / "outputs" / path


def build_scenario_figure(
    scenario: str,
    rows: pd.DataFrame,
    root: Path,
    true_grid: np.ndarray,
    truth: np.ndarray,
) -> tuple[plt.Figure, list[dict[str, object]]]:
    loaded: list[tuple[object, np.ndarray, np.ndarray, Path, float]] = []
    records: list[dict[str, object]] = []
    for row in rows.sort_values("seed").itertuples(index=False):
        result_path = resolve_result_path(root, row.result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        analysis_grid = np.asarray(result["time_grid"], dtype=float)
        estimate = np.asarray(result["coef"], dtype=float)
        if estimate.shape != (len(analysis_grid) - 1, truth.shape[1]):
            raise ValueError(f"coefficient shape does not match grid in {result_path}")
        rmise = coefficient_rmise(analysis_grid, estimate, true_grid, truth)
        loaded.append((row, analysis_grid, estimate, result_path, rmise))
        records.append(
            {
                "data_name": row.data_name,
                "scenario": scenario,
                "seed": int(row.seed),
                "lambda_fuse": float(row.lambda_fuse),
                "bic": float(row.bic),
                "c_td_test": getattr(row, "c_td_test", np.nan),
                "coefficient_rmise": rmise,
                "result_path": str(result_path.relative_to(root)),
            }
        )

    fig, axes = plt.subplots(
        len(loaded),
        truth.shape[1],
        figsize=(12.8, 2.35 * len(loaded) + 1.2),
        sharex=True,
        squeeze=False,
    )
    color = COLORS[scenario]
    for coefficient_index in range(truth.shape[1]):
        all_values = [truth[:, coefficient_index]] + [
            estimate[:, coefficient_index] for _, _, estimate, _, _ in loaded
        ]
        low = min(float(values.min()) for values in all_values)
        high = max(float(values.max()) for values in all_values)
        margin = max(0.08, 0.12 * (high - low))
        for row_index, (row, analysis_grid, estimate, _, rmise) in enumerate(loaded):
            ax = axes[row_index, coefficient_index]
            ax.step(
                true_grid,
                step_values(truth[:, coefficient_index]),
                where="post",
                color="#1F2937",
                linestyle="--",
                linewidth=2.2,
                label="Truth",
                zorder=3,
            )
            ax.step(
                analysis_grid,
                step_values(estimate[:, coefficient_index]),
                where="post",
                color=color,
                linewidth=2.0,
                label="BIC-selected estimate",
                zorder=2,
            )
            true_changes = true_grid[1:-1][
                np.abs(np.diff(truth[:, coefficient_index])) > 1e-12
            ]
            for change in true_changes:
                ax.axvline(change, color="#9CA3AF", linewidth=0.8, alpha=0.45)
            ax.axhline(0.0, color="#CBD5E1", linewidth=0.7, zorder=0)
            ax.set_ylim(low - margin, high + margin)
            ax.grid(alpha=0.16)
            if row_index == 0:
                ax.set_title(rf"$\beta_{{{coefficient_index + 1}}}(t)$", fontsize=12)
            if coefficient_index == 0:
                ax.set_ylabel(f"Seed {int(row.seed)}\nCoefficient")
            if row_index == len(loaded) - 1:
                ax.set_xlabel("Time t")
            if coefficient_index == truth.shape[1] - 1:
                ax.text(
                    0.98,
                    0.94,
                    f"lambda={row.lambda_fuse:g}\nRMISE={rmise:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#D1D5DB",
                        "alpha": 0.9,
                    },
                )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"{LABELS[scenario]}: true and BIC-selected coefficient functions",
        y=1.015,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.8, w_pad=1.0)
    return fig, records


def write_report(records: pd.DataFrame, scenarios: list[str], output_path: Path) -> None:
    lines = [
        "# BIC選択係数関数と真値の比較",
        "",
        "黒破線が生成時の真値、色付き実線が正式収束した候補のうちBIC最小の推定値である。",
        "各行はseed、各列は係数を表し、右列に選択lambdaと係数RMISEを示す。",
        "",
    ]
    for scenario in scenarios:
        lines.extend(
            [
                f"## {LABELS[scenario]}",
                "",
                f"![{LABELS[scenario]}](bic_selected_beta_{scenario}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 選択結果",
            "",
            "| Scenario | Seed | lambda | BIC | Ctd(test) | RMISE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in records.itertuples(index=False):
        lines.append(
            f"| {LABELS[row.scenario]} | {row.seed} | {row.lambda_fuse:g} | "
            f"{row.bic:.2f} | {row.c_td_test:.4f} | {row.coefficient_rmise:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/pilot") / f"{DEFAULT_RUN}_summary.csv",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44,45,46"))
    parser.add_argument(
        "--scenarios", type=parse_scenarios, default=parse_scenarios(",".join(SCENARIOS))
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pilot") / f"{DEFAULT_RUN}_beta_comparison",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    selected = select_bic_fits(summary, args.scenarios, args.seeds)
    all_records: list[dict[str, object]] = []
    pdf_path = output_dir / "bic_selected_beta_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        for scenario in args.scenarios:
            true_grid, truth = load_truth(
                root / "generation" / "pilot" / f"{scenario}.json"
            )
            rows = selected.loc[selected["scenario"] == scenario]
            fig, records = build_scenario_figure(
                scenario, rows, root, true_grid, truth
            )
            fig.savefig(
                output_dir / f"bic_selected_beta_{scenario}.png",
                dpi=200,
                bbox_inches="tight",
                facecolor="white",
            )
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            all_records.extend(records)

    records = pd.DataFrame(all_records)
    records.to_csv(output_dir / "bic_selected_beta_records.csv", index=False)
    write_report(records, args.scenarios, output_dir / "bic_selected_beta_report.md")
    print(f"Selected fits: {len(records)}")
    print(f"Output directory: {output_dir}")
    print(records.to_string(index=False))


if __name__ == "__main__":
    main()
