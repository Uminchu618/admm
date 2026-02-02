from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_result_files(input_dir: Path) -> list[Path]:
    direct_json = list(sorted(input_dir.glob("*.json")))
    if direct_json:
        return direct_json
    nested_results = list(sorted(input_dir.glob("**/result.json")))
    return nested_results


def extract_seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("extended_aft_seed_"):
            try:
                return int(part.split("extended_aft_seed_")[-1])
            except ValueError:
                return None
    stem = path.stem
    if "seed_" in stem:
        try:
            return int(stem.split("seed_")[-1])
        except ValueError:
            return None
    return None


def extract_lambda_label(path: Path) -> Optional[str]:
    for part in path.parts:
        if part.startswith("lambda_") and part != "lambda_experiments":
            return part
    return None


def load_results(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        seed = extract_seed_from_path(path)
        lambda_label = extract_lambda_label(path)

        history = data.get("history", {}) or {}
        summary = data.get("summary", {}) or {}

        objective_hist = history.get("objective", [])
        primal_hist = history.get("primal_residual", [])
        dual_hist = history.get("dual_residual", [])

        rows.append(
            {
                "file": path.name,
                "path": path,
                "seed": seed,
                "lambda": lambda_label,
                "objective_last": summary.get(
                    "objective_last", objective_hist[-1] if objective_hist else np.nan
                ),
                "primal_residual_last": summary.get(
                    "primal_residual_last", primal_hist[-1] if primal_hist else np.nan
                ),
                "dual_residual_last": summary.get(
                    "dual_residual_last", dual_hist[-1] if dual_hist else np.nan
                ),
                "objective_init": objective_hist[0] if objective_hist else np.nan,
                "objective_min": np.min(objective_hist) if objective_hist else np.nan,
                "n_admm_iter": len(objective_hist),
            }
        )

    if not rows:
        raise FileNotFoundError("No JSON files found.")

    df = (
        pd.DataFrame(rows)
        .sort_values("seed", na_position="last")
        .reset_index(drop=True)
    )
    return df


def load_time_grid(files: list[Path]) -> Optional[np.ndarray]:
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        time_grid = data.get("time_grid", None)
        if time_grid is not None:
            return np.asarray(time_grid, dtype=float)
    return None


def compute_true_betas(
    time_grid: np.ndarray, config_path: Path
) -> Optional[dict[str, np.ndarray]]:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    td = cfg.get("time_dependence", {})
    scenario = cfg.get("scenario", None)
    if not td or scenario is None:
        return None

    t_left = time_grid[:-1]
    t_right = time_grid[1:]
    t_mid = 0.5 * (t_left + t_right)

    b11 = td["b11"]
    c1 = td["c1"]
    b21 = td["b21"]
    c2 = td["c2"]
    b31 = td["b31"]
    t0 = td["t0"]
    b30 = td["b30"]

    beta1 = b11 * np.exp(-c1 * t_mid)
    beta2 = b21 * np.log1p(c2 * t_mid)
    if int(scenario) == 1:
        beta3 = b31 * (t_mid - t0) ** 2
    else:
        beta3 = np.full_like(t_mid, b30, dtype=float)

    return {"x1": beta1, "x2": beta2, "x3": beta3}


def plot_coef_boxplot_by_k_combined(
    input_dir: Path, output_dir: Path, df: pd.DataFrame, config_path: Path
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    coef_rows = []
    for _, row in df.iterrows():
        path = row["path"]
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        coef = data.get("coef", None)
        if coef is None:
            continue
        coef_array = np.asarray(coef, dtype=float)
        if coef_array.ndim != 2:
            continue
        k_len, p_len = coef_array.shape
        for k in range(k_len):
            for j in range(p_len):
                coef_rows.append(
                    {
                        "seed": row["seed"],
                        "k": k,
                        "feature": f"x{j+1}",
                        "coef": coef_array[k, j],
                    }
                )

    if not coef_rows:
        return None

    coef_df = pd.DataFrame(coef_rows)

    features = sorted(coef_df["feature"].unique())
    k_values = sorted(coef_df["k"].unique())
    n_features = len(features)
    if n_features == 0:
        return output_dir / "coef_boxplot_by_k.png"

    time_grid = load_time_grid(list(df["path"]))
    true_betas = None
    if time_grid is not None:
        true_betas = compute_true_betas(time_grid, config_path)

    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4), sharey=True)
    if n_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, features, strict=False):
        sub = coef_df.loc[coef_df["feature"] == feature]
        data = [sub.loc[sub["k"] == k, "coef"].values for k in k_values]
        ax.boxplot(data, tick_labels=[str(k) for k in k_values], showfliers=False)
        if true_betas is not None and feature in true_betas:
            true_vals = true_betas[feature]
            if len(true_vals) == len(k_values):
                x_pos = np.arange(1, len(k_values) + 1)
                ax.plot(
                    x_pos,
                    true_vals,
                    color="#D62728",
                    marker="o",
                    linewidth=1.5,
                    label="true",
                )
        ax.set_title(f"{feature}")
        ax.set_xlabel("Time segment k")
        ax.grid(True, axis="y", alpha=0.3)
        if true_betas is not None and feature in true_betas:
            ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Coefficient")
    fig.suptitle("Coefficient distribution by k (per feature)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / "coef_boxplot_by_k.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def remove_legacy_plots(output_dir: Path) -> None:
    legacy_files = [
        "objective_last_by_seed.png",
        "objective_last_hist.png",
        "residuals_scatter.png",
        "summary_boxplot.png",
        "objective_convergence_examples.png",
        "coef_hist_all.png",
        "coef_boxplot_by_feature.png",
        "coef_boxplot_by_k_x1.png",
        "coef_boxplot_by_k_x2.png",
        "coef_boxplot_by_k_x3.png",
        "summary.csv",
    ]
    for name in legacy_files:
        path = output_dir / name
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize extended AFT JSON results.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs") / "extended_aft",
        help="Directory containing JSON result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "extended_aft" / "plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=Path("generation") / "extended_aft_generator.config.json",
        help="Generator config JSON for true beta overlay.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    files = find_result_files(input_dir)
    if not files and args.input_dir == Path("outputs") / "extended_aft":
        fallback_dir = Path("outputs") / "lambda_experiments"
        files = find_result_files(fallback_dir)
        if files:
            input_dir = fallback_dir

    if not files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    output_dir = args.output_dir
    default_output = Path("outputs") / "extended_aft" / "plots"
    if output_dir == default_output and input_dir != Path("outputs") / "extended_aft":
        output_dir = input_dir / "plots"

    df = load_results(files)

    if df["lambda"].notna().any():
        for lambda_label, sub in df.groupby("lambda"):
            if pd.isna(lambda_label):
                continue
            out_dir = output_dir / str(lambda_label)
            remove_legacy_plots(out_dir)
            output_path = plot_coef_boxplot_by_k_combined(
                input_dir, out_dir, sub, args.generator_config
            )
            if output_path is not None:
                print(f"Saved plot to: {output_path}")
            else:
                print(f"No coefficients found for {lambda_label}.")
    else:
        remove_legacy_plots(output_dir)
        output_path = plot_coef_boxplot_by_k_combined(
            input_dir, output_dir, df, args.generator_config
        )
        if output_path is not None:
            print(f"Saved plot to: {output_path}")
        else:
            print("No coefficients found. Plot was not generated.")


if __name__ == "__main__":
    main()
