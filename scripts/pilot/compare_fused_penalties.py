#!/usr/bin/env python3
"""同一 seed・同一 lambda 候補の fused lasso と fused MCP を比較する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pilot.prepare_slides22_results import (  # noqa: E402
    SCENARIOS,
    change_point_counts,
    coefficient_rmise,
)
from scripts.pilot.visualize_cv_results import (  # noqa: E402
    prepare_cv_selected_records,
)


METHODS = ("lasso", "mcp")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _scenario_seed(data_name: str) -> tuple[str, int]:
    for scenario in SCENARIOS:
        prefix = f"{scenario}_seed_"
        if data_name.startswith(prefix):
            return scenario, int(data_name.removeprefix(prefix))
    raise ValueError(f"unknown pilot data name: {data_name}")


def _resolve_result_path(value: str, result_root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if result_root is not None:
        candidate = result_root / path
        if candidate.exists():
            return candidate
    return ROOT / "outputs" / path


def _truth_configs() -> dict[str, dict[str, object]]:
    return {
        scenario: json.loads(
            (ROOT / "generation" / "pilot" / f"{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario in SCENARIOS
    }


def evaluate_method(
    selections: pd.DataFrame,
    refits: pd.DataFrame,
    *,
    method: str,
    z_tolerance: float,
    result_root: Path | None = None,
) -> pd.DataFrame:
    """CV 選択済み refit に真値ベースの係数・変化点指標を付ける。"""

    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    joined = prepare_cv_selected_records(selections, refits)
    truths = _truth_configs()
    rows: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        scenario, seed = _scenario_seed(row.data_name)
        result_path = _resolve_result_path(str(row.result_path), result_root)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result.get("config", {})
        recorded_method = result.get("history", {}).get(
            "fuse_penalty", config.get("fuse_penalty", "lasso")
        )
        if recorded_method != method:
            raise ValueError(
                f"penalty mismatch for {row.data_name}: "
                f"expected={method}, actual={recorded_method}"
            )
        true_positive, detected, truth = change_point_counts(
            result, truths[scenario], scenario, z_tolerance
        )
        rows.append(
            {
                "method": method,
                "data_name": row.data_name,
                "scenario": scenario,
                "seed": seed,
                "lambda_fuse": float(row.selected_lambda),
                "mcp_gamma": config.get("mcp_gamma") if method == "mcp" else None,
                "cv_mean_c_td": float(row.mean_c_td),
                "c_td_train": float(row.c_td_train),
                "c_td_test": float(row.c_td_test),
                "converged": _as_bool(row.converged),
                "rmise": coefficient_rmise(result, truths[scenario]),
                "true_positive": true_positive,
                "detected": detected,
                "truth": truth,
                "false_positive": detected - true_positive,
                "initialization_source": result.get(
                    "initialization_source",
                    result.get("history", {}).get("initialization_source", "default"),
                ),
                "result_path": str(row.result_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "seed"]).reset_index(drop=True)


def summarize_methods(records: pd.DataFrame) -> pd.DataFrame:
    """予測、係数誤差、変化点検出を penalty × scenario で要約する。"""

    rows: list[dict[str, object]] = []
    for method in METHODS:
        for scenario in SCENARIOS:
            subset = records.loc[
                (records["method"] == method) & (records["scenario"] == scenario)
            ]
            if subset.empty:
                continue
            n = len(subset)
            tp = int(subset["true_positive"].sum())
            detected = int(subset["detected"].sum())
            truth = int(subset["truth"].sum())
            precision = tp / detected if detected else np.nan
            recall = tp / truth if truth else np.nan
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if np.isfinite(precision + recall) and precision + recall > 0.0
                else np.nan
            )
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "n": n,
                    "convergence_rate": subset["converged"].mean(),
                    "lambda_mean": subset["lambda_fuse"].mean(),
                    "lambda_median": subset["lambda_fuse"].median(),
                    "c_td_test_mean": subset["c_td_test"].mean(),
                    "c_td_test_se": subset["c_td_test"].std(ddof=1) / np.sqrt(n),
                    "rmise_mean": subset["rmise"].mean(),
                    "rmise_se": subset["rmise"].std(ddof=1) / np.sqrt(n),
                    "change_points_mean": subset["detected"].mean(),
                    "true_positive": tp,
                    "detected": detected,
                    "truth": truth,
                    "false_positive": detected - tp,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def pair_methods(records: pd.DataFrame) -> pd.DataFrame:
    """同一シナリオ・seed の MCP − lasso 差を作る。"""

    lasso = records.loc[records["method"] == "lasso"].drop(columns="method")
    mcp = records.loc[records["method"] == "mcp"].drop(columns="method")
    paired = lasso.merge(
        mcp,
        on=["data_name", "scenario", "seed"],
        suffixes=("_lasso", "_mcp"),
        validate="one_to_one",
    )
    for metric in (
        "lambda_fuse",
        "cv_mean_c_td",
        "c_td_test",
        "rmise",
        "detected",
        "true_positive",
        "false_positive",
    ):
        paired[f"mcp_minus_lasso_{metric}"] = (
            paired[f"{metric}_mcp"] - paired[f"{metric}_lasso"]
        )
    return paired.sort_values(["scenario", "seed"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lasso-selections", type=Path, required=True)
    parser.add_argument("--lasso-refits", type=Path, required=True)
    parser.add_argument("--mcp-selections", type=Path, required=True)
    parser.add_argument("--mcp-refits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    lasso = evaluate_method(
        pd.read_csv(args.lasso_selections),
        pd.read_csv(args.lasso_refits),
        method="lasso",
        z_tolerance=args.z_tolerance,
        result_root=args.lasso_refits.resolve().parents[2],
    )
    mcp = evaluate_method(
        pd.read_csv(args.mcp_selections),
        pd.read_csv(args.mcp_refits),
        method="mcp",
        z_tolerance=args.z_tolerance,
        result_root=args.mcp_refits.resolve().parents[2],
    )
    records = pd.concat([lasso, mcp], ignore_index=True)
    paired_all = pair_methods(records)
    paired_converged = paired_all.loc[
        paired_all["converged_lasso"] & paired_all["converged_mcp"]
    ].copy()
    valid_names = set(paired_converged["data_name"])
    primary_records = records.loc[records["data_name"].isin(valid_names)].copy()
    summary = summarize_methods(primary_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "penalty_fit_metrics_all.csv", index=False)
    primary_records.to_csv(
        args.output_dir / "penalty_fit_metrics_converged_pairs.csv", index=False
    )
    paired_all.to_csv(args.output_dir / "penalty_pairs_all.csv", index=False)
    paired_converged.to_csv(
        args.output_dir / "penalty_pairs_converged.csv", index=False
    )
    summary.to_csv(args.output_dir / "penalty_summary_by_scenario.csv", index=False)
    print(
        f"Saved {len(paired_converged)} converged lasso/MCP pairs to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
