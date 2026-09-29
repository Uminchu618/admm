#!/usr/bin/env python3
"""Render fold-wise and selected MCP coefficient trajectories for one real dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.common import lambda_label  # noqa: E402
from scripts.visualize_real_beta import (  # noqa: E402
    plot_beta_trajectories,
    plot_cv_beta_trajectories_by_lambda,
)


def selected_lambda(base_dir: Path) -> float:
    selection = json.loads((base_dir / "selected_lambda.json").read_text(encoding="utf-8"))
    if selection.get("selection_method") != "five_fold_cv_mean_c_td":
        raise ValueError("Unexpected selected lambda method")
    return float(selection["selected_lambda"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["framingham", "support2"], required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()

    value = selected_lambda(args.base_dir)
    plot_dir = args.base_dir / "plots"
    if not args.full_only:
        plot_cv_beta_trajectories_by_lambda(
            args.dataset, args.base_dir, plot_dir / "beta_by_lambda"
        )
        selected_plots = plot_cv_beta_trajectories_by_lambda(
            args.dataset, args.base_dir, plot_dir / "selected_beta", {value}
        )
        print(f"Selected-fold trajectories: {selected_plots[0]}")

    if args.full_dir is not None:
        full_result = args.full_dir / lambda_label(value) / "result.json"
        if not full_result.is_file():
            raise FileNotFoundError(full_result)
        output = plot_beta_trajectories(
            full_result, plot_dir / "selected_lambda_full_beta.png"
        )
        print(f"Full-data trajectories: {output}")


if __name__ == "__main__":
    main()
