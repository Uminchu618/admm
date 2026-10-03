#!/usr/bin/env python3
"""5-fold CVで選択し全データ再学習した係数関数を真値と比較する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pilot.plot_bic_selected_beta import (
    LABELS,
    SCENARIOS,
    build_scenario_figure,
    load_truth,
    parse_scenarios,
    parse_seeds,
)
from scripts.pilot.visualize_cv_results import prepare_cv_selected_records


def select_cv_refits(
    selections: pd.DataFrame,
    refits: pd.DataFrame,
    scenarios: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    records = prepare_cv_selected_records(selections, refits)
    selected = records.loc[
        records["scenario"].isin(scenarios) & records["seed"].isin(seeds)
    ].copy()
    expected = {(scenario, seed) for scenario in scenarios for seed in seeds}
    observed = set(zip(selected["scenario"], selected["seed"]))
    if observed != expected:
        raise ValueError(f"missing CV-selected refits: {sorted(expected - observed)}")
    failed = selected.loc[~selected["converged"]]
    if not failed.empty:
        raise ValueError(
            "nonconverged CV-selected refits: "
            f"{failed['data_name'].tolist()}"
        )
    selected["lambda_fuse"] = selected["selected_lambda"]
    return selected.sort_values(["scenario", "seed"]).reset_index(drop=True)


def write_report(records: pd.DataFrame, scenarios: list[str], output_path: Path) -> None:
    lines = [
        "# 5-fold CV選択係数関数と真値の比較",
        "",
        "黒破線が生成時の真値、色付き実線が平均検証Ctd最大のlambdaで全データ再学習した推定値である。",
        "lambda選択には独立評価データを使用していない。Ctd(test)は再学習後の独立評価値である。",
        "",
    ]
    for scenario in scenarios:
        lines.extend(
            [
                f"## {LABELS[scenario]}",
                "",
                f"![{LABELS[scenario]}](cv_selected_beta_{scenario}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 選択結果",
            "",
            "| Scenario | Seed | lambda | Ctd(test) | RMISE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in records.itertuples(index=False):
        lines.append(
            f"| {LABELS[row.scenario]} | {row.seed} | {row.lambda_fuse:g} | "
            f"{row.c_td_test:.4f} | {row.coefficient_rmise:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-selections", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=parse_seeds, default=parse_seeds("42,43,44,45,46")
    )
    parser.add_argument(
        "--scenarios", type=parse_scenarios, default=parse_scenarios(",".join(SCENARIOS))
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = ROOT
    selected = select_cv_refits(
        pd.read_csv(args.cv_selections),
        pd.read_csv(args.refit_summary),
        args.scenarios,
        args.seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, object]] = []
    pdf_path = args.output_dir / "cv_selected_beta_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        for scenario in args.scenarios:
            true_grid, truth = load_truth(
                root / "generation" / "pilot" / f"{scenario}.json"
            )
            rows = selected.loc[selected["scenario"] == scenario]
            fig, records = build_scenario_figure(
                scenario,
                rows,
                root,
                true_grid,
                truth,
                selection_label="5-fold CV-selected estimate",
                title_selection_label="5-fold CV-selected",
            )
            fig.savefig(
                args.output_dir / f"cv_selected_beta_{scenario}.png",
                dpi=200,
                bbox_inches="tight",
                facecolor="white",
            )
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            all_records.extend(records)

    records = pd.DataFrame(all_records)
    records.to_csv(args.output_dir / "cv_selected_beta_records.csv", index=False)
    write_report(records, args.scenarios, args.output_dir / "cv_selected_beta_report.md")
    print(f"CV-selected fits: {len(records)}")


if __name__ == "__main__":
    main()
