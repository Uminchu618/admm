#!/usr/bin/env python3
"""粗いCV選択値の周辺に局所21点gridと未計算task manifestを作る。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


N_POINTS_PER_SIDE = 10
N_LOCAL_GRID = 2 * N_POINTS_PER_SIDE + 1


def canonical_float(value: float) -> float:
    """出力ディレクトリ名と一致する15桁表現へ丸める。"""

    return float(f"{float(value):.15g}")


def lambda_dir_name(value: float) -> str:
    return f"lambda_{float(value):.15g}"


def _selected_index(coarse_grid: np.ndarray, selected: float) -> int:
    matches = np.flatnonzero(np.isclose(coarse_grid, selected, rtol=0.0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError(f"selected lambda {selected:g} is not unique in coarse grid")
    return int(matches[0])


def local_lambda_grid(
    selected: float,
    coarse_grid: list[float] | np.ndarray,
    *,
    upper_extension: float = 0.75,
) -> np.ndarray:
    """選択値と上下の粗い候補の間を各10分割した21点を返す。"""

    coarse = np.asarray(coarse_grid, dtype=float)
    if coarse.ndim != 1 or len(coarse) < 3 or np.any(np.diff(coarse) <= 0):
        raise ValueError("coarse_grid must be a strictly increasing one-dimensional grid")
    selected = float(selected)
    index = _selected_index(coarse, selected)

    if index == 0:
        # lambda=0は対数分割できない。次の2候補までを区間別に線形分割する。
        lower_half = np.linspace(0.0, coarse[1], N_POINTS_PER_SIDE + 1)
        upper_half = np.linspace(coarse[1], coarse[2], N_POINTS_PER_SIDE + 1)
        values = np.concatenate([lower_half, upper_half[1:]])
    else:
        lower = float(coarse[index - 1])
        upper = (
            float(coarse[index + 1])
            if index + 1 < len(coarse)
            else float(upper_extension)
        )
        if upper <= selected:
            raise ValueError("upper_extension must be greater than the largest coarse lambda")
        if lower == 0.0:
            lower_half = np.linspace(lower, selected, N_POINTS_PER_SIDE + 1)
            upper_half = np.linspace(selected, upper, N_POINTS_PER_SIDE + 1)
        else:
            lower_half = np.geomspace(lower, selected, N_POINTS_PER_SIDE + 1)
            upper_half = np.geomspace(selected, upper, N_POINTS_PER_SIDE + 1)
        values = np.concatenate([lower_half, upper_half[1:]])

    values = np.asarray([canonical_float(value) for value in values], dtype=float)
    if len(values) != N_LOCAL_GRID or np.any(np.diff(values) <= 0):
        raise ValueError(f"failed to construct {N_LOCAL_GRID} unique lambda values")
    if not np.isclose(values, selected, rtol=0.0, atol=1e-12).any():
        raise ValueError("local grid does not contain the coarse selected lambda")
    return values


def build_grid_table(
    selections: pd.DataFrame,
    coarse_grid: list[float],
    *,
    upper_extension: float = 0.75,
) -> pd.DataFrame:
    required = {"data_name", "selected_lambda"}
    missing = sorted(required - set(selections.columns))
    if missing:
        raise ValueError(f"selection table is missing columns: {missing}")
    if selections["data_name"].duplicated().any():
        raise ValueError("selection table must contain one row per data_name")

    rows: list[dict[str, object]] = []
    for selection in selections.sort_values("data_name").itertuples(index=False):
        selected = float(selection.selected_lambda)
        values = local_lambda_grid(
            selected, coarse_grid, upper_extension=upper_extension
        )
        for grid_index, value in enumerate(values):
            rows.append(
                {
                    "data_name": selection.data_name,
                    "coarse_selected_lambda": selected,
                    "grid_index": grid_index,
                    "lambda_fuse": value,
                    "is_coarse_selected": bool(
                        np.isclose(value, selected, rtol=0.0, atol=1e-12)
                    ),
                    "is_local_boundary": grid_index in {0, len(values) - 1},
                }
            )
    return pd.DataFrame(rows)


def result_path(base_dir: Path, data_name: str, lambda_fuse: float, fold: int) -> Path:
    return (
        base_dir
        / data_name
        / lambda_dir_name(lambda_fuse)
        / f"fold_{fold:02d}"
        / "result.json"
    )


def result_is_reusable(path: Path) -> bool:
    """収束判定を満たし、有限な検証Ctdを持つ結果だけを再利用する。"""

    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        history = payload.get("history", {})
        converged = summary.get("converged", history.get("converged", False))
        if isinstance(converged, str):
            converged = converged.strip().lower() in {"true", "1", "yes"}
        c_td_test = summary.get(
            "c_td_test", summary.get("c_td_eval", summary.get("c_td"))
        )
        return bool(converged) and math.isfinite(float(c_td_test))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_task_manifest(
    grid_table: pd.DataFrame,
    *,
    n_folds: int,
    existing_base_dirs: list[Path],
    output_base_dir: Path,
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    rows: list[dict[str, object]] = []
    for grid_row in grid_table.itertuples(index=False):
        for fold in range(n_folds):
            existing = next(
                (
                    path
                    for base_dir in existing_base_dirs
                    if result_is_reusable(path := result_path(
                        base_dir,
                        grid_row.data_name,
                        grid_row.lambda_fuse,
                        fold,
                    ))
                ),
                None,
            )
            output_path = result_path(
                output_base_dir,
                grid_row.data_name,
                grid_row.lambda_fuse,
                fold,
            )
            if existing is not None or result_is_reusable(output_path):
                continue
            rows.append(
                {
                    "task_id": len(rows) + 1,
                    "data_name": grid_row.data_name,
                    "coarse_selected_lambda": grid_row.coarse_selected_lambda,
                    "grid_index": grid_row.grid_index,
                    "lambda_fuse": grid_row.lambda_fuse,
                    "fold": fold,
                    "output_path": str(output_path),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "task_id",
            "data_name",
            "coarse_selected_lambda",
            "grid_index",
            "lambda_fuse",
            "fold",
            "output_path",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-selections", type=Path, required=True)
    parser.add_argument("--coarse-grid", type=Path, required=True)
    parser.add_argument(
        "--existing-base-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-base-dir", type=Path, required=True)
    parser.add_argument("--grid-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--data-names",
        type=str,
        default=None,
        help="Comma-separated data_name subset for a smoke run.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--upper-extension", type=float, default=0.75)
    args = parser.parse_args()

    selections = pd.read_csv(args.cv_selections)
    if args.data_names:
        requested = [value.strip() for value in args.data_names.split(",") if value.strip()]
        selections = selections.loc[selections["data_name"].isin(requested)].copy()
        missing_names = sorted(set(requested) - set(selections["data_name"]))
        if missing_names:
            raise ValueError(f"unknown data names: {missing_names}")
    coarse_payload = json.loads(args.coarse_grid.read_text(encoding="utf-8"))
    coarse_grid = [float(value) for value in coarse_payload["lambda_values"]]
    grid_table = build_grid_table(
        selections, coarse_grid, upper_extension=args.upper_extension
    )
    manifest = build_task_manifest(
        grid_table,
        n_folds=args.n_folds,
        existing_base_dirs=args.existing_base_dir,
        output_base_dir=args.output_base_dir,
    )

    args.grid_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    grid_table.to_csv(args.grid_output, index=False)
    manifest.to_csv(args.manifest_output, index=False)

    summary = {
        "datasets": int(grid_table["data_name"].nunique()),
        "lambda_values_per_dataset": N_LOCAL_GRID,
        "candidate_fold_combinations": int(len(grid_table) * args.n_folds),
        "tasks_to_run": int(len(manifest)),
        "reused_or_already_completed": int(
            len(grid_table) * args.n_folds - len(manifest)
        ),
        "n_folds": args.n_folds,
        "upper_extension": args.upper_extension,
    }
    summary_output = args.summary_output or args.manifest_output.with_suffix(".json")
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
