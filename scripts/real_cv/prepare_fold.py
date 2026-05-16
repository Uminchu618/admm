#!/usr/bin/env python3
"""実データ CV の 1 fold 分の train/test CSV と config を作る。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from admm.config import load_config  # noqa: E402
from scripts.real_cv.common import (  # noqa: E402
    build_fold_long_data,
    load_time_grid,
)
from scripts.real_cv.datasets import get_dataset_spec, load_real_base  # noqa: E402


def _jsonable_config(config: dict[str, Any], lambda_fuse: float) -> dict[str, Any]:
    """main.py に渡す JSON config を作る。"""

    out = dict(config)
    out["lambda_fuse"] = float(lambda_fuse)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one real-data CV fold")
    parser.add_argument(
        "--dataset",
        type=str,
        default="support2",
        choices=["support2", "framingham"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw CSV. Defaults to the dataset-specific raw file.",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/real/cv/splits/support2/support2_5fold_seed1234.csv"),
        help="Path to fold assignments CSV made by make_splits.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Base ADMM config.",
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lambda-fuse", type=float, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory, e.g. outputs/real_cv/support2/exp/lambda_x/fold_00.",
    )
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    input_path = args.input or spec.default_input
    base = load_real_base(args.dataset, input_path)
    assignments = pd.read_csv(args.splits)
    time_grid = load_time_grid(args.config)
    train_long, test_long, summary = build_fold_long_data(
        base=base,
        assignments=assignments,
        fold=args.fold,
        time_grid=time_grid,
        spec=spec,
    )

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_long.to_csv(train_path, index=False)
    test_long.to_csv(test_path, index=False)

    config = _jsonable_config(load_config(args.config), args.lambda_fuse)
    config_path = args.output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    summary.update(
        {
            "dataset": spec.name,
            "input_path": str(input_path),
            "splits_path": str(args.splits),
            "base_config_path": str(args.config),
            "config_path": str(config_path),
            "lambda_fuse": float(args.lambda_fuse),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
    )

    meta_path = args.output_dir / "fold_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
