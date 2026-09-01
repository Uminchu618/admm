#!/usr/bin/env python3
"""実データ全体を使う推定用の long-format CSV と config を作る。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from admm.config import load_config  # noqa: E402
from scripts.real_cv.common import build_full_long_data, load_time_grid  # noqa: E402
from scripts.real_cv.datasets import get_dataset_spec, load_real_base  # noqa: E402


def _jsonable_config(config: dict[str, Any], lambda_fuse: float) -> dict[str, Any]:
    """main.py に渡す JSON config を作る。"""

    out = dict(config)
    out["lambda_fuse"] = float(lambda_fuse)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full real-data fit input")
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
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Base ADMM config.",
    )
    parser.add_argument("--lambda-fuse", type=float, required=True)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Optional selected_lambda.json used to choose lambda_fuse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory, e.g. outputs/real_full/support2/exp/lambda_x.",
    )
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    input_path = args.input or spec.default_input
    base = load_real_base(args.dataset, input_path)
    time_grid = load_time_grid(args.config)
    full_long, summary = build_full_long_data(
        base=base,
        time_grid=time_grid,
        spec=spec,
    )

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "all.csv"
    full_long.to_csv(data_path, index=False)

    data_meta_path = Path(f"{data_path}.meta.json")
    with data_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": spec.name,
                "time_grid": summary["time_grid"],
                "feature_cols": summary["feature_cols"],
                "time_scale_max_original": summary["time_scale_max_original"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    config = _jsonable_config(load_config(args.config), args.lambda_fuse)
    config_path = args.output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    summary.update(
        {
            "input_path": str(input_path),
            "base_config_path": str(args.config),
            "config_path": str(config_path),
            "lambda_fuse": float(args.lambda_fuse),
            "data_path": str(data_path),
            "data_meta_path": str(data_meta_path),
            "lambda_selection_file": (
                str(args.selection_file) if args.selection_file is not None else None
            ),
        }
    )
    if args.selection_file is not None:
        with args.selection_file.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        selected_lambda = float(selection["selected_lambda"])
        if not math.isclose(
            selected_lambda, args.lambda_fuse, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "lambda_fuse does not match selected_lambda in selection-file"
            )
        summary["lambda_selection"] = selection

    meta_path = args.output_dir / "full_data_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
