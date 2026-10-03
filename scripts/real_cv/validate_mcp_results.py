#!/usr/bin/env python3
"""Check that an MCP real-data CV experiment has every requested fit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.common import lambda_label, load_lambda_values  # noqa: E402


def validate(
    base_dir: Path, grid_path: Path, n_folds: int, *, allow_missing: bool = False
) -> int:
    values = load_lambda_values(grid_path)
    if len(values) != len(set(values)) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid lambda grid")
    missing: list[str] = []
    invalid: list[str] = []
    expected_paths: set[Path] = set()
    for value in values:
        for fold in range(n_folds):
            path = base_dir / lambda_label(value) / f"fold_{fold:02d}" / "result.json"
            expected_paths.add(path)
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                config = payload["config"]
                if config.get("fuse_penalty") != "mcp" or not math.isclose(
                    float(config["lambda_fuse"]), value, rel_tol=0, abs_tol=1e-12
                ):
                    invalid.append(str(path))
            except (OSError, ValueError, KeyError, TypeError):
                invalid.append(str(path))
    unexpected = sorted(str(path) for path in base_dir.rglob("result.json") if path not in expected_paths)
    if (missing and not allow_missing) or invalid or unexpected:
        raise RuntimeError(
            f"MCP CV incomplete/invalid: {len(missing)} missing, "
            f"{len(invalid)} invalid, {len(unexpected)} unexpected\n"
            + "\n".join((missing + invalid + unexpected)[:10])
        )
    available = len(values) * n_folds - len(missing)
    if available == 0:
        raise RuntimeError(f"No MCP CV result.json files found in {base_dir}")
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--lambda-grid", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error("n-folds must be >= 2")
    count = validate(
        args.base_dir, args.lambda_grid, args.n_folds,
        allow_missing=args.allow_missing,
    )
    expected = len(load_lambda_values(args.lambda_grid)) * args.n_folds
    status = "complete" if count == expected else "partial"
    if args.status_only:
        print(status)
        print(f"MCP CV results: {count}/{expected} ({status})", file=sys.stderr)
        if status == "partial":
            for value in load_lambda_values(args.lambda_grid):
                for fold in range(args.n_folds):
                    path = args.base_dir / lambda_label(value) / f"fold_{fold:02d}" / "result.json"
                    if not path.is_file():
                        print(f"Pending: {path}", file=sys.stderr)
    else:
        print(f"Validated {count}/{expected} MCP CV results in {args.base_dir} ({status})")


if __name__ == "__main__":
    main()
