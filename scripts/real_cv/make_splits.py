#!/usr/bin/env python3
"""実データ CV 用の id 単位 fold 割当を作る。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.common import make_fold_assignments  # noqa: E402
from scripts.real_cv.datasets import get_dataset_spec, load_real_base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create real-data CV splits")
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
        "--output",
        type=Path,
        default=None,
        help="Path to write fold assignments CSV.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=1234)
    parser.add_argument(
        "--no-stratify-event",
        action="store_true",
        help="Disable event-stratified fold assignment.",
    )
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    input_path = args.input or spec.default_input
    output_path = args.output or Path(
        f"data/real/cv/splits/{spec.name}/{spec.name}_{args.n_folds}fold_seed{args.random_state}.csv"
    )
    base = load_real_base(args.dataset, input_path)
    assignments = make_fold_assignments(
        base=base,
        n_folds=args.n_folds,
        random_state=args.random_state,
        stratify_event=not args.no_stratify_event,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_path, index=False)

    fold_counts = (
        assignments.groupby("fold")
        .agg(n_subjects=("id", "size"), n_events=("event_original", "sum"))
        .reset_index()
    )
    summary = {
        "dataset": spec.name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "n_folds": int(args.n_folds),
        "random_state": int(args.random_state),
        "stratify_event": not args.no_stratify_event,
        "n_subjects": int(assignments.shape[0]),
        "n_events": int(assignments["event_original"].sum()),
        "fold_counts": fold_counts.to_dict(orient="records"),
    }

    meta_path = Path(f"{output_path}.meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
