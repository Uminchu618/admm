#!/usr/bin/env python3
"""long-format シミュレーションデータを被験者単位で CV 分割する。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.real_cv.common import make_fold_assignments  # noqa: E402


def split_long_data(
    data: pd.DataFrame,
    *,
    fold: int,
    n_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """各 id の全区間行を同じ fold に保ったまま train/test に分ける。"""

    required = {"id", "k", "time", "event"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}")

    per_subject_counts = data.groupby("id").size()
    if per_subject_counts.nunique() != 1:
        raise ValueError("every id must have the same number of long-format rows")
    for column in ["time", "event"]:
        if data.groupby("id")[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"{column} must be constant within id")

    subject = (
        data.groupby("id", as_index=False)
        .agg(event_original=("event", "first"))
        .sort_values("id")
        .reset_index(drop=True)
    )
    assignments = make_fold_assignments(
        subject,
        n_folds=n_folds,
        random_state=random_state,
        stratify_event=True,
    )
    test_ids = set(assignments.loc[assignments["fold"] == fold, "id"].tolist())
    is_test = data["id"].isin(test_ids)
    train = data.loc[~is_test].copy().reset_index(drop=True)
    test = data.loc[is_test].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("train or test fold is empty")

    train_ids = set(train["id"].unique().tolist())
    test_ids_observed = set(test["id"].unique().tolist())
    if train_ids & test_ids_observed:
        raise AssertionError("subject leakage detected between train and test")

    summary: dict[str, object] = {
        "fold": int(fold),
        "n_folds": int(n_folds),
        "random_state": int(random_state),
        "stratify_event": True,
        "n_train": len(train_ids),
        "n_test": len(test_ids_observed),
        "n_train_events": int(train.groupby("id")["event"].first().sum()),
        "n_test_events": int(test.groupby("id")["event"].first().sum()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    return train, test, summary


def _copy_metadata(source_data: Path, target_data: Path) -> None:
    source_meta = Path(f"{source_data}.meta.json")
    if source_meta.exists():
        shutil.copyfile(source_meta, Path(f"{target_data}.meta.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=1234)
    args = parser.parse_args()

    train, test, summary = split_long_data(
        pd.read_csv(args.data),
        fold=args.fold,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    _copy_metadata(args.data, train_path)
    _copy_metadata(args.data, test_path)

    summary.update(
        {
            "source_data": str(args.data),
            "train_data": str(train_path),
            "test_data": str(test_path),
        }
    )
    meta_path = args.output_dir / "fold_meta.json"
    meta_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
