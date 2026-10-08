#!/usr/bin/env python3
"""FraminghamのK × lambda CVを準備・実行・集計する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from admm.config import load_config
from admm.model import ADMMHazardAFT
from scripts.real_cv.aggregate_results import summarize_by_lambda
from scripts.real_cv.common import (
    build_fold_subject_data, build_full_long_data, fold_label, lambda_label,
)
from scripts.real_cv.datasets import get_dataset_spec, load_real_base


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_grid(grid: dict) -> dict:
    ks = grid["k_values"]
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("k_values must be distinct positive integers")
    lambdas = [float(v) for v in grid["lambda_values"]]
    if not lambdas or any(not math.isfinite(v) or v < 0 for v in lambdas):
        raise ValueError("lambda_values must be finite and nonnegative")
    if len(set(lambdas)) != len(lambdas) or len({lambda_label(v) for v in lambdas}) != len(lambdas):
        raise ValueError("lambda_values must have distinct values and directory labels")
    span = np.asarray(grid["time_range"], dtype=float)
    if span.shape != (2,) or not np.isfinite(span).all() or span[0] != 0 or span[1] <= 0:
        raise ValueError("time_range must be [0, positive finite end]")
    return dict(k_values=ks, lambda_values=lambdas, time_range=span.tolist())


def task_config(manifest: dict, task: dict) -> dict:
    config = dict(manifest["base_config"])
    config["time_grid"] = np.linspace(*manifest["grid"]["time_range"], task["K"] + 1).tolist()
    config["lambda_fuse"] = task["lambda_fuse"]
    return config


def prepare(args) -> None:
    grid = validate_grid(read_json(args.grid))
    config = load_config(args.config)
    # Estimatorで設定キーを検証する。Kごとのtime_gridは実行時に生成する。
    ADMMHazardAFT.from_config(config)
    if config.get("random_state") is None:
        raise ValueError("Set random_state in the base config for reproducible comparisons")
    if args.n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    spec = get_dataset_spec("framingham")
    base = load_real_base("framingham", args.input)
    assignments = pd.read_csv(args.splits)
    if assignments["id"].duplicated().any() or set(assignments["id"]) != set(base["id"]):
        raise ValueError("splits must assign every complete-case id exactly once")
    if set(assignments["fold"]) != set(range(args.n_folds)):
        raise ValueError("splits must contain folds 0..n_folds-1")
    split_meta = Path(f"{args.splits}.meta.json")
    if split_meta.exists():
        meta = read_json(split_meta)
        if meta.get("dataset") != "framingham" or meta.get("n_folds") != args.n_folds:
            raise ValueError("split metadata does not match Framingham/fold count")

    # 書き込み前に全foldの前処理を検証する。
    folds = [build_fold_subject_data(base, assignments, f, np.asarray(grid["time_range"]), spec)
             for f in range(args.n_folds)]
    full, full_meta = build_full_long_data(base, np.asarray(grid["time_range"]), spec)
    args.search_dir.mkdir(parents=True, exist_ok=False)
    shared = args.search_dir / "prepared"
    shared.mkdir()
    hashes = {}
    fold_summaries = {}
    for fold, (train, test, meta) in enumerate(folds):
        folder = shared / fold_label(fold)
        folder.mkdir()
        for name, frame in (("train", train), ("test", test)):
            path = folder / f"{name}.csv"
            frame.to_csv(path, index=False)
            hashes[str(path.relative_to(args.search_dir))] = sha256(path)
        write_json(folder / "fold_meta.json", meta)
        fold_summaries[str(fold)] = meta
    full_path = shared / "all.csv"
    full.drop(columns="k").to_csv(full_path, index=False)
    hashes[str(full_path.relative_to(args.search_dir))] = sha256(full_path)
    write_json(shared / "full_meta.json", full_meta)
    tasks = []
    for k in grid["k_values"]:
        for value in grid["lambda_values"]:
            for fold in range(args.n_folds):
                tasks.append(dict(task_id=len(tasks) + 1, K=k, lambda_fuse=value, fold=fold,
                                  directory=f"K_{k:02d}/{lambda_label(value)}/{fold_label(fold)}"))
    manifest = dict(schema_version=1, dataset="framingham", n_folds=args.n_folds,
                    grid=grid, base_config=config, tasks=tasks, data_hashes=hashes,
                    fold_summaries=fold_summaries, n_samples=len(base),
                    feature_cols=spec.feature_cols,
                    sources={name: dict(path=str(path), sha256=sha256(path))
                             for name, path in (("input", args.input), ("splits", args.splits),
                                                ("config", args.config), ("grid", args.grid))})
    write_json(args.search_dir / "manifest.json", manifest)
    print(f"Prepared {len(tasks)} tasks; shared {args.n_folds} train/test pairs: {args.search_dir}")


def verify_data(directory: Path, manifest: dict, relative: str) -> Path:
    path = directory / relative
    if sha256(path) != manifest["data_hashes"][relative]:
        raise ValueError(f"Prepared data changed: {path}")
    return path


def validate_result(result: dict, manifest: dict, task: dict) -> None:
    expected = task_config(manifest, task)
    meta = manifest["fold_summaries"][str(task["fold"])]
    if result.get("config") != expected or result.get("time_grid") != expected["time_grid"]:
        raise ValueError(f"Result config/time_grid mismatch: task {task['task_id']}")
    if (result.get("n_samples") != meta["n_train"]
            or result.get("n_eval_samples") != meta["n_test"]
            or result.get("feature_cols") != manifest["feature_cols"]
            or not result.get("eval_data_path")):
        raise ValueError(f"Result train/test metadata mismatch: task {task['task_id']}")
    if not isinstance(result.get("summary", {}).get("converged"), bool):
        raise ValueError("Result must record boolean summary.converged")


def run_fit(directory: Path, config: dict, train: Path, test: Path | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # 同じtaskの二重起動を防ぐ。中断後のlockは手動確認の対象。
    lock = directory / ".running"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    try:
        config_path = directory / "config.json"
        write_json(config_path, config)
        command = [sys.executable, str(ROOT / "main.py"), "--config", str(config_path),
                   "--data", str(train), "--output", str(directory / "result.partial.json")]
        if test is not None:
            command += ["--eval-data", str(test)]
        started = time.monotonic()
        subprocess.run(command, cwd=ROOT, check=True)
        (directory / "result.partial.json").replace(directory / "result.json")
        write_json(directory / "runtime.json", {"elapsed_seconds": time.monotonic() - started})
    finally:
        lock.unlink()


def run_task(args) -> None:
    directory = args.search_dir.resolve()
    manifest = read_json(directory / "manifest.json")
    if not 1 <= args.task_id <= len(manifest["tasks"]):
        raise ValueError(f"task_id must be 1..{len(manifest['tasks'])}")
    task = manifest["tasks"][args.task_id - 1]
    train = verify_data(directory, manifest, f"prepared/{fold_label(task['fold'])}/train.csv")
    test = verify_data(directory, manifest, f"prepared/{fold_label(task['fold'])}/test.csv")
    output = directory / task["directory"]
    if (output / "result.json").exists():
        validate_result(read_json(output / "result.json"), manifest, task)
        if args.skip_existing:
            print(f"Skip completed task {args.task_id}")
            return
        raise FileExistsError(f"Result already exists: {output}; use --skip-existing")
    print(f"Task {args.task_id}: K={task['K']}, lambda={task['lambda_fuse']}, fold={task['fold']}")
    run_fit(output, task_config(manifest, task), train, test)
    validate_result(read_json(output / "result.json"), manifest, task)


def select_pair(summary: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie tolerance must be finite and nonnegative")
    summary = summary.copy()
    summary["c_td_test_mean"] = pd.to_numeric(summary["c_td_test_mean"], errors="coerce")
    summary["selected"] = False
    eligible = summary.loc[summary["cv_eligible"] & np.isfinite(summary["c_td_test_mean"])]
    if not eligible.empty:
        best = eligible["c_td_test_mean"].max()
        tied = eligible.loc[(eligible["c_td_test_mean"] - best).abs() <= tolerance]
        winner = tied.sort_values(["K", "lambda_fuse"], ascending=[True, False]).index[0]
        summary.loc[winner, "selected"] = True
    return summary


def aggregate(args) -> dict:
    directory = args.search_dir.resolve()
    manifest = read_json(directory / "manifest.json")
    for relative in manifest["data_hashes"]:
        verify_data(directory, manifest, relative)
    expected_paths = {directory / t["directory"] / "result.json" for t in manifest["tasks"]}
    unexpected = set(directory.glob("K_*/**/result.json")) - expected_paths
    if unexpected:
        raise ValueError(f"Unexpected result paths: {sorted(unexpected)}")
    rows = []
    for task in manifest["tasks"]:
        path = directory / task["directory"] / "result.json"
        if not path.exists():
            continue
        result = read_json(path)
        validate_result(result, manifest, task)
        metrics = result["summary"]
        rows.append(dict(K=task["K"], lambda_fuse=task["lambda_fuse"], fold=task["fold"],
                         c_td_test=metrics.get("c_td_test"), c_td_train=metrics.get("c_td_train"),
                         converged=metrics["converged"], stopping_reason=metrics.get("stopping_reason"),
                         n_admm_iter=metrics.get("n_admm_iter"),
                         objective_last=metrics.get("objective_last"),
                         primal_residual_last=metrics.get("primal_residual_last"),
                         dual_residual_last=metrics.get("dual_residual_last"), result_path=str(path)))
    fold_df = pd.DataFrame(rows, columns=["K", "lambda_fuse", "fold", "c_td_test", "c_td_train",
                             "converged", "stopping_reason", "n_admm_iter", "objective_last",
                             "primal_residual_last", "dual_residual_last", "result_path"])
    summaries = []
    for k in manifest["grid"]["k_values"]:
        subset = fold_df.loc[fold_df["K"] == k]
        summary = summarize_by_lambda(subset, expected_n_folds=manifest["n_folds"])
        if summary.empty:
            summary = pd.DataFrame(columns=["lambda_fuse", "cv_eligible", "c_td_test_mean"])
        summary = summary.set_index("lambda_fuse").reindex(manifest["grid"]["lambda_values"])
        summary["cv_eligible"] = summary["cv_eligible"].astype("boolean").fillna(False).astype(bool)
        for column in ("n_results", "n_folds", "n_converged_folds", "n_finite_c_td_folds"):
            if column not in summary:
                summary[column] = 0
            summary[column] = summary[column].fillna(0).astype(int)
        if "cv_exclusion_reason" not in summary:
            summary["cv_exclusion_reason"] = "no_results"
        else:
            summary["cv_exclusion_reason"] = summary["cv_exclusion_reason"].fillna("no_results")
        summary["n_folds_expected"] = manifest["n_folds"]
        summary["K"] = k
        summaries.append(summary.reset_index())
    summary = select_pair(pd.DataFrame([row for frame in summaries for row in frame.to_dict("records")]),
                          args.tie_tolerance)
    complete = len(rows) == len(manifest["tasks"])
    summary["provisional_selected"] = summary["selected"]
    if not complete:
        summary["selected"] = False
    selected = summary.loc[summary["provisional_selected"]]
    winner = selected.iloc[0] if len(selected) == 1 else None
    payload = dict(selection_method="joint_cv_mean_c_td" if complete and winner is not None else
                   ("no_eligible_candidate" if complete else "pending_cv_completion"),
                   selected_K=int(winner["K"]) if complete and winner is not None else None,
                   selected_lambda=float(winner["lambda_fuse"]) if complete and winner is not None else None,
                   provisional_K=int(winner["K"]) if winner is not None else None,
                   provisional_lambda=float(winner["lambda_fuse"]) if winner is not None else None,
                   mean_c_td=float(winner["c_td_test_mean"]) if winner is not None else None,
                   std_c_td=float(winner["c_td_test_std"]) if winner is not None else None,
                   n_folds=manifest["n_folds"], n_results_available=len(rows),
                   n_results_expected=len(manifest["tasks"]),
                   time_range=manifest["grid"]["time_range"],
                   tie_break="smallest_K_then_largest_lambda_within_tolerance",
                   tie_tolerance=args.tie_tolerance, manifest_sha256=sha256(directory / "manifest.json"))
    if payload["selected_K"] is not None:
        payload["time_grid"] = np.linspace(*payload["time_range"], payload["selected_K"] + 1).tolist()
    fold_df.to_csv(directory / "fold_results.csv", index=False)
    summary.to_csv(directory / "summary_by_K_lambda.csv", index=False)
    write_json(directory / "selected_params.json", payload)
    plot_summary(summary, directory)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def plot_summary(summary: pd.DataFrame, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, data in summary.groupby("K"):
        data = data.loc[data["cv_eligible"]].sort_values("lambda_fuse")
        if not data.empty:
            ax.errorbar(data["lambda_fuse"], data["c_td_test_mean"],
                        yerr=data["c_td_test_se"], marker="o", label=f"K={k}")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set(xlabel="lambda", ylabel="Mean validation c_td",
           title="Framingham: complete, converged CV candidates")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "K_lambda_vs_c_td.png", dpi=160)
    plt.close(fig)


def refit(args) -> None:
    # 選択を現在の結果から再確認し、暫定選択を全データfitへ渡さない。
    selection = aggregate(args)
    if selection["selection_method"] != "joint_cv_mean_c_td":
        raise ValueError("Refit requires completed CV and an eligible selected pair")
    directory = args.search_dir.resolve()
    manifest = read_json(directory / "manifest.json")
    config = task_config(manifest, dict(K=selection["selected_K"], lambda_fuse=selection["selected_lambda"]))
    train = verify_data(directory, manifest, "prepared/all.csv")
    output = directory / "selected_full"
    if (output / "result.json").exists():
        raise FileExistsError(f"Full refit already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_fit(output, config, train, None)
    write_json(output / "selection.json", selection)


def submit(args) -> None:
    directory = args.search_dir.resolve()
    manifest = read_json(directory / "manifest.json")
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    uv = os.environ.get("UV_BIN", "/home/sagara/.local/bin/uv")
    if any(c in str(directory) + uv for c in ",\n"):
        raise ValueError("qsub environment paths must not contain commas or newlines")
    command = ["qsub", "-v", f"SEARCH_DIR={directory},UV_BIN={uv}"]
    if args.refit:
        selection = aggregate(args)
        if selection["selection_method"] != "joint_cv_mean_c_td":
            raise ValueError("Cannot submit refit before final selection")
        script = ROOT / "qsub_real_klambda_refit.sh"
    else:
        command += ["-t", f"1-{len(manifest['tasks'])}", "-tc", str(args.max_concurrent)]
        script = ROOT / "qsub_real_klambda_cv.sh"
    subprocess.run(command + [str(script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "aggregate", "refit", "submit"):
        p = sub.add_parser(action)
        p.add_argument("--search-dir", type=Path, required=True)
        if action == "prepare":
            p.add_argument("--input", type=Path, default=ROOT / "data/real/framingham/framingham.csv")
            p.add_argument("--splits", type=Path, default=ROOT / "data/real/cv/splits/framingham/framingham_5fold_seed1234.csv")
            p.add_argument("--config", type=Path, default=ROOT / "config.toml")
            p.add_argument("--grid", type=Path, default=ROOT / "framingham_klambda_grid.json")
            p.add_argument("--n-folds", type=int, default=5)
        if action == "run":
            p.add_argument("--task-id", type=int, default=int(os.environ.get("SGE_TASK_ID", "1")))
            p.add_argument("--skip-existing", action="store_true")
        if action in ("aggregate", "refit", "submit"):
            p.add_argument("--tie-tolerance", type=float, default=1e-12)
        if action == "submit":
            p.add_argument("--max-concurrent", type=int, default=50)
            p.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": run_task, "aggregate": aggregate, "refit": refit, "submit": submit}[args.action](args)


if __name__ == "__main__":
    main()
