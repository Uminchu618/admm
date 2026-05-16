"""ADMMHazardAFT のパラメータ推定値に対するブートストラップ信頼区間。

long-format CSV（main.py と同じ形式）を読み込み、被験者単位のリサンプリングで
ADMMHazardAFT を繰り返し推定する。出力は β(coef) と γ(baseline) の
percentile 信頼区間で、JSON と CSV の両方に保存できる。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admm.config import load_config
from admm.model import ADMMHazardAFT

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm は任意依存だが pyproject には含めている
    tqdm = None


@dataclass
class LongFormatDataset:
    """ブートストラップ用に整形済みの long-format データ。"""

    X: np.ndarray
    y: np.ndarray
    feature_cols: list[str]
    k_count: int
    n_subjects: int


@dataclass
class BootstrapFitResult:
    """1 回のブートストラップ推定結果。"""

    replicate: int
    seed: int
    coef: Optional[list[list[float]]]
    gamma: Optional[list[float]]
    score: Optional[float]
    history_summary: Optional[dict[str, Optional[float]]]
    error: Optional[str]


def _load_long_format_dataset(data_path: Path) -> LongFormatDataset:
    """main.py と同じ long-format CSV を NumPy 配列へ変換する。"""

    data = pd.read_csv(data_path)
    required_cols = {"id", "k", "time", "event"}
    if not required_cols.issubset(data.columns):
        missing = sorted(required_cols - set(data.columns))
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    feature_cols = [
        col
        for col in data.columns
        if col
        not in {
            "id",
            "k",
            "time",
            "event",
            "time_true",
            "c1",
            "c2",
        }
    ]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {data_path}")

    data_sorted = data.sort_values(["id", "k"]).reset_index(drop=True)
    k_values = data_sorted["k"].to_numpy()
    if k_values.size == 0:
        raise ValueError(f"Empty dataset: {data_path}")
    if k_values.min() < 0:
        raise ValueError("k must be non-negative in long format")

    k_count = int(k_values.max()) + 1
    group_sizes = data_sorted.groupby("id", sort=True)["k"].size().to_numpy()
    if not np.all(group_sizes == k_count):
        raise ValueError("Each id must have exactly K rows in long format")

    expected_k = np.tile(np.arange(k_count), group_sizes.size)
    if not np.array_equal(k_values, expected_k):
        raise ValueError("k must be 0..K-1 in order for each id")

    X = (
        data_sorted[feature_cols]
        .to_numpy(dtype=float)
        .reshape(group_sizes.size, k_count, len(feature_cols))
    )
    y_rows = data_sorted.iloc[::k_count]
    y = y_rows[["time", "event"]].to_numpy(dtype=float)

    return LongFormatDataset(
        X=X,
        y=y,
        feature_cols=feature_cols,
        k_count=k_count,
        n_subjects=int(group_sizes.size),
    )


def _load_config_with_data_meta(config_path: Path, data_path: Path) -> dict[str, Any]:
    """config を読み込み、CSV 横の meta.json に time_grid があれば優先する。"""

    config = load_config(config_path)
    meta_path = Path(f"{data_path}.meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if "time_grid" in meta:
            config["time_grid"] = meta["time_grid"]
    return config


def _history_last(
    history: dict[str, Any],
    key: str,
) -> Optional[float]:
    """history[key] の最後の値を JSON 化しやすい float/None にする。"""

    values = history.get(key)
    if not values:
        return None
    value = float(values[-1])
    return value if np.isfinite(value) else None


def _fit_once(
    config: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[ADMMHazardAFT, Optional[float]]:
    """ADMMHazardAFT を 1 回 fit し、可能なら score も返す。"""

    model = ADMMHazardAFT.from_config(config)
    model.fit(X, y)
    try:
        score = float(model.score(X, y))
        if not np.isfinite(score):
            score = None
    except Exception:
        score = None
    return model, score


def _fit_bootstrap_replicate(
    args: tuple[int, int, dict[str, Any], np.ndarray, np.ndarray],
) -> BootstrapFitResult:
    """ProcessPoolExecutor から呼ぶための top-level worker。"""

    replicate, seed, config, X, y = args
    try:
        rng = np.random.default_rng(seed)
        n_samples = int(X.shape[0])
        indices = rng.integers(0, n_samples, size=n_samples)
        model, score = _fit_once(config, X[indices], y[indices])
        history = getattr(model, "history_", {})
        history_summary = {
            "objective_last": _history_last(history, "objective"),
            "neg_loglik_last": _history_last(history, "neg_loglik"),
            "primal_residual_last": _history_last(history, "primal_residual"),
            "dual_residual_last": _history_last(history, "dual_residual"),
        }
        return BootstrapFitResult(
            replicate=replicate,
            seed=seed,
            coef=np.asarray(model.coef_, dtype=float).tolist(),
            gamma=np.asarray(model.gamma_, dtype=float).reshape(-1).tolist(),
            score=score,
            history_summary=history_summary,
            error=None,
        )
    except Exception as exc:  # ブートストラップでは一部失敗を集計して継続する
        return BootstrapFitResult(
            replicate=replicate,
            seed=seed,
            coef=None,
            gamma=None,
            score=None,
            history_summary=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _iter_with_progress(
    iterable: Iterable[BootstrapFitResult],
    total: int,
) -> Iterable[BootstrapFitResult]:
    """tqdm があれば進捗を出し、なければ素通しする。"""

    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc="bootstrap")


def _finite_or_none(value: float) -> Optional[float]:
    """JSON に NaN/inf を混ぜないための変換。"""

    value_float = float(value)
    return value_float if np.isfinite(value_float) else None


def _std(values: np.ndarray, axis: int) -> np.ndarray:
    """成功回数が 1 のときも扱いやすい標準偏差。"""

    if values.shape[axis] <= 1:
        return np.zeros(values.shape[1:], dtype=float)
    return np.std(values, axis=axis, ddof=1)


def _coef_ci_records(
    point: np.ndarray,
    boot: np.ndarray,
    time_grid: Sequence[float],
    feature_cols: Sequence[str],
    ci_level: float,
) -> list[dict[str, Any]]:
    """β の推定値・ブートストラップ平均・信頼区間を long table にする。"""

    alpha = (1.0 - ci_level) / 2.0
    lower = np.quantile(boot, alpha, axis=0)
    upper = np.quantile(boot, 1.0 - alpha, axis=0)
    mean = np.mean(boot, axis=0)
    std = _std(boot, axis=0)

    records: list[dict[str, Any]] = []
    for k in range(point.shape[0]):
        interval = f"[{float(time_grid[k])}, {float(time_grid[k + 1])})"
        for j, feature in enumerate(feature_cols):
            records.append(
                {
                    "parameter": f"beta[{k},{feature}]",
                    "kind": "coef",
                    "interval_index": k,
                    "interval": interval,
                    "feature": feature,
                    "estimate": _finite_or_none(point[k, j]),
                    "bootstrap_mean": _finite_or_none(mean[k, j]),
                    "bootstrap_std": _finite_or_none(std[k, j]),
                    "ci_lower": _finite_or_none(lower[k, j]),
                    "ci_upper": _finite_or_none(upper[k, j]),
                    "ci_level": ci_level,
                }
            )
    return records


def _gamma_ci_records(
    point: np.ndarray,
    boot: np.ndarray,
    ci_level: float,
) -> list[dict[str, Any]]:
    """γ の推定値・ブートストラップ平均・信頼区間を long table にする。"""

    alpha = (1.0 - ci_level) / 2.0
    lower = np.quantile(boot, alpha, axis=0)
    upper = np.quantile(boot, 1.0 - alpha, axis=0)
    mean = np.mean(boot, axis=0)
    std = _std(boot, axis=0)

    records: list[dict[str, Any]] = []
    for m in range(point.shape[0]):
        records.append(
            {
                "parameter": f"gamma[{m}]",
                "kind": "gamma",
                "basis_index": m,
                "estimate": _finite_or_none(point[m]),
                "bootstrap_mean": _finite_or_none(mean[m]),
                "bootstrap_std": _finite_or_none(std[m]),
                "ci_lower": _finite_or_none(lower[m]),
                "ci_upper": _finite_or_none(upper[m]),
                "ci_level": ci_level,
            }
        )
    return records


def _read_base_result(
    base_result_path: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """既存 result.json から点推定値を読み、config も必要なら更新する。"""

    with base_result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if "coef" not in result or "gamma" not in result:
        raise ValueError("--base-result には coef と gamma が必要です。")

    config_out = dict(config)
    if isinstance(result.get("config"), dict):
        config_out.update(result["config"])
    if "time_grid" in result:
        config_out["time_grid"] = result["time_grid"]

    coef = np.asarray(result["coef"], dtype=float)
    gamma = np.asarray(result["gamma"], dtype=float).reshape(-1)
    return coef, gamma, config_out


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """records を CSV に保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _default_csv_path(output_json: Path, suffix: str) -> Path:
    """JSON 出力パスから CSV 出力パスを作る。"""

    return output_json.with_name(f"{output_json.stem}_{suffix}.csv")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(
        description="ADMMHazardAFT パラメータのブートストラップ信頼区間を計算します。"
    )
    parser.add_argument("--data", type=Path, required=True, help="long-format CSV")
    parser.add_argument("--config", type=Path, required=True, help="TOML/JSON config")
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="信頼区間サマリ JSON の保存先",
    )
    parser.add_argument(
        "--coef-ci-csv",
        type=Path,
        default=None,
        help="β 信頼区間 CSV の保存先（省略時は output-json から自動命名）",
    )
    parser.add_argument(
        "--gamma-ci-csv",
        type=Path,
        default=None,
        help="γ 信頼区間 CSV の保存先（省略時は output-json から自動命名）",
    )
    parser.add_argument(
        "--base-result",
        type=Path,
        default=None,
        help="既存 result.json。指定時は点推定値を再 fit せず読み込む。",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="ブートストラップ反復回数",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="信頼係数（例: 0.95）",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=1234,
        help="リサンプリング用 seed",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="並列プロセス数。1 なら逐次実行。",
    )
    parser.add_argument(
        "--include-bootstrap-estimates",
        action="store_true",
        help="成功した全ブートストラップ推定値を JSON に含める。",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI エントリポイント。"""

    args = _parse_args(argv)
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap は正の整数である必要があります。")
    if not 0.0 < float(args.ci_level) < 1.0:
        raise ValueError("--ci-level は 0 と 1 の間である必要があります。")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs は正の整数である必要があります。")

    data = _load_long_format_dataset(args.data)
    config = _load_config_with_data_meta(args.config, args.data)

    if args.base_result is not None:
        point_coef, point_gamma, config = _read_base_result(args.base_result, config)
        original_score = None
    else:
        original_model, original_score = _fit_once(config, data.X, data.y)
        point_coef = np.asarray(original_model.coef_, dtype=float)
        point_gamma = np.asarray(original_model.gamma_, dtype=float).reshape(-1)

    if point_coef.shape != (
        len(config["time_grid"]) - 1,
        len(data.feature_cols),
    ):
        raise ValueError("点推定 coef の形状が data/config と整合しません。")
    if point_gamma.ndim != 1:
        raise ValueError("点推定 gamma は 1 次元である必要があります。")

    seeds = np.random.default_rng(args.random_state).integers(
        0,
        np.iinfo(np.int32).max,
        size=args.n_bootstrap,
        dtype=np.int64,
    )
    worker_args = [
        (replicate, int(seeds[replicate]), config, data.X, data.y)
        for replicate in range(args.n_bootstrap)
    ]

    if args.n_jobs == 1:
        results_iter = (_fit_bootstrap_replicate(item) for item in worker_args)
        results = list(_iter_with_progress(results_iter, total=args.n_bootstrap))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            mapped = executor.map(_fit_bootstrap_replicate, worker_args)
            results = list(_iter_with_progress(mapped, total=args.n_bootstrap))

    successes = [result for result in results if result.error is None]
    failures = [result for result in results if result.error is not None]
    if not successes:
        failure_preview = [failure.error for failure in failures[:5]]
        raise RuntimeError(
            "ブートストラップ推定が全て失敗しました: "
            + json.dumps(failure_preview, ensure_ascii=False)
        )

    boot_coef = np.asarray([result.coef for result in successes], dtype=float)
    boot_gamma = np.asarray([result.gamma for result in successes], dtype=float)
    if boot_coef.shape[1:] != point_coef.shape:
        raise ValueError("bootstrap coef の形状が点推定と整合しません。")
    if boot_gamma.shape[1:] != point_gamma.shape:
        raise ValueError("bootstrap gamma の形状が点推定と整合しません。")

    coef_records = _coef_ci_records(
        point=point_coef,
        boot=boot_coef,
        time_grid=config["time_grid"],
        feature_cols=data.feature_cols,
        ci_level=float(args.ci_level),
    )
    gamma_records = _gamma_ci_records(
        point=point_gamma,
        boot=boot_gamma,
        ci_level=float(args.ci_level),
    )

    coef_csv = args.coef_ci_csv or _default_csv_path(args.output_json, "coef_ci")
    gamma_csv = args.gamma_ci_csv or _default_csv_path(args.output_json, "gamma_ci")
    _write_csv(coef_csv, coef_records)
    _write_csv(gamma_csv, gamma_records)

    failure_records = [
        {
            "replicate": failure.replicate,
            "seed": failure.seed,
            "error": failure.error,
        }
        for failure in failures
    ]
    payload: dict[str, Any] = {
        "data_path": str(args.data),
        "config_path": str(args.config),
        "base_result_path": str(args.base_result) if args.base_result else None,
        "n_samples": int(data.n_subjects),
        "n_features": int(len(data.feature_cols)),
        "feature_cols": data.feature_cols,
        "time_grid": [float(value) for value in config["time_grid"]],
        "ci_method": "percentile",
        "ci_level": float(args.ci_level),
        "n_bootstrap_requested": int(args.n_bootstrap),
        "n_bootstrap_success": int(len(successes)),
        "n_bootstrap_failed": int(len(failures)),
        "random_state": int(args.random_state),
        "n_jobs": int(args.n_jobs),
        "original_score": original_score,
        "coef_ci_csv": str(coef_csv),
        "gamma_ci_csv": str(gamma_csv),
        "coef_ci": coef_records,
        "gamma_ci": gamma_records,
        "failures": failure_records,
        "config": config,
    }
    if args.include_bootstrap_estimates:
        payload["bootstrap_coef"] = boot_coef.tolist()
        payload["bootstrap_gamma"] = boot_gamma.tolist()
        payload["bootstrap_scores"] = [result.score for result in successes]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(
        {
            "output_json": str(args.output_json),
            "coef_ci_csv": str(coef_csv),
            "gamma_ci_csv": str(gamma_csv),
            "n_bootstrap_success": int(len(successes)),
            "n_bootstrap_failed": int(len(failures)),
        }
    )


if __name__ == "__main__":
    main()
