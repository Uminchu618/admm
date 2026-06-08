"""CLI エントリポイント。

目的:
    設定ファイル（TOML/JSON）から `ADMMHazardAFT` 推定器を構築し、
    学習・推論処理へ接続するためのコマンドライン実行口を提供する。

現状:
    - 学習データの読み込みや fit/predict などは今後の実装で追加される想定。

想定される例外:
    - 設定ファイルが存在しない: FileNotFoundError
    - JSON/TOML の構文エラー: パーサ由来の例外
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from admm.config import load_config
from admm.logger import WandBLogger, wandb_available
from admm.model import ADMMHazardAFT


@dataclass
class LongFormatDataset:
    """main.py が扱う long format CSV を NumPy 配列にしたもの。"""

    X: np.ndarray
    y: np.ndarray
    feature_cols: list[str]
    k_count: int
    n_subjects: int


def _parse_predict_times(raw: Optional[str]) -> Optional[list[float]]:
    """カンマ区切り文字列を予測時刻リストへ変換する。"""
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    try:
        times = [float(token.strip()) for token in text.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError(
            "--predict-times はカンマ区切りの数値で指定してください。"
        ) from exc
    if len(times) == 0:
        return None
    return times


def _load_long_format_dataset(data_path: Path) -> LongFormatDataset:
    """long format CSV を ADMMHazardAFT の入力配列へ変換する。"""

    data = pd.read_csv(data_path)
    required_cols = {"id", "k", "time", "event"}
    if not required_cols.issubset(data.columns):
        missing = sorted(required_cols - set(data.columns))
        raise ValueError(
            f"Missing required columns in {data_path} (long format): {missing}"
        )

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
        .to_numpy()
        .reshape(group_sizes.size, k_count, len(feature_cols))
    )
    y_rows = data_sorted.iloc[::k_count]
    y = y_rows[["time", "event"]].to_numpy()

    return LongFormatDataset(
        X=X,
        y=y,
        feature_cols=feature_cols,
        k_count=k_count,
        n_subjects=int(group_sizes.size),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """コマンドライン引数を解釈し、推定器を初期化する。

    Args:
        argv: 引数リスト。None の場合は `sys.argv` を argparse が参照する。

    Returns:
        なし。現状は初期化結果を標準出力に表示するのみ。
    """

    # argparse のパーサを作成し、ユーザー向けの説明文を設定する。
    parser = argparse.ArgumentParser(description="ADMMHazardAFT runner")

    # --config 引数:
    # - 設定ファイルの場所を受け取る
    # - 既定ではカレントディレクトリの config.toml を使う
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to a TOML or JSON config file.",
    )

    # --data 引数:
    # - 学習に使う CSV を指定する
    # - 既定では data/simulated_data.csv を使う
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/simulated_data.csv"),
        help="Path to a CSV dataset (must include time/event columns).",
    )

    # --output 引数:
    # - 結果 JSON の出力先を指定する
    # - 指定がない場合は出力しない
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write result JSON (optional).",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Optional long-format CSV used only for evaluation after fitting.",
    )

    # --plot 引数:
    # - β のステッププロットを保存するか
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save beta step plot (requires matplotlib).",
    )
    parser.add_argument(
        "--load-result",
        type=Path,
        default=None,
        help="Path to existing result.json. If set, skip fit and run prediction only.",
    )
    parser.add_argument(
        "--predict-times",
        type=str,
        default=None,
        help="Comma-separated prediction times (e.g. 1.0,2.0,3.5).",
    )

    # 引数を解析する。argv が None なら OS のコマンドライン引数を使う。
    args = parser.parse_args(argv)

    # 設定を読み込む。ファイル不在・拡張子非対応・パース失敗は例外として伝播する。
    config = load_config(args.config)

    # WandB ログの準備（任意）。
    wandb_logger = None
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_enabled = os.getenv("WANDB_ENABLED", "").lower() in {"1", "true", "yes"}
    if wandb_project or wandb_enabled:
        if wandb_project is None or wandb_project == "":
            wandb_project = "admm"
        if wandb_available():
            wandb_logger = WandBLogger(project=wandb_project, name="admm-run")
            wandb_logger.start_run(config={"config": config})
        else:
            print("WandB が利用できないためロギングをスキップします。")

    # データを読み込み、fit を呼び出す（fit 本体は未実装のため例外はそのまま伝播する）。
    data_path = args.data
    meta_path = Path(f"{data_path}.meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if "time_grid" in meta:
            config["time_grid"] = meta["time_grid"]

    # 実行パラメータを表示してから実行する。
    print("\n=== Run parameters ===")
    print(
        {
            "config_path": str(args.config),
            "data_path": str(args.data),
            "eval_data_path": (
                str(args.eval_data) if args.eval_data is not None else None
            ),
            "output_path": str(args.output) if args.output is not None else None,
            "load_result": (
                str(args.load_result) if args.load_result is not None else None
            ),
            "predict_times": args.predict_times,
            "plot": bool(args.plot),
            "config": config,
        }
    )

    train_data = _load_long_format_dataset(data_path)
    X = train_data.X
    y = train_data.y
    feature_cols = train_data.feature_cols

    eval_data = None
    if args.eval_data is not None:
        eval_data = _load_long_format_dataset(args.eval_data)
        if eval_data.feature_cols != feature_cols:
            raise ValueError("eval-data の特徴量列が train data と一致しません。")
        if eval_data.k_count != train_data.k_count:
            raise ValueError("eval-data の K が train data と一致しません。")

    prediction_times = _parse_predict_times(args.predict_times)

    if args.load_result is not None:
        # result.json に config が含まれる場合は優先してモデルを再構築する。
        with args.load_result.open("r", encoding="utf-8") as handle:
            loaded_result = json.load(handle)
        model_config = dict(config)
        if isinstance(loaded_result.get("config"), dict):
            model_config.update(loaded_result["config"])
        if "time_grid" in loaded_result:
            model_config["time_grid"] = loaded_result["time_grid"]

        model = ADMMHazardAFT.from_config(model_config)
        model.load_params_from_result_json(args.load_result)

        survival = model.predict_survival_function(X, times=prediction_times)
        cumulative = model.predict_cumulative_hazard(X, times=prediction_times)
        c_td = model.score(X, y)
        times_out = (
            np.asarray(prediction_times, dtype=float)
            if prediction_times is not None
            else np.asarray(model.time_grid_[1:], dtype=float)
        )

        print("\n=== Predict-only mode ===")
        print(
            {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[2]),
                "n_times": int(times_out.size),
                "times": times_out.tolist(),
                "c_td": c_td,
            }
        )
        preview_n = min(3, survival.shape[0])
        print("\n=== Survival preview (first rows) ===")
        print(survival[:preview_n])

        if args.output is not None:
            output_path = args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "mode": "predict_only",
                "data_path": str(data_path),
                "loaded_result_path": str(args.load_result),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[2]),
                "feature_cols": feature_cols,
                "time_grid": list(map(float, model.time_grid_)),
                "predict_times": times_out.tolist(),
                "summary": {"c_td": c_td},
                "survival": survival.tolist(),
                "cumulative_hazard": cumulative.tolist(),
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            print(f"Saved prediction JSON to {output_path}")

        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[2]),
                    "n_times": int(times_out.size),
                    "c_td": c_td,
                },
                prefix="predict_only",
            )
            wandb_logger.finish()
        return

    # 設定辞書から推定器を構築する。
    # 余計なキーや型不一致があれば TypeError が発生し得る。
    model = ADMMHazardAFT.from_config(config)
    model.fit(X, y)
    c_td_train = model.score(X, y)
    c_td_eval = None
    if eval_data is not None:
        c_td_eval = model.score(eval_data.X, eval_data.y)
    c_td = c_td_eval if c_td_eval is not None else c_td_train

    # 推定された β を見やすく表示する。
    coef = model.coef_
    time_grid = model.time_grid_
    cols = feature_cols
    index = [f"[{time_grid[k]}, {time_grid[k+1]})" for k in range(len(time_grid) - 1)]
    coef_df = pd.DataFrame(coef, columns=cols, index=index)
    pd.set_option("display.max_columns", 100)
    print("\n=== Estimated beta (coef_) ===")
    print(coef_df)
    print("\n=== Estimated gamma (gamma_) ===")
    print(model.gamma_)
    print("\n=== ADMM history (last) ===")
    last_obj = model.history_["objective"][-1] if model.history_["objective"] else None
    last_neg_loglik = (
        model.history_["neg_loglik"][-1] if model.history_["neg_loglik"] else None
    )
    last_pr = (
        model.history_["primal_residual"][-1]
        if model.history_["primal_residual"]
        else None
    )
    last_dr = (
        model.history_["dual_residual"][-1] if model.history_["dual_residual"] else None
    )
    stopping_reason = model.history_.get("stopping_reason")
    n_admm_iter = model.history_.get("n_admm_iter", len(model.history_["objective"]))
    lambda_fuse_scale = model.history_.get("lambda_fuse_scale")
    lambda_fuse_effective = model.history_.get("lambda_fuse_effective")
    print(
        {
            "objective": last_obj,
            "neg_loglik": last_neg_loglik,
            "primal_residual": last_pr,
            "dual_residual": last_dr,
            "stopping_reason": stopping_reason,
            "n_admm_iter": n_admm_iter,
            "lambda_fuse_scale": lambda_fuse_scale,
            "lambda_fuse_effective": lambda_fuse_effective,
            "c_td": c_td,
            "c_td_train": c_td_train,
            "c_td_test": c_td_eval,
        }
    )
    print("\n=== ADMM last z (z_) ===")
    print(model.z_)

    # β の推定値を時間軸でステップ表示（区分一定）
    if args.plot:
        if plt is None:
            print("matplotlib が利用できないため β のプロットをスキップします。")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            for j, name in enumerate(feature_cols):
                beta_step = np.r_[coef[:, j], coef[-1, j]]
                ax.step(time_grid, beta_step, where="post", label=name)
            ax.set_xlabel("time")
            ax.set_ylabel("Estimated β")
            ax.set_title("Estimated β by time interval")
            ax.legend(loc="best", fontsize="small", ncol=2)
            ax.grid(True, linestyle=":", alpha=0.6)
            output_path = Path("beta_step.png")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            print(f"Saved beta plot to {output_path}")
            plt.show()

    # 結果 JSON を出力
    if args.output is not None:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "data_path": str(data_path),
            "eval_data_path": str(args.eval_data) if args.eval_data else None,
            "n_samples": int(X.shape[0]),
            "n_eval_samples": (
                int(eval_data.X.shape[0]) if eval_data is not None else None
            ),
            "n_features": int(X.shape[2]),
            "feature_cols": feature_cols,
            "time_grid": list(map(float, time_grid)),
            "coef": coef.tolist(),
            "gamma": model.gamma_.tolist(),
            "z_last": model.z_.tolist(),
            "history": model.history_,
            "summary": {
                "objective_last": last_obj,
                "neg_loglik_last": last_neg_loglik,
                "primal_residual_last": last_pr,
                "dual_residual_last": last_dr,
                "stopping_reason": stopping_reason,
                "n_admm_iter": n_admm_iter,
                "lambda_fuse_scale": lambda_fuse_scale,
                "lambda_fuse_effective": lambda_fuse_effective,
                "c_td": c_td,
                "c_td_train": c_td_train,
                "c_td_test": c_td_eval,
            },
            "config": config,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved result JSON to {output_path}")

    # WandB に履歴を可視化（時系列ログ）
    if wandb_logger is not None:
        wandb_logger.log_history(model.history_)
        wandb_logger.log_metrics(
            {
                "objective_last": last_obj,
                "neg_loglik_last": last_neg_loglik,
                "primal_residual_last": last_pr,
                "dual_residual_last": last_dr,
                "stopping_reason": stopping_reason,
                "n_admm_iter": n_admm_iter,
                "lambda_fuse_scale": lambda_fuse_scale,
                "lambda_fuse_effective": lambda_fuse_effective,
                "c_td": c_td,
                "c_td_train": c_td_train,
                "c_td_test": c_td_eval,
                "z_last": model.z_.tolist(),
            },
            prefix="summary",
        )
        wandb_logger.finish()


if __name__ == "__main__":
    # 直接実行時のみ main() を呼び出す（import された場合に副作用を起こさない）。
    main()
