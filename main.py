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

    # --plot 引数:
    # - β のステッププロットを保存するか
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save beta step plot (requires matplotlib).",
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

    # 実行パラメータを表示してから実行する。
    print("\n=== Run parameters ===")
    print(
        {
            "config_path": str(args.config),
            "data_path": str(args.data),
            "output_path": str(args.output) if args.output is not None else None,
            "plot": bool(args.plot),
            "config": config,
        }
    )

    # 設定辞書から推定器を構築する。
    # 余計なキーや型不一致があれば TypeError が発生し得る。
    model = ADMMHazardAFT.from_config(config)

    # データを読み込み、fit を呼び出す（fit 本体は未実装のため例外はそのまま伝播する）。
    data_path = args.data
    data = pd.read_csv(data_path)
    required_cols = {"time", "event"}
    if not required_cols.issubset(data.columns):
        missing = sorted(required_cols - set(data.columns))
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    feature_cols = [
        col
        for col in data.columns
        if col not in {"time", "event", "time_true", "c1", "c2"}
    ]
    X = data[feature_cols].to_numpy()
    y = data[["time", "event"]].to_numpy()
    model.fit(X, y)

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
    last_pr = (
        model.history_["primal_residual"][-1]
        if model.history_["primal_residual"]
        else None
    )
    last_dr = (
        model.history_["dual_residual"][-1] if model.history_["dual_residual"] else None
    )
    print({"objective": last_obj, "primal_residual": last_pr, "dual_residual": last_dr})
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
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_cols": feature_cols,
            "time_grid": list(map(float, time_grid)),
            "coef": coef.tolist(),
            "gamma": model.gamma_.tolist(),
            "z_last": model.z_.tolist(),
            "history": model.history_,
            "summary": {
                "objective_last": last_obj,
                "primal_residual_last": last_pr,
                "dual_residual_last": last_dr,
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
                "primal_residual_last": last_pr,
                "dual_residual_last": last_dr,
                "z_last": model.z_.tolist(),
            },
            prefix="summary",
        )
        wandb_logger.finish()


if __name__ == "__main__":
    # 直接実行時のみ main() を呼び出す（import された場合に副作用を起こさない）。
    main()
