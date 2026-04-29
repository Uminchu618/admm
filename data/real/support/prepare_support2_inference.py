"""SUPPORT2 データを ADMMHazardAFT の CLI 入力形式へ変換する。

このリポジトリの `main.py` は、すでに学習用に整形された long format の
CSV を読む作りになっている。生の SUPPORT2 CSV は 1 患者 1 行なので、
そのままでは `main.py` に渡せない。

`main.py` が期待する行形式:
    id, k, time, event, feature...

ここで行っていること:
    - config.toml から time_grid を読み、時間区間数 K と時刻範囲を決める
    - SUPPORT2 の `d.time` / `death` を生存時間・イベントとして使う
    - ADMMHazardAFT が扱いやすいように観測時間を [t0, tK] に線形変換する
    - 連続特徴量を平均 0・標準偏差 1 に標準化する
    - 各患者を K 行に複製し、k = 0..K-1 を付けて long format にする
    - 後から再現できるように、標準化係数などを meta JSON に保存する
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import tomllib

import numpy as np
import pandas as pd


# `main.py` は `id`, `k`, `time`, `event` 以外の列をすべて特徴量として扱う。
# そのため、ここで出力する特徴量は数値列だけになるように決めておく。
#
# SUPPORT2 には疾患分類や人種などのカテゴリ列もあるが、このスクリプトでは
# Framingham と同じ「まずは確実に動く最小構成」を優先し、欠損が少なく、
# 数値化の解釈が比較的明確なベースライン情報だけを使う。
DEFAULT_FEATURE_COLS = [
    "age",
    # `sex` は文字列なので、そのままではモデルに渡せない。
    # male=1, female=0 のダミー変数を後で作成する。
    "sex_male",
    "num.co",
    "diabetes",
    "dementia",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "sps",
    "aps",
    "surv2m",
    "surv6m",
    "crea",
    "sod",
]

# 標準化する連続特徴量。
#
# 0/1 のダミー・フラグである `sex_male`, `diabetes`, `dementia` は、
# 値の意味を保つため標準化しない。
# 一方で、年齢・血圧・心拍数・スコア類はスケールが大きく異なるため、
# Newton 更新や ADMM の数値安定性を考えて標準化する。
DEFAULT_STANDARDIZE_COLS = [
    "age",
    "num.co",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "sps",
    "aps",
    "surv2m",
    "surv6m",
    "crea",
    "sod",
]


def _load_time_grid(config_path: Path) -> np.ndarray:
    """config.toml から time_grid を読み込む。

    ADMMHazardAFT は `time_grid = [t0, ..., tK]` によって、回帰係数 beta(t)
    をどの時間区間で区分一定にするかを決める。前処理側も同じ K を使わないと、
    `main.py` が reshape するときに「各 id が K 行ある」という前提が崩れる。
    """

    # TOML は Python 標準の `tomllib` で読む。
    # このプロジェクトは Python 3.14 前提なので追加依存は不要。
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    time_grid = np.asarray(config["time_grid"], dtype=float)

    # time_grid は少なくとも開始・終了の 2 点が必要。
    # 1 次元でない場合は、時間区間の解釈が曖昧なので早めに落とす。
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be a one-dimensional array with >= 2 values")

    # 区間端点は単調増加でないと [t_k, t_{k+1}) が定義できない。
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")

    return time_grid


def prepare_support2(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    meta_output_path: Path | None = None,
) -> dict[str, Any]:
    """SUPPORT2 の生 CSV から long-format CSV と meta JSON を作成する。

    Args:
        input_path: 生の `support2.csv`。
        output_path: `main.py --data` に渡すための long-format CSV 出力先。
        config_path: `time_grid` を含む設定ファイル。
        meta_output_path: 前処理の要約 JSON 出力先。None なら
            `{output_path}.meta.json` を使う。

    Returns:
        作成行数、イベント数、標準化係数などの要約辞書。
    """

    # Framingham と同様に、学習設定ファイルの time_grid を前処理にも使う。
    # K は区間数なので、端点数から 1 を引く。
    time_grid = _load_time_grid(config_path)
    k_count = int(time_grid.size - 1)
    t0 = float(time_grid[0])
    tK = float(time_grid[-1])

    # SUPPORT2 CSV では欠損が "NA" 文字列で入っているため、pandas の欠損値として読む。
    source = pd.read_csv(input_path, na_values=["NA"])

    # SUPPORT2 には明示的な患者 ID 列がないので、元 CSV の行番号から安定した id を作る。
    # 1 始まりにしておくと、後で結果を見たときに 0 始まりより人間が追いやすい。
    source = source.reset_index(names="source_row")
    source["id"] = source["source_row"].astype(int) + 1

    # 文字列カテゴリ `sex` を数値特徴量へ変換する。
    # female は 0 として表現され、基準カテゴリの役割になる。
    source["sex_male"] = (source["sex"] == "male").astype(int)

    # `d.time`: 登録から死亡または打ち切りまでの日数
    # `death`: 追跡期間中に死亡したかどうか
    # この 2 列を ADMMHazardAFT の y=(time,event) に対応させる。
    required_cols = ["id", "d.time", "death", *DEFAULT_FEATURE_COLS]
    base = source.loc[:, required_cols].copy()

    # モデル入力はすべて数値である必要がある。
    # 変換できない値が混ざっていれば NaN にして、後続の dropna で除外する。
    for col in required_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    # 欠損を含む症例は、今回は単純に除外する。
    # SUPPORT2 は欠損列が多いので、特徴量セットは欠損が少ない列を中心にしている。
    base = base.dropna(subset=required_cols).reset_index(drop=True)

    # 0 日以下の生存時間は、このモデルの time_grid [t0, tK] 上で扱いづらい。
    # SUPPORT2 の `d.time` は今回のデータでは正の値だが、防御的に条件を入れておく。
    base = base.loc[base["d.time"] > 0].reset_index(drop=True)
    if base.empty:
        raise ValueError("No rows remain after SUPPORT2 preprocessing")

    # SUPPORT2 の追跡日数は最大 2029 日程度で、config.toml の time_grid は 0..6。
    # `main.py` / `ADMMHazardAFT` は time が time_grid の範囲内にあることを前提にするため、
    # Framingham と同じく、元の観測時間を [t0, tK] に線形スケールする。
    #
    # 例: max_followup の患者は tK に対応し、それより短い観測時間は比例して 0..tK に入る。
    # これにより、既存の config.toml を変更せずに Support2 でも同じ実行経路を使える。
    max_followup = float(base["d.time"].max())
    base["time"] = (base["d.time"] * (tK / max_followup)).clip(lower=t0, upper=tK)

    # event は 1=死亡, 0=打ち切り。
    base["event"] = base["death"].astype(int)

    # 標準化係数は学習データ（base）から計算する。
    # 今回は train/test split を切っていないので、出力 CSV 全体が学習対象になる。
    # 将来 split する場合は、train で計算した係数を valid/test に適用するのが望ましい。
    standardization = {}
    for col in DEFAULT_STANDARDIZE_COLS:
        mean = float(base[col].mean())
        std = float(base[col].std(ddof=0))

        # 標準偏差が 0 だと割り算できず、標準化後に inf/NaN が出る。
        # その場合は特徴量として情報がないので、ここで明示的に失敗させる。
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid standard deviation for {col}")

        base[col] = (base[col] - mean) / std

        # 再現性確認・推論時の同じ変換に使えるように meta JSON へ残す。
        standardization[f"{col}_mean"] = mean
        standardization[f"{col}_std"] = std

    # ここまでは 1 患者 1 行。
    # `main.py` は各 id について k=0..K-1 の K 行がある long format を期待している。
    base_keep = base[["id", *DEFAULT_FEATURE_COLS, "time", "event"]].copy()

    # 各患者行を K 回繰り返す。
    # SUPPORT2 は今回、時間依存共変量を持たせず「各区間で同じ特徴量」を使う。
    # beta(t) は k ごとに変わるので、同じ特徴量でも区間別の係数が推定される。
    long_df = base_keep.loc[base_keep.index.repeat(k_count)].reset_index(drop=True)

    # 繰り返した各患者行に、区間番号 k=0..K-1 を割り当てる。
    long_df["k"] = np.tile(np.arange(k_count, dtype=int), base_keep.shape[0])

    # `main.py` が読みやすい列順に揃える。
    # `id`, `k`, `time`, `event` 以外は自動的に特徴量として扱われる。
    long_df = long_df[["id", "k", "time", "event", *DEFAULT_FEATURE_COLS]]

    # id, k の順に整列する。
    # `main.py` 側で reshape するとき、この順序が崩れると患者と区間の対応が壊れる。
    long_df = long_df.sort_values(["id", "k"]).reset_index(drop=True)

    # long-format CSV を保存する。これが `main.py --data` に渡す本体ファイル。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path, index=False)

    # 前処理結果の要約を作る。
    # 学習結果 JSON だけでは「どの列を使い、何件落とし、どう標準化したか」が分からないので、
    # 変換時点の情報を別 JSON として残しておく。
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config_path": str(config_path),
        "time_grid": time_grid.tolist(),
        "K": k_count,
        "t0": t0,
        "tK": tK,
        "source_rows": int(source.shape[0]),
        "n_subjects": int(base_keep.shape[0]),
        "n_rows": int(long_df.shape[0]),
        "n_events": int(base_keep["event"].sum()),
        "n_censored": int((base_keep["event"] == 0).sum()),
        "followup_max_original": max_followup,
        "time_min": float(long_df["time"].min()),
        "time_max": float(long_df["time"].max()),
        "feature_cols": DEFAULT_FEATURE_COLS,
        "standardize_cols": DEFAULT_STANDARDIZE_COLS,
        "standardization": standardization,
    }

    # 明示的な出力先が指定されなければ、CSV の隣に
    # `support2_inference.csv.meta.json` のような名前で保存する。
    if meta_output_path is None:
        meta_output_path = Path(f"{output_path}.meta.json")

    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    """コマンドライン引数を受け取り、SUPPORT2 前処理を実行する。"""

    # スクリプト単体で実行できるように CLI を用意する。
    # 既定値はリポジトリ直下から実行する前提のパスにしている。
    parser = argparse.ArgumentParser(description="Prepare SUPPORT2 inference CSV")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/real/support/support2.csv"),
        help="Path to raw SUPPORT2 CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/support/support2_inference.csv"),
        help="Path to write the long-format CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to ADMM config with time_grid.",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Optional path to write preprocessing metadata JSON.",
    )
    args = parser.parse_args()

    # 実際の変換処理は `prepare_support2` に寄せる。
    # こうしておくと、将来テストや Notebook から関数として再利用しやすい。
    summary = prepare_support2(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        meta_output_path=args.meta_output,
    )

    # 実行ログとして、人間がすぐ確認できる形で要約を標準出力にも出す。
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # import されたときに前処理が勝手に走らないよう、直接実行時だけ main を呼ぶ。
    main()
