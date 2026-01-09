"""CLI エントリポイント。

目的:
    設定ファイル（TOML/JSON）から `ADMMHazardAFT` 推定器を構築し、
    学習・推論処理へ接続するためのコマンドライン実行口を提供する。

現状:
    - 推定器本体は skeleton 段階のため、この CLI は主に "設定が読める" "初期化できる" ことの確認用。
    - 学習データの読み込みや fit/predict などは今後の実装で追加される想定。

想定される例外:
    - 設定ファイルが存在しない: FileNotFoundError
    - JSON/TOML の構文エラー: パーサ由来の例外
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from admm.config import load_config
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

    # 引数を解析する。argv が None なら OS のコマンドライン引数を使う。
    args = parser.parse_args(argv)

    # 設定を読み込む。ファイル不在・拡張子非対応・パース失敗は例外として伝播する。
    config = load_config(args.config)

    # 設定辞書から推定器を構築する。
    # 余計なキーや型不一致があれば TypeError が発生し得る。
    model = ADMMHazardAFT.from_config(config)

    # skeleton 段階のため、生成できたことを表示して終了する。
    print("ADMMHazardAFT skeleton initialized from config.", model)


if __name__ == "__main__":
    # 直接実行時のみ main() を呼び出す（import された場合に副作用を起こさない）。
    main()
