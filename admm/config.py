"""設定ファイル（TOML/JSON）を読み込むユーティリティ。

目的:
        実験・学習ジョブを「設定ファイルで再現可能」にするため、ハイパーパラメータ等を
        JSON/TOML として外部化し、辞書（dict）としてロードする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import tomllib



def load_config(path: Path) -> Dict[str, Any]:
    """設定ファイルを読み込み、Python の辞書として返す。

    Args:
        path: 設定ファイルへのパス。拡張子でフォーマットを判定する。

    Returns:
        設定内容を表す辞書。値は JSON/TOML の型（数値/文字列/配列/辞書など）として返る。

    Raises:
        FileNotFoundError: 指定パスが存在しない場合。
        RuntimeError: TOML 指定だが tomllib が利用できない場合（Python < 3.11）。
        ValueError: 対応していない拡張子の場合。
        json.JSONDecodeError: JSON のパースに失敗した場合。
        Exception: TOML のパースに失敗した場合（tomllib が投げる例外）。
    """

    # 設定ファイルが存在しない場合は、早期に失敗させて原因を明確化する。
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # 拡張子でフォーマットを判定する。
    # ここでは大文字小文字の正規化（.TOML 等）までは行っていない。
    if path.suffix == ".toml":
        # tomllib が無い環境（Python < 3.11）では TOML を扱えない。
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+.")

        # tomllib.load はバイナリファイルオブジェクトを想定する。
        # そのため 'rb' で開く。
        with path.open("rb") as handle:
            return tomllib.load(handle)

    if path.suffix == ".json":
        # JSON は UTF-8 を既定として読み込む。
        # ファイルが壊れている/途中で切れている等の場合、JSONDecodeError が発生し得る。
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # ここに到達するのは想定外の拡張子。
    # 例: .yaml など。対応を増やす場合はこの分岐に追加する。
    raise ValueError(f"Unsupported config format: {path.suffix}")
