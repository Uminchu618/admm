"""admm パッケージ。

外部に公開する最小 API（推定器と学習ヘルパ）をここで再エクスポートする。
利用者は基本的に `from admm import ADMMHazardAFT` の形で import できる。
"""

# 公開 API は model.py で定義している。
from .model import ADMMHazardAFT

# __all__:
# - `from admm import *` の対象を明示する。
# - 公開対象を絞り、内部コンポーネント（objective/solver など）を隠蔽する目的。
__all__ = ["ADMMHazardAFT"]
