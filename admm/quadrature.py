"""区間積分を数値近似するための求積（Quadrature）インターフェース。

目的:
    近似対数尤度では、区間 [a,b] 上の積分が現れる。
    これを Q 点の評価点 v と重み w による加重和で近似する。

設計意図:
    - 求積法の差し替え（Gauss-Legendre / Simpson など）を容易にする
    - 目的関数側は nodes_weights(a,b) だけを使えばよい

注意:
    現状は skeleton のため nodes_weights は未実装。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .types import ArrayLike


class QuadratureRule:
    """求積（Quadrature）ルールを表すクラス。"""

    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        # config は None の可能性があるため、空 dict に正規化して保持する。
        # ここで deepcopy は行わない（呼び出し側が共有しない前提）。
        self.config = config or {}

    def nodes_weights(self, a: float, b: float) -> Tuple[ArrayLike, ArrayLike]:
        """区間 [a,b] に対する求積点と重みを返す（未実装）。

        Args:
            a: 区間左端。
            b: 区間右端。a <= b を想定。

        Returns:
            (v, w)
            - v: 求積点（形状 (Q,)）
            - w: 重み（形状 (Q,)）

        想定されるエラー:
            - a > b の場合は ValueError とするのが自然（実装時）
            - a, b が NaN/inf の場合は数値不安定になるため、事前検証が望ましい

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("nodes_weights is not implemented yet.")
