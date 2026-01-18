"""区間積分を数値近似するための求積（Quadrature）インターフェース。

目的:
    近似対数尤度では、区間 [a,b] 上の積分が現れる。
    これを Q 点の評価点 v と重み w による加重和で近似する。

設計意図:
    - 求積法の差し替え（Gauss-Legendre / Simpson など）を容易にする
    - 目的関数側は nodes_weights(a,b) だけを使えばよい

注意:
    デフォルトは Gauss-Legendre（Q=10）を用いる。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .types import ArrayLike


class QuadratureRule:
    """求積（Quadrature）ルールを表すクラス。

    config の想定:
        - "Q": 求積点数（正の整数）
        - "rule": "gauss_legendre" または "simpson"
    """

    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        # config は None の可能性があるため、空 dict に正規化して保持する。
        # ここで deepcopy は行わない（呼び出し側が共有しない前提）。
        self.config = config or {}
        self.rule = self._normalize_rule(self.config.get("rule", "gauss_legendre"))
        self.q = int(self.config.get("Q", 10))
        self._validate_rule()

    def nodes_weights(self, a: float, b: float) -> Tuple[ArrayLike, ArrayLike]:
        """区間 [a,b] に対する求積点と重みを返す。

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
            ValueError: a > b、または rule/Q の条件を満たさない場合。
        """
        a_float = float(a)
        b_float = float(b)
        if not np.isfinite(a_float) or not np.isfinite(b_float):
            raise ValueError("a,b は有限値である必要があります。")
        if a_float > b_float:
            raise ValueError("a は b 以下である必要があります。")

        if a_float == b_float:
            v = np.full(self.q, a_float, dtype=float)
            w = np.zeros(self.q, dtype=float)
            return v, w

        if self.rule == "gauss_legendre":
            nodes, weights = np.polynomial.legendre.leggauss(self.q)
            half = 0.5 * (b_float - a_float)
            center = 0.5 * (b_float + a_float)
            v = half * nodes + center
            w = half * weights
            return v, w

        v = np.linspace(a_float, b_float, self.q, dtype=float)
        h = (b_float - a_float) / (self.q - 1)
        weights = np.ones(self.q, dtype=float)
        weights[1:-1:2] = 4.0
        weights[2:-2:2] = 2.0
        w = (h / 3.0) * weights
        return v, w

    def _validate_rule(self) -> None:
        if self.q <= 0:
            raise ValueError("Q は正の整数である必要があります。")
        if self.rule == "gauss_legendre":
            return
        if self.rule == "simpson":
            if self.q < 3 or self.q % 2 == 0:
                raise ValueError("Simpson の Q は 3 以上の奇数である必要があります。")
            return
        raise ValueError(f"未知の rule が指定されました: {self.rule!r}")

    def _normalize_rule(self, rule: Any) -> str:
        if rule is None:
            return "gauss_legendre"
        return str(rule).strip().lower().replace("-", "_")
