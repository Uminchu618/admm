"""時間分割（time_grid）にもとづき、区間情報や線形予測子 η を扱うユーティリティ。

責務:
    - time_grid = (t0, t1, ..., tK) を保持する
    - 観測時刻 T_i が属する区間インデックス k(i) を計算する
    - 積分のための区間列 (k, a_{ik}, b_{ik}) を生成する
    - β と X から η_{ik} を一括計算する（将来のベクトル化ポイント）

注意:
    現状は skeleton であり、interval_index/iter_intervals/eta は未実装。
"""

from __future__ import annotations

from typing import Sequence

from .types import ArrayLike


class TimePartition:
    """時間区間の管理と η 計算を担うクラス（骨格）。"""

    def __init__(self, time_grid: Sequence[float]) -> None:
        # time_grid は後で参照しやすいよう tuple 化して保持する（不変化）。
        # 典型的には (t0 < t1 < ... < tK) の単調増加を要求する。
        # ただし本 skeleton では検証をまだ実装していない。
        self.time_grid = tuple(time_grid)

    def interval_index(self, T: ArrayLike) -> ArrayLike:
        """各観測時刻 T_i が属する区間インデックス k(i) を返す（未実装）。

        想定:
            time_grid = (t0,...,tK) に対し、T_i が (t_{k-1}, t_k] に入る k を返すなど。
            境界の扱い（左閉右開/右閉など）は資料の定義と一致させる必要がある。

        Args:
            T: 観測時刻の配列。

        Returns:
            区間インデックスの配列。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("interval_index is not implemented yet.")

    def iter_intervals(self, T: ArrayLike) -> ArrayLike:
        """各個体 i の積分区間列を生成する（未実装）。

        典型的には、個体 i ごとに
            a_{ik} = t_{k-1},  b_{ik} = min(T_i, t_k)
        の形で [a_{ik}, b_{ik}] を列挙する。

        Args:
            T: 観測時刻。

        Returns:
            実装方針により、list/iterator/配列など。skeleton では未確定。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("iter_intervals is not implemented yet.")

    def eta(self, beta: ArrayLike, X: ArrayLike) -> ArrayLike:
        """線形予測子 η を計算する（未実装）。

        想定:
            β を (K, p) とし、X を (n, p) とすると、η は (n, K) の行列になる。
            include_intercept=True の場合は切片を考慮した形にする必要がある。

        Args:
            beta: 時間区間ごとの係数。
            X: 特徴量。

        Returns:
            η 行列。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("eta is not implemented yet.")
