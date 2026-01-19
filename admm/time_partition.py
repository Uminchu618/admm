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

import numpy as np

from .types import ArrayLike


class TimePartition:
    """時間区間の管理と η 計算を担うクラス（骨格）。"""

    def __init__(self, time_grid: Sequence[float]) -> None:
        # time_grid は後で参照しやすいよう tuple 化して保持する（不変化）。
        # 典型的には (t0 < t1 < ... < tK) の単調増加を要求する。
        # ただし本 skeleton では検証をまだ実装していない。
        self.time_grid = tuple(float(t) for t in time_grid)
        if len(self.time_grid) < 2:
            raise ValueError("time_grid は 2 点以上である必要があります")
        if any(not np.isfinite(t) for t in self.time_grid):
            raise ValueError("time_grid に NaN/inf が含まれています")
        if np.any(np.diff(np.asarray(self.time_grid, dtype=float)) <= 0):
            raise ValueError("time_grid は狭義単調増加である必要があります")

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
        T_array = np.asarray(T, dtype=float)
        if T_array.ndim != 1:
            raise ValueError("T は 1 次元配列である必要があります")
        if np.any(~np.isfinite(T_array)):
            raise ValueError("T に NaN/inf が含まれています")

        grid = np.asarray(self.time_grid, dtype=float)
        t0 = float(grid[0])
        tK = float(grid[-1])
        if np.any(T_array < t0):
            raise ValueError("T に time_grid の範囲外（t0 未満）が含まれています")

        # slides15.md の定義: t_{k-1} <= T < t_k を満たす k(i) を返す（1-based）。
        # searchsorted(grid, T, side='right') は T==t_k のとき k+1 を返す。
        # T==tK の場合のみ範囲外になるので K に丸める。
        idx = np.searchsorted(grid, T_array, side="right")
        K = len(grid) - 1
        idx = np.clip(idx, 1, K)

        # 念のため: 上限 tK より大きい値が入っている場合はエラーにする。
        if np.any(T_array > tK):
            raise ValueError("T に time_grid の範囲外（tK 超）が含まれています")
        return idx.astype(int)

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
        T_array = np.asarray(T, dtype=float)
        if T_array.ndim != 1:
            raise ValueError("T は 1 次元配列である必要があります")
        k_idx = np.asarray(self.interval_index(T_array), dtype=int)
        grid = np.asarray(self.time_grid, dtype=float)

        intervals = []
        for i, Ti in enumerate(T_array):
            ki = int(k_idx[i])
            row = []
            for k in range(1, ki + 1):
                a = float(grid[k - 1])
                b = float(min(Ti, grid[k]))
                if b < a:
                    raise ValueError(
                        "不正な区間が生成されました (a>b)。time_grid と T を確認してください"
                    )
                row.append((k, a, b))
            intervals.append(row)
        return intervals

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
        beta_arr = np.asarray(beta, dtype=float)
        X_arr = np.asarray(X, dtype=float)

        if beta_arr.ndim != 2:
            raise ValueError("beta は 2 次元配列である必要があります")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X は 2 次元配列である必要があります")
        if np.any(~np.isfinite(beta_arr)) or np.any(~np.isfinite(X_arr)):
            raise ValueError("beta または X に NaN/inf が含まれています")

        K_expected = len(self.time_grid) - 1
        K, n_beta = beta_arr.shape
        if K != K_expected:
            raise ValueError("beta の行数（K）が time_grid と一致しません")

        n_samples, n_features = X_arr.shape
        if n_beta == n_features + 1:
            X_design = np.column_stack([np.ones(n_samples, dtype=float), X_arr])
        elif n_beta == n_features:
            X_design = X_arr
        else:
            raise ValueError("beta の列数が X の特徴量数（+切片）と整合しません")

        # (n, p_beta) @ (p_beta, K) -> (n, K)
        return X_design @ beta_arr.T
