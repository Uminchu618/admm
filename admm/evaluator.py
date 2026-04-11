"""Hazard-AFT モデルの評価ユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .types import ArrayLike

if TYPE_CHECKING:
    from .model import ADMMHazardAFT


@dataclass
class HazardAFTEvaluator:
    """学習済みモデルの評価指標を計算する。"""

    tie_score: float = 0.5
    metric_name: str = "c_td"

    def evaluate(
        self,
        model: "ADMMHazardAFT",
        X: ArrayLike,
        y: ArrayLike,
        times: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """1モデルの評価値を返す。"""

        c_td = self.compute_c_td(model, X, y, times=times)
        return {self.metric_name: c_td}

    def compare(
        self,
        models: Mapping[str, "ADMMHazardAFT"],
        X: ArrayLike,
        y: ArrayLike,
        times: Optional[Sequence[float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """複数モデルを同一データで比較する。"""

        return {
            name: self.evaluate(model, X, y, times=times)
            for name, model in models.items()
        }

    def compute_c_td(
        self,
        model: "ADMMHazardAFT",
        X: ArrayLike,
        y: ArrayLike,
        times: Optional[Sequence[float]] = None,
    ) -> float:
        """定義式に基づく C^{td} を計算する。"""

        y_arr = np.asarray(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != 2:
            raise ValueError("y は (n, 2) 形式（time, event）である必要があります。")

        t = np.asarray(y_arr[:, 0], dtype=float)
        d = np.asarray(y_arr[:, 1])
        if t.ndim != 1:
            raise ValueError("time 列は 1 次元である必要があります。")
        if np.any(~np.isfinite(t)):
            raise ValueError("time に無限大または NaN が含まれています。")
        if np.any(t < 0.0):
            raise ValueError("time は非負である必要があります。")

        if d.dtype.kind == "f" and np.any(~np.isfinite(d)):
            raise ValueError("event に無限大または NaN が含まれています。")
        unique_d = np.unique(d)
        if not np.all(np.isin(unique_d, [0, 1])):
            raise ValueError("event は 0/1 の二値である必要があります。")
        d = d.astype(int)

        x_arr = model._prepare_predict_X(X)
        if x_arr.shape[0] != t.shape[0]:
            raise ValueError("X と y のサンプル数が一致しません。")

        if times is None:
            eval_times = np.unique(t)
        else:
            eval_times = np.asarray(times, dtype=float).reshape(-1)
        if eval_times.size == 0:
            return float("nan")

        survival = np.asarray(
            model.predict_survival_function(x_arr, times=eval_times), dtype=float
        )

        if survival.ndim != 2 or survival.shape[0] != t.size:
            raise ValueError("predict_survival_function の出力 shape が不正です。")

        time_to_col = {float(v): idx for idx, v in enumerate(eval_times.tolist())}
        if len(time_to_col) != eval_times.size:
            raise ValueError("times に重複値が含まれています。")

        concordant = 0.0
        comparable = 0.0
        eps = 1e-12

        for i in range(t.size):
            if d[i] != 1:
                continue
            if float(t[i]) not in time_to_col:
                raise ValueError(
                    "times が指定された場合、すべての event time を含む必要があります。"
                )

            col = time_to_col[float(t[i])]
            s_i = float(survival[i, col])

            mask_j = t > t[i]
            j_indices = np.flatnonzero(mask_j)
            if j_indices.size == 0:
                continue

            comparable += float(j_indices.size)
            s_j = survival[j_indices, col]
            concordant += float(np.sum(s_i + eps < s_j))
            ties = np.abs(s_j - s_i) <= eps
            concordant += float(self.tie_score) * float(np.sum(ties))

        if comparable <= 0.0:
            return float("nan")
        return float(concordant / comparable)
