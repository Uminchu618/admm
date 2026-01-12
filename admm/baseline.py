"""ベースラインハザード（基準ハザード）を表現するための抽象インターフェース

本プロジェクトでは「ハザード（hazard）」そのもの、または「対数ハザード（log hazard）」を
基底関数展開で表現し、目的関数（近似対数尤度）や ADMM ソルバから利用できる形に切り出す

設計上の狙い:
- 将来、B-spline 以外（M-spline / I-spline など）へ差し替え可能にする
- 目的関数や求積（Quadrature）側は「基底値とその導関数」を受け取るだけにして依存を最小化する

注意:
- BaselineHazardModel は抽象インターフェースであり、具象実装で計算を定義する
- BSplineBaseline は B-spline 基底を SciPy で評価し、微分はスライドの式に従って計算する
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline

from .types import ArrayLike


class BaselineHazardModel:
    """ベースラインハザード基底のインターフェース

    目的:
        時刻（または変換後の時刻）x に対して、基底関数値を返す
        目的関数の勾配・ヘッセのため、必要に応じて 1階/2階導関数も返す

    想定する入出力:
        - x: 形状 (n,) 相当の配列（スカラーでもよいが、内部ではベクトル化を想定）
        - 戻り値: 形状 (n, M) の行列（M は基底数）

    例外:
        - 具象クラスが未実装の場合は NotImplementedError
        - 実装により、x が単調でない/負値を含む等の不正入力に ValueError を投げることがある
    """

    def basis(self, x: ArrayLike) -> ArrayLike:
        """基底関数値を返す

        Args:
            x: 評価点 通常は時刻 t（またはスケーリング/変換済みの時刻）

        Returns:
            基底行列 S(x)形状は概ね (n, M)

        Raises:
            NotImplementedError: 具象クラスで未実装の場合
        """
        raise NotImplementedError("basis is not implemented yet.")

    def basis_deriv(self, x: ArrayLike) -> ArrayLike:
        """基底関数の 1階導関数を返す

        勾配・ヘッセを厳密に計算する場合に必要になる

        Args:
            x: 評価点

        Returns:
            dS/dx形状は概ね (n, M)

        Raises:
            NotImplementedError: 具象クラスで未実装の場合
        """
        raise NotImplementedError("basis_deriv is not implemented yet.")

    def basis_second_deriv(self, x: ArrayLike) -> ArrayLike:
        """基底関数の 2階導関数を返す

        Newton 法で 2階情報を用いる場合などに利用する

        Args:
            x: 評価点

        Returns:
            d^2S/dx^2形状は概ね (n, M)

        Raises:
            NotImplementedError: 具象クラスで未実装の場合
        """
        raise NotImplementedError("basis_second_deriv is not implemented yet.")


class BSplineBaseline(BaselineHazardModel):
    """B-spline 基底によるベースラインハザードモデル

    Args:
        n_basis: 基底数 M
        degree: スプライン次数 p（例: 3 は三次）
        knots: 結節点列 t長さは M + p + 1 を要求する
            None の場合は knot_range に基づく open uniform を構成する
        knot_range: knots を自動生成する際の範囲 (min, max)
        extrapolate: SciPy の BSpline に渡す外挿設定
    """

    def __init__(
        self,
        n_basis: int,
        degree: int = 3,
        knots: Optional[Sequence[float]] = None,
        knot_range: Tuple[float, float] = (0.0, 1.0),
        extrapolate: bool = False,
    ) -> None:
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        self.extrapolate = bool(extrapolate)

        if self.n_basis <= 0:
            raise ValueError("n_basis は正の整数である必要があります")
        if self.degree < 0:
            raise ValueError("degree は 0 以上の整数である必要があります")
        if knots is None:
            raise ValueError("knots を指定してください")
        else:
            self.knots = self._validate_knots(knots)

    def basis(self, x: ArrayLike) -> ArrayLike:
        """B-spline 基底行列 S(x) を返す"""
        x_array = self._as_1d_array(x)
        return self._design_matrix(x_array, self.degree)

    def basis_deriv(self, x: ArrayLike) -> ArrayLike:
        """B-spline 基底の 1 階導関数を返す（式(7)）"""
        if self.degree < 1:
            raise ValueError("degree が 1 未満のため 1 階導関数を定義できません")

        x_array = self._as_1d_array(x)
        p = self.degree
        t = self.knots
        m = self.n_basis

        s_low = self._design_matrix(x_array, p - 1)
        s_m = s_low[:, :m]
        s_m1 = s_low[:, 1 : m + 1]

        den_left = t[p : m + p] - t[:m]
        den_right = t[p + 1 : m + p + 1] - t[1 : m + 1]
        coef_left = self._safe_divide(p, den_left)
        coef_right = self._safe_divide(p, den_right)

        return s_m * coef_left - s_m1 * coef_right

    def basis_second_deriv(self, x: ArrayLike) -> ArrayLike:
        """B-spline 基底の 2 階導関数を返す（式(8)）"""
        if self.degree < 2:
            raise ValueError("degree が 2 未満のため 2 階導関数を定義できません")

        x_array = self._as_1d_array(x)
        p = self.degree
        t = self.knots
        m = self.n_basis

        s_low = self._design_matrix(x_array, p - 2)
        s_m = s_low[:, :m]
        s_m1 = s_low[:, 1 : m + 1]
        s_m2 = s_low[:, 2 : m + 2]

        factor = float(p * (p - 1))

        den_a1 = t[p : m + p] - t[:m]
        den_a2 = t[p - 1 : m + p - 1] - t[:m]
        coef_a = self._safe_divide(factor, den_a1 * den_a2)

        den_b1 = t[p : m + p] - t[1 : m + 1]
        den_b2 = t[p : m + p] - t[:m]
        den_b3 = t[p + 1 : m + p + 1] - t[1 : m + 1]
        term_b = self._safe_divide(1.0, den_b2) + self._safe_divide(1.0, den_b3)
        coef_b = self._safe_divide(factor, den_b1) * term_b

        den_c1 = t[p + 1 : m + p + 1] - t[1 : m + 1]
        den_c2 = t[p + 1 : m + p + 1] - t[2 : m + 2]
        coef_c = self._safe_divide(factor, den_c1 * den_c2)

        return s_m * coef_a - s_m1 * coef_b + s_m2 * coef_c

    def _as_1d_array(self, x: ArrayLike) -> np.ndarray:
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 0:
            x_array = x_array.reshape(1)
        if x_array.ndim != 1:
            raise ValueError("x は 1 次元配列（またはスカラー）である必要があります")
        if np.any(~np.isfinite(x_array)):
            raise ValueError("x に無限大または NaN が含まれています")
        return x_array

    def _design_matrix(self, x: np.ndarray, degree: int) -> np.ndarray:
        n_basis = len(self.knots) - degree - 1
        coeffs = np.eye(n_basis, dtype=float)
        spline = BSpline(self.knots, coeffs, degree, extrapolate=self.extrapolate)
        return spline(x)

    def _validate_knots(self, knots: Sequence[float]) -> np.ndarray:
        knots_array = np.asarray(knots, dtype=float)
        if knots_array.ndim != 1:
            raise ValueError("knots は 1 次元配列である必要があります")
        if np.any(~np.isfinite(knots_array)):
            raise ValueError("knots に無限大または NaN が含まれています")
        expected = self.n_basis + self.degree + 1
        if knots_array.size != expected:
            raise ValueError("knots の長さが n_basis + degree + 1 と一致しません")
        if np.any(np.diff(knots_array) < 0):
            raise ValueError("knots は非減少列である必要があります")
        return knots_array

    def _safe_divide(self, numerator: float, denominator: np.ndarray) -> np.ndarray:
        out = np.zeros_like(denominator, dtype=float)
        np.divide(numerator, denominator, out=out, where=denominator != 0)
        return out
