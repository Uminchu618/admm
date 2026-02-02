"""近似対数尤度（log \tilde{L}）とその微分量を提供する目的関数。

責務:
    - 区分求積（Quadrature）を用いた近似対数尤度の計算
    - β（時間区間ごとの係数）と γ（ベースライン係数）に関する勾配・ヘッセの計算

設計意図:
    ソルバ（ADMM）は微分の詳細を知らず、本クラスを通じて value と
    β/γ ごとの勾配・ヘッセのみ利用する。
    これにより、ベースライン表現や求積法の差し替えが容易になる。

注意:
    現状は骨格のみで、value/grad_beta/grad_gamma/hess_* の本体は未実装。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .baseline import BaselineHazardModel
from .quadrature import QuadratureRule
from .time_partition import TimePartition
from .types import ArrayLike


class HazardAFTObjective:
    """Hazard-AFT の目的関数（近似対数尤度）を計算するクラス。"""

    def __init__(
        self,
        baseline: BaselineHazardModel,
        time_partition: TimePartition,
        quadrature: QuadratureRule,
        clip_eta: float,
    ) -> None:
        # baseline: ベースライン（基準）ハザードを表す基底モデル。
        self.baseline = baseline

        # time_partition: time_grid に基づく区間分割と η の組み立てを担当する。
        self.time_partition = time_partition

        # quadrature: 区間積分の数値近似（求積点と重み）を提供する。
        self.quadrature = quadrature

        # clip_eta: exp(η) の爆発を防ぐためのクリッピング幅。
        # 典型的には η を [-clip_eta, clip_eta] に制限する。
        self.clip_eta = clip_eta

    def value(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> float:
        """目的関数値（=-log\tilde{L}）を返す。

        想定:
            - -log\tilde{L}を実装し、ソルバで最小化する形にする。

        Args:
            beta: 時間区間ごとの回帰係数。形状の一例: (K, p) または (K, p+1)。
            gamma: ベースラインの係数ベクトル
            X: 特徴量
            T: 観測時刻
            delta: イベント指示（1=イベント, 0=打ち切り）

        """
        beta_arr, gamma_arr, X_arr, T_arr, delta_arr, eta, exp_eta, k_idx = (
            self._prepare_inputs(beta, gamma, X, T, delta)
        )

        time_grid = np.asarray(self.time_partition.time_grid, dtype=float)
        log_tilde_L = 0.0

        for i in range(X_arr.shape[0]):
            ki = int(k_idx[i])  # 1..K
            Ti = float(T_arr[i])
            di = int(delta_arr[i])

            # イベント項: δ_i h_{i,k(i)}(T_i)
            if di == 1:
                k0 = ki - 1
                eta_ik = float(eta[i, k0])
                x = float(exp_eta[i, k0] * Ti)
                S = np.asarray(
                    self.baseline.basis(np.array([x], dtype=float)), dtype=float
                )
                h = eta_ik + float(S.reshape(-1) @ gamma_arr)
                h = float(np.clip(h, -self.clip_eta, self.clip_eta))
                log_tilde_L += h

            # 積分項: - sum_{k<=k(i)} sum_l w exp(h(v))
            for k0 in range(ki):
                a = float(time_grid[k0])
                b = float(min(Ti, time_grid[k0 + 1]))
                v, w = self.quadrature.nodes_weights(a, b)
                v_arr = np.asarray(v, dtype=float)
                w_arr = np.asarray(w, dtype=float)
                if w_arr.size == 0:
                    continue

                eta_ik = float(eta[i, k0])
                x = exp_eta[i, k0] * v_arr
                S = np.asarray(self.baseline.basis(x), dtype=float)
                h = eta_ik + (S @ gamma_arr)
                h = np.clip(h, -self.clip_eta, self.clip_eta)
                log_tilde_L -= float(np.sum(w_arr * np.exp(h)))

        # ソルバは最小化するので loss=-log\tilde{L} を返す
        return float(-log_tilde_L)

    def grad_beta(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """β に関する勾配（= -∂log\tilde{L}/∂β）を返す。

        Returns:
            g_beta: beta と同形状

        注意:
            実装では η の計算、exp(η) のクリップ、求積による積分近似が関与する。
            形状不一致は実装時に頻出するため、入力検証（shape チェック）が重要。

        """
        beta_arr, gamma_arr, X_arr, T_arr, delta_arr, eta, exp_eta, k_idx = (
            self._prepare_inputs(beta, gamma, X, T, delta)
        )

        time_grid = np.asarray(self.time_partition.time_grid, dtype=float)
        n, _ = X_arr.shape
        K, p_beta = beta_arr.shape

        X_design = self._design_matrix_for_beta(beta_arr, X_arr)
        g_beta = np.zeros((K, p_beta), dtype=float)

        for i in range(n):
            ki = int(k_idx[i])
            Ti = float(T_arr[i])
            di = int(delta_arr[i])
            Xi = X_design[i]

            # イベント項: -δ_i * ∂h/∂β （loss の符号）
            if di == 1:
                k0 = ki - 1
                eta_ik = float(eta[i, k0])
                exp_eta_ik = float(exp_eta[i, k0])
                x = exp_eta_ik * Ti
                S1 = np.asarray(
                    self.baseline.basis_deriv(np.array([x], dtype=float)), dtype=float
                )
                a = float(S1.reshape(-1) @ gamma_arr)
                factor = 1.0 + x * a
                g_beta[k0] -= Xi * factor

            # 積分項: + sum w exp(h) * ∂h/∂β （loss の符号）
            for k0 in range(ki):
                a = float(time_grid[k0])
                b = float(min(Ti, time_grid[k0 + 1]))
                v, w = self.quadrature.nodes_weights(a, b)
                v_arr = np.asarray(v, dtype=float)
                w_arr = np.asarray(w, dtype=float)
                if w_arr.size == 0:
                    continue

                eta_ik = float(eta[i, k0])
                exp_eta_ik = float(exp_eta[i, k0])
                x = exp_eta_ik * v_arr

                S = np.asarray(self.baseline.basis(x), dtype=float)
                S1 = np.asarray(self.baseline.basis_deriv(x), dtype=float)
                a_vec = S1 @ gamma_arr
                factor = 1.0 + x * a_vec

                h = eta_ik + (S @ gamma_arr)
                h = np.clip(h, -self.clip_eta, self.clip_eta)
                weights = w_arr * np.exp(h)
                g_beta[k0] += (weights[:, None] * (factor[:, None] * Xi[None, :])).sum(
                    axis=0
                )

        return g_beta

    def grad_gamma(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """γ に関する勾配（= -∂log\tilde{L}/∂γ）を返す。

        Returns:
            g_gamma: gamma と同形状

        注意:
            実装では η の計算、exp(η) のクリップ、求積による積分近似が関与する。
            形状不一致は実装時に頻出するため、入力検証（shape チェック）が重要。

        """
        beta_arr, gamma_arr, X_arr, T_arr, delta_arr, eta, exp_eta, k_idx = (
            self._prepare_inputs(beta, gamma, X, T, delta)
        )

        time_grid = np.asarray(self.time_partition.time_grid, dtype=float)
        n = X_arr.shape[0]
        M = gamma_arr.size
        g = np.zeros(M, dtype=float)

        for i in range(n):
            ki = int(k_idx[i])
            Ti = float(T_arr[i])
            di = int(delta_arr[i])

            # イベント項: -δ_i S(x(T)) （loss の符号）
            if di == 1:
                k0 = ki - 1
                x = float(exp_eta[i, k0] * Ti)
                S = np.asarray(
                    self.baseline.basis(np.array([x], dtype=float)), dtype=float
                )
                g -= S.reshape(-1)

            # 積分項: + sum w exp(h) S(x(v)) （loss の符号）
            for k0 in range(ki):
                a = float(time_grid[k0])
                b = float(min(Ti, time_grid[k0 + 1]))
                v, w = self.quadrature.nodes_weights(a, b)
                v_arr = np.asarray(v, dtype=float)
                w_arr = np.asarray(w, dtype=float)
                if w_arr.size == 0:
                    continue

                eta_ik = float(eta[i, k0])
                x = exp_eta[i, k0] * v_arr
                S = np.asarray(self.baseline.basis(x), dtype=float)
                h = eta_ik + (S @ gamma_arr)
                h = np.clip(h, -self.clip_eta, self.clip_eta)
                weights = w_arr * np.exp(h)
                g += S.T @ weights

        return g

    def hess_beta(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """β に関するヘッセ行列（= -∂^2log\tilde{L}/∂β^2）を返す。

        Returns:
            - H_bb: β に関するヘッセ（大きくなりやすい）

        注意:
            実用上はブロック対角近似として扱い、
            β-γ の混合ヘッセは別メソッドで管理する前提。

        """
        beta_arr, gamma_arr, X_arr, T_arr, delta_arr, eta, exp_eta, k_idx = (
            self._prepare_inputs(beta, gamma, X, T, delta)
        )

        time_grid = np.asarray(self.time_partition.time_grid, dtype=float)
        n = X_arr.shape[0]
        K, p_beta = beta_arr.shape
        X_design = self._design_matrix_for_beta(beta_arr, X_arr)

        H = np.zeros((K, p_beta, p_beta), dtype=float)

        for i in range(n):
            ki = int(k_idx[i])
            Ti = float(T_arr[i])
            di = int(delta_arr[i])
            Xi = X_design[i]
            Xi_outer = np.outer(Xi, Xi)

            # イベント項（k=k(i) のみ）: loss では -δ_i * ∂^2 h /∂β^2
            if di == 1:
                k0 = ki - 1
                eta_ik = float(eta[i, k0])
                exp_eta_ik = float(exp_eta[i, k0])
                x = exp_eta_ik * Ti
                S1 = np.asarray(
                    self.baseline.basis_deriv(np.array([x], dtype=float)), dtype=float
                )
                S2 = np.asarray(
                    self.baseline.basis_second_deriv(np.array([x], dtype=float)),
                    dtype=float,
                )
                a = float(S1.reshape(-1) @ gamma_arr)
                b = float(S2.reshape(-1) @ gamma_arr)
                scalar = x * a + (x * x) * b
                H[k0] -= Xi_outer * scalar

            # 積分項: loss では + sum w exp(h) * (∂^2 h + (∂h)(∂h)^T)
            for k0 in range(ki):
                a_int = float(time_grid[k0])
                b_int = float(min(Ti, time_grid[k0 + 1]))
                v, w = self.quadrature.nodes_weights(a_int, b_int)
                v_arr = np.asarray(v, dtype=float)
                w_arr = np.asarray(w, dtype=float)
                if w_arr.size == 0:
                    continue

                eta_ik = float(eta[i, k0])
                exp_eta_ik = float(exp_eta[i, k0])
                x = exp_eta_ik * v_arr

                S = np.asarray(self.baseline.basis(x), dtype=float)
                S1 = np.asarray(self.baseline.basis_deriv(x), dtype=float)
                S2 = np.asarray(self.baseline.basis_second_deriv(x), dtype=float)

                a_vec = S1 @ gamma_arr
                b_vec = S2 @ gamma_arr
                dh_factor = 1.0 + x * a_vec
                dh = dh_factor[:, None] * Xi[None, :]
                d2h_scalar = x * a_vec + (x * x) * b_vec

                h = eta_ik + (S @ gamma_arr)
                h = np.clip(h, -self.clip_eta, self.clip_eta)
                weights = w_arr * np.exp(h)

                term_d2h = float(np.dot(weights, d2h_scalar)) * Xi_outer
                term_dh = dh.T @ (dh * weights[:, None])
                H[k0] += term_d2h + term_dh

        return H

    def hess_gamma(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """γ に関するヘッセ行列（= -∂^2log\tilde{L}/∂γ^2）を返す。

        Returns:
            - H_gg: γ に関するヘッセ

        """
        beta_arr, gamma_arr, X_arr, T_arr, delta_arr, eta, exp_eta, k_idx = (
            self._prepare_inputs(beta, gamma, X, T, delta)
        )

        time_grid = np.asarray(self.time_partition.time_grid, dtype=float)
        n = X_arr.shape[0]
        M = gamma_arr.size
        H = np.zeros((M, M), dtype=float)

        for i in range(n):
            ki = int(k_idx[i])
            Ti = float(T_arr[i])

            for k0 in range(ki):
                a = float(time_grid[k0])
                b = float(min(Ti, time_grid[k0 + 1]))
                v, w = self.quadrature.nodes_weights(a, b)
                v_arr = np.asarray(v, dtype=float)
                w_arr = np.asarray(w, dtype=float)
                if w_arr.size == 0:
                    continue

                eta_ik = float(eta[i, k0])
                x = exp_eta[i, k0] * v_arr
                S = np.asarray(self.baseline.basis(x), dtype=float)
                h = eta_ik + (S @ gamma_arr)
                h = np.clip(h, -self.clip_eta, self.clip_eta)
                weights = w_arr * np.exp(h)
                H += S.T @ (S * weights[:, None])

        return H

    def hess_beta_gamma(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """β-γ の混合ヘッセ（現状は 0 で返す）。

        Returns:
            - H_bg: β-γ の混合ヘッセ

        注意:
            実装簡略化で無視する場合もあるが、式通りに実装するための枠を用意する。

        """

        beta_arr = np.asarray(beta, dtype=float)
        gamma_arr = np.asarray(gamma, dtype=float)
        if beta_arr.ndim != 2:
            raise ValueError("beta は 2 次元配列である必要があります")
        if gamma_arr.ndim != 1:
            raise ValueError("gamma は 1 次元配列である必要があります")
        K, p_beta = beta_arr.shape
        M = gamma_arr.size
        return np.zeros((K, p_beta, M), dtype=float)

    def _prepare_inputs(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        beta_arr = np.asarray(beta, dtype=float)
        gamma_arr = np.asarray(gamma, dtype=float).reshape(-1)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        T_arr = np.asarray(T, dtype=float).reshape(-1)
        delta_arr = np.asarray(delta, dtype=int).reshape(-1)

        if beta_arr.ndim != 2:
            raise ValueError("beta は 2 次元配列である必要があります")
        if gamma_arr.ndim != 1:
            raise ValueError("gamma は 1 次元配列である必要があります")
        if X_arr.ndim != 2:
            raise ValueError("X は 2 次元配列である必要があります")
        if T_arr.ndim != 1 or delta_arr.ndim != 1:
            raise ValueError("T, delta は 1 次元配列である必要があります")
        if not (X_arr.shape[0] == T_arr.shape[0] == delta_arr.shape[0]):
            raise ValueError("X, T, delta のサンプル数が一致しません")
        if np.any(~np.isfinite(beta_arr)) or np.any(~np.isfinite(gamma_arr)):
            raise ValueError("beta/gamma に NaN/inf が含まれています")
        if np.any(~np.isfinite(X_arr)) or np.any(~np.isfinite(T_arr)):
            raise ValueError("X/T に NaN/inf が含まれています")
        if np.any((delta_arr != 0) & (delta_arr != 1)):
            raise ValueError("delta は 0/1 である必要があります")

        # time partition から η を組み立て、clip_eta で η をクリップする。
        eta = np.asarray(self.time_partition.eta(beta_arr, X_arr), dtype=float)
        eta = np.clip(eta, -float(self.clip_eta), float(self.clip_eta))
        exp_eta = np.exp(eta)
        k_idx = np.asarray(self.time_partition.interval_index(T_arr), dtype=int)

        return beta_arr, gamma_arr, X_arr, T_arr, delta_arr, eta, exp_eta, k_idx

    def _design_matrix_for_beta(
        self, beta_arr: np.ndarray, X_arr: np.ndarray
    ) -> np.ndarray:
        n_beta = beta_arr.shape[1]
        n, p = X_arr.shape
        if n_beta == p + 1:
            return np.column_stack([np.ones(n, dtype=float), X_arr])
        if n_beta == p:
            return X_arr
        raise ValueError("beta の列数が X と整合しません")
