"""fused lasso 付き最適化問題を ADMM で解くソルバ（骨格）。

責務:
    - ADMM 反復（β,γ の更新 → z の prox 更新 → u の双対更新）を回す
    - 収束判定（primal/dual residual）と履歴の記録

設計意図:
    - 目的関数（HazardAFTObjective）と分離し、ソルバが微分式を知らずに済むようにする
    - 将来、(β,γ) 更新を inexact Newton で行う際の枠組みを提供する

注意:
    目的関数（objective）は未実装のため、solve の数値計算自体は後続実装に依存する。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .objective import HazardAFTObjective
from .types import ArrayLike


class FusedLassoADMMSolver:
    """ADMM による fused lasso ソルバ（骨格）。"""

    def __init__(
        self,
        objective: HazardAFTObjective,
        lambda_fuse: float,
        rho: float,
        max_admm_iter: int,
        admm_tol_primal: float,
        admm_tol_dual: float,
        newton_steps_per_admm: int,
        max_newton_iter: int,
        newton_tol: float,
        random_state: Optional[int],
    ) -> None:
        # objective: 近似対数尤度の value と β/γ の勾配・ヘッセを提供する目的関数。
        self.objective = objective

        # lambda_fuse: fused lasso（差分の L1）正則化の強さ。
        self.lambda_fuse = lambda_fuse

        # rho: ADMM のペナルティ係数。大きいと primal を重視しやすいが、数値的に硬くなることがある。
        self.rho = rho

        # max_admm_iter: ADMM 反復の最大回数。
        self.max_admm_iter = max_admm_iter

        # admm_tol_primal/admm_tol_dual: primal/dual residual の収束閾値。
        self.admm_tol_primal = admm_tol_primal
        self.admm_tol_dual = admm_tol_dual

        # newton_steps_per_admm: ADMM 1反復あたりの Newton ステップ数（inexact Newton の制御）。
        self.newton_steps_per_admm = newton_steps_per_admm

        # max_newton_iter/newton_tol: (β,γ) 更新での Newton 反復制御。
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol

        # random_state: 初期化や乱数を使う場合の再現性のためのシード。
        self.random_state = random_state

    def solve(
        self,
        beta0: ArrayLike,
        gamma0: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Dict[str, Any]]:
        """ADMM により (β,γ) を最適化する。

        Args:
            beta0: β の初期値。
            gamma0: γ の初期値。
            X: 特徴量。
            T: 観測時刻。
            delta: イベント指示。

        Returns:
            (beta, gamma, z, u, history)
            - beta: 推定された β
            - gamma: 推定された γ
            - z: fused lasso 用の補助変数（差分に対する prox の結果）
            - u: scaled dual 変数
            - history: 反復履歴（目的関数、残差、ステップ幅など）

        注意:
            - objective.value/grad_* / hess_* は「最小化対象（例: -log \u007e{L}）」の
              符号規約に合わせて実装する前提。
            - H_bb は (K, p, p) のブロック配列、または (K*p, K*p) のフル行列を想定する。

        想定される例外:
            - 入力 shape が不正な場合の ValueError
            - Newton 更新でヘッセが特異/非正定の場合の数値例外
            - exp(η) の発散による overflow（clip_eta 等で緩和する設計）
        """
        # 入力を NumPy 配列へ正規化する。
        beta = np.asarray(beta0, dtype=float).copy()
        gamma = np.asarray(gamma0, dtype=float).copy()
        X_array = np.asarray(X)
        T_array = np.asarray(T)
        delta_array = np.asarray(delta)

        # 最低限の shape 検証。
        if beta.ndim != 2:
            raise ValueError("beta0 は 2 次元配列である必要があります。")
        if gamma.ndim != 1:
            raise ValueError("gamma0 は 1 次元配列である必要があります。")
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if X_array.ndim != 2:
            raise ValueError("X は 2 次元配列である必要があります。")
        if T_array.ndim != 1:
            raise ValueError("T は 1 次元配列である必要があります。")
        if delta_array.ndim != 1:
            raise ValueError("delta は 1 次元配列である必要があります。")
        if not (X_array.shape[0] == T_array.shape[0] == delta_array.shape[0]):
            raise ValueError("X, T, delta のサンプル数が一致しません。")

        K, n_beta = beta.shape
        n_features = X_array.shape[1]
        if n_beta == n_features + 1:
            penalized_cols = np.arange(1, n_beta)
        elif n_beta == n_features:
            penalized_cols = np.arange(n_beta)
        else:
            raise ValueError("beta0 の列数が X の特徴量数と整合しません。")

        n_penalized = int(penalized_cols.size)
        diff_len = max(K - 1, 0)

        def diff_beta(beta_matrix: np.ndarray) -> np.ndarray:
            if diff_len == 0 or n_penalized == 0:
                return np.zeros((n_penalized, 0), dtype=float)
            diff = beta_matrix[1:, penalized_cols] - beta_matrix[:-1, penalized_cols]
            return diff.T

        def d_transpose(v: np.ndarray) -> np.ndarray:
            if diff_len == 0 or n_penalized == 0:
                return np.zeros((n_penalized, K), dtype=float)
            out = np.zeros((n_penalized, K), dtype=float)
            out[:, 0] = -v[:, 0]
            if K > 2:
                out[:, 1:-1] = v[:, :-1] - v[:, 1:]
            out[:, -1] = v[:, -1]
            return out

        def soft_threshold(v: np.ndarray, thresh: float) -> np.ndarray:
            return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)

        # 初期値: z は Dbeta の値、u はゼロで開始する。
        z = diff_beta(beta)
        u = np.zeros_like(z)

        if diff_len <= 0:
            dtd = np.zeros((K, K), dtype=float)
        else:
            dtd = np.zeros((K, K), dtype=float)
            dtd[0, 0] = 1.0
            dtd[-1, -1] = 1.0
            if K > 2:
                diag = np.full(K - 2, 2.0)
                dtd[1:-1, 1:-1] = np.diag(diag)
            dtd[:-1, 1:] -= np.eye(K - 1)
            dtd[1:, :-1] -= np.eye(K - 1)

        history: Dict[str, Any] = {
            # 目的関数値（最小化対象）：-log\tilde{L} + fused lasso ペナルティ
            "objective": [],
            # primal residual: ||Dβ - z||
            "primal_residual": [],
            # dual residual: ||ρ D^T (z^k - z^{k-1})||
            "dual_residual": [],
            # ADMM ペナルティ係数 ρ（適応化する場合は更新後の値）
            "rho": [],
            # ADMM 1反復あたりの Newton ステップ数
            "newton_steps": [],
            # β 更新量のノルム（damped Newton のステップ長を含む）
            "beta_step_norm": [],
            # γ 更新量のノルム（damped Newton のステップ長を含む）
            "gamma_step_norm": [],
        }

        newton_steps = max(1, int(self.newton_steps_per_admm))
        newton_steps = min(newton_steps, max(1, int(self.max_newton_iter)))

        for admm_iter in range(int(self.max_admm_iter)):
            beta_step_norm = 0.0
            gamma_step_norm = 0.0

            # (1) gamma を Newton 更新 → (2) beta を Newton 更新（ブロック座標）
            for _ in range(newton_steps):
                # gamma 更新
                g_gamma = self.objective.grad_gamma(
                    beta, gamma, X_array, T_array, delta_array
                )
                h_gg = self.objective.hess_gamma(
                    beta, gamma, X_array, T_array, delta_array
                )

                h_gg = np.atleast_2d(np.asarray(h_gg, dtype=float))
                g_gamma_vec = np.asarray(g_gamma, dtype=float).reshape(-1)
                if h_gg.shape[0] != h_gg.shape[1]:
                    raise ValueError("H_gg は正方行列である必要があります。")
                if h_gg.shape[0] != g_gamma_vec.shape[0]:
                    raise ValueError("H_gg と g_gamma の次元が一致しません。")

                try:
                    gamma_step = np.linalg.solve(h_gg, g_gamma_vec)
                except np.linalg.LinAlgError:
                    damp = 1e-6
                    gamma_step = np.linalg.solve(
                        h_gg + damp * np.eye(h_gg.shape[0]), g_gamma_vec
                    )
                gamma = gamma - gamma_step
                gamma_step_norm = float(np.linalg.norm(gamma_step))

                # beta 更新
                g_beta = self.objective.grad_beta(
                    beta, gamma, X_array, T_array, delta_array
                )
                h_bb = self.objective.hess_beta(
                    beta, gamma, X_array, T_array, delta_array
                )

                g_beta_mat = np.asarray(g_beta, dtype=float)
                if g_beta_mat.shape != beta.shape:
                    raise ValueError("g_beta の形状が beta と一致しません。")

                # ADMM 罰則項の勾配を追加
                if n_penalized > 0 and diff_len > 0:
                    residual = diff_beta(beta) - z + u
                    dtr = d_transpose(residual)
                    for idx, col in enumerate(penalized_cols):
                        g_beta_mat[:, col] += self.rho * dtr[idx]

                # H_bb をフル行列に整形する
                h_bb_arr = np.asarray(h_bb, dtype=float)
                n_beta_total = beta.size
                if h_bb_arr.ndim == 3 and h_bb_arr.shape == (K, n_beta, n_beta):
                    h_full = np.zeros((n_beta_total, n_beta_total), dtype=float)
                    for k in range(K):
                        start = k * n_beta
                        end = start + n_beta
                        h_full[start:end, start:end] = h_bb_arr[k]
                elif h_bb_arr.ndim == 2 and h_bb_arr.shape == (
                    n_beta_total,
                    n_beta_total,
                ):
                    h_full = h_bb_arr
                else:
                    raise ValueError("H_bb の形状が想定と一致しません。")

                # ADMM 罰則項のヘッセ行列を加算
                if n_penalized > 0 and diff_len > 0:
                    for col in penalized_cols:
                        idx = np.arange(K) * n_beta + col
                        h_full[np.ix_(idx, idx)] += self.rho * dtd

                g_beta_vec = g_beta_mat.reshape(-1)
                try:
                    beta_step = np.linalg.solve(h_full, g_beta_vec)
                except np.linalg.LinAlgError:
                    damp = 1e-6
                    beta_step = np.linalg.solve(
                        h_full + damp * np.eye(h_full.shape[0]), g_beta_vec
                    )

                beta_step_mat = beta_step.reshape(beta.shape)
                beta = beta - beta_step_mat
                beta_step_norm = float(np.linalg.norm(beta_step_mat))

                if (
                    beta_step_norm < self.newton_tol
                    and gamma_step_norm < self.newton_tol
                ):
                    break

            # z 更新（prox）
            z_prev = z.copy()
            if n_penalized > 0 and diff_len > 0:
                d_beta = diff_beta(beta)
                z = soft_threshold(d_beta + u, self.lambda_fuse / self.rho)
                u = u + d_beta - z
            else:
                d_beta = diff_beta(beta)

            # 残差を計算
            if n_penalized > 0 and diff_len > 0:
                primal_residual = float(np.linalg.norm(d_beta - z))
                dual_step = d_transpose(z - z_prev)
                dual_residual = float(self.rho * np.linalg.norm(dual_step))
            else:
                primal_residual = 0.0
                dual_residual = 0.0

            # 履歴を記録（目的関数は最小化対象として扱う）
            base_value = float(
                self.objective.value(beta, gamma, X_array, T_array, delta_array)
            )
            penalty = float(self.lambda_fuse * np.sum(np.abs(d_beta)))
            history["objective"].append(base_value + penalty)
            history["primal_residual"].append(primal_residual)
            history["dual_residual"].append(dual_residual)
            history["rho"].append(float(self.rho))
            history["newton_steps"].append(int(newton_steps))
            history["beta_step_norm"].append(beta_step_norm)
            history["gamma_step_norm"].append(gamma_step_norm)

            if (
                primal_residual <= self.admm_tol_primal
                and dual_residual <= self.admm_tol_dual
            ):
                break

        return beta, gamma, z, u, history
