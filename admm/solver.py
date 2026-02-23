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
from tqdm.auto import tqdm

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
        line_search_max_steps: int,
        line_search_shrink: float,
        line_search_c1: float,
        return_best_iterate: bool,
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

        # line_search_*: Newton ステップのバックトラッキング設定。
        self.line_search_max_steps = line_search_max_steps
        self.line_search_shrink = line_search_shrink
        self.line_search_c1 = line_search_c1
        self.return_best_iterate = return_best_iterate

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
        if X_array.ndim != 3:
            raise ValueError("X は 3 次元配列である必要があります。")
        if T_array.ndim != 1:
            raise ValueError("T は 1 次元配列である必要があります。")
        if delta_array.ndim != 1:
            raise ValueError("delta は 1 次元配列である必要があります。")
        if not (X_array.shape[0] == T_array.shape[0] == delta_array.shape[0]):
            raise ValueError("X, T, delta のサンプル数が一致しません。")

        K, n_beta = beta.shape
        if X_array.shape[1] != K:
            raise ValueError("X の K 次元が beta と一致しません。")
        n_features = X_array.shape[2]
        if n_beta != n_features:
            raise ValueError("beta0 の列数が X の特徴量数と整合しません。")
        penalized_cols = np.arange(n_beta)

        n_penalized = int(penalized_cols.size)
        diff_len = max(K - 1, 0)

        if int(self.line_search_max_steps) <= 0:
            raise ValueError("line_search_max_steps は正の整数である必要があります。")
        if not (0.0 < float(self.line_search_shrink) < 1.0):
            raise ValueError("line_search_shrink は (0,1) の範囲である必要があります。")
        if not (0.0 < float(self.line_search_c1) < 1.0):
            raise ValueError("line_search_c1 は (0,1) の範囲である必要があります。")

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

        def safe_base_value(beta_mat: np.ndarray, gamma_vec: np.ndarray) -> float:
            value = float(
                self.objective.value(beta_mat, gamma_vec, X_array, T_array, delta_array)
            )
            if not np.isfinite(value):
                return float(np.inf)
            return value

        def beta_augmented_value(
            beta_mat: np.ndarray,
            gamma_vec: np.ndarray,
            z_mat: np.ndarray,
            u_mat: np.ndarray,
        ) -> float:
            base = safe_base_value(beta_mat, gamma_vec)
            if not np.isfinite(base):
                return float(np.inf)
            if n_penalized > 0 and diff_len > 0:
                residual = diff_beta(beta_mat) - z_mat + u_mat
                base += 0.5 * float(self.rho) * float(np.sum(residual * residual))
            return base

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
            # ペナルティなしの -log\tilde{L}
            "neg_loglik": [],
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

        d_beta_init = diff_beta(beta)
        init_obj = safe_base_value(beta, gamma)
        init_obj += float(self.lambda_fuse * np.sum(np.abs(d_beta_init)))
        best_objective = float(init_obj)
        best_beta = beta.copy()
        best_gamma = gamma.copy()
        best_z = z.copy()
        best_u = u.copy()
        best_iter = -1
        stopped_due_to_invalid = False

        newton_steps = max(1, int(self.newton_steps_per_admm))
        newton_steps = min(newton_steps, max(1, int(self.max_newton_iter)))

        ls_max_steps = int(self.line_search_max_steps)
        ls_shrink = float(self.line_search_shrink)
        ls_c1 = float(self.line_search_c1)

        for admm_iter in tqdm(
            range(int(self.max_admm_iter)),
            desc="ADMM",
            leave=False,
        ):
            beta_step_norm = 0.0
            gamma_step_norm = 0.0
            invalid_state = False

            # (1) gamma を更新 → (2) beta を更新（ブロック座標）
            for _ in range(newton_steps):
                # ----- gamma 更新（Newton + line search） -----
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
                if np.any(~np.isfinite(h_gg)) or np.any(~np.isfinite(g_gamma_vec)):
                    invalid_state = True
                    break

                gamma_old = gamma.copy()
                gamma_ref_value = safe_base_value(beta, gamma_old)

                gamma_newton_step = None
                damp = 0.0
                eye_gg = np.eye(h_gg.shape[0], dtype=float)
                for _ in range(6):
                    try:
                        gamma_newton_step = np.linalg.solve(
                            h_gg + damp * eye_gg, g_gamma_vec
                        )
                        break
                    except np.linalg.LinAlgError:
                        damp = 1e-6 if damp == 0.0 else damp * 10.0

                if gamma_newton_step is None:
                    gamma_direction = -g_gamma_vec
                else:
                    gamma_direction = -gamma_newton_step

                gamma_dir_deriv = float(np.dot(g_gamma_vec, gamma_direction))
                if (not np.isfinite(gamma_dir_deriv)) or gamma_dir_deriv >= 0.0:
                    gamma_direction = -g_gamma_vec
                    gamma_dir_deriv = -float(np.dot(g_gamma_vec, g_gamma_vec))

                gamma_dir_norm = float(np.linalg.norm(gamma_direction))
                if gamma_dir_norm > 0.0 and np.isfinite(gamma_dir_norm):
                    step_scale = 1.0
                    accepted = False
                    accepted_scale = 0.0
                    gamma_candidate = gamma_old
                    for _ in range(ls_max_steps):
                        cand = gamma_old + step_scale * gamma_direction
                        cand_value = safe_base_value(beta, cand)
                        if np.isfinite(cand_value):
                            if np.isfinite(gamma_ref_value):
                                if gamma_dir_deriv < 0.0:
                                    rhs = gamma_ref_value + ls_c1 * step_scale * (
                                        gamma_dir_deriv
                                    )
                                    if cand_value <= rhs:
                                        accepted = True
                                elif cand_value <= gamma_ref_value:
                                    accepted = True
                            else:
                                accepted = True
                        if accepted:
                            gamma_candidate = cand
                            accepted_scale = step_scale
                            break
                        step_scale *= ls_shrink

                    if accepted:
                        gamma = gamma_candidate
                        gamma_step_norm = accepted_scale * gamma_dir_norm
                    else:
                        gamma = gamma_old
                        gamma_step_norm = 0.0
                else:
                    gamma = gamma_old
                    gamma_step_norm = 0.0

                if np.any(~np.isfinite(gamma)):
                    invalid_state = True
                    break

                # ----- beta 更新（Newton + line search） -----
                g_beta = self.objective.grad_beta(
                    beta, gamma, X_array, T_array, delta_array
                )
                h_bb = self.objective.hess_beta(
                    beta, gamma, X_array, T_array, delta_array
                )

                g_beta_mat = np.asarray(g_beta, dtype=float)
                if g_beta_mat.shape != beta.shape:
                    raise ValueError("g_beta の形状が beta と一致しません。")
                if np.any(~np.isfinite(g_beta_mat)):
                    invalid_state = True
                    break

                # ADMM 罰則項の勾配を追加
                if n_penalized > 0 and diff_len > 0:
                    residual = diff_beta(beta) - z + u
                    dtr = d_transpose(residual)
                    for idx, col in enumerate(penalized_cols):
                        g_beta_mat[:, col] += self.rho * dtr[idx]

                # H_bb をフル行列に整形
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
                if np.any(~np.isfinite(h_full)):
                    invalid_state = True
                    break

                # ADMM 罰則項のヘッセ行列を加算
                if n_penalized > 0 and diff_len > 0:
                    for col in penalized_cols:
                        idx = np.arange(K) * n_beta + col
                        h_full[np.ix_(idx, idx)] += self.rho * dtd

                beta_old = beta.copy()
                beta_ref_value = beta_augmented_value(beta_old, gamma, z, u)
                g_beta_vec = g_beta_mat.reshape(-1)

                beta_newton_step = None
                damp = 0.0
                eye_beta = np.eye(h_full.shape[0], dtype=float)
                for _ in range(6):
                    try:
                        beta_newton_step = np.linalg.solve(
                            h_full + damp * eye_beta, g_beta_vec
                        )
                        break
                    except np.linalg.LinAlgError:
                        damp = 1e-6 if damp == 0.0 else damp * 10.0

                if beta_newton_step is None:
                    beta_direction = -g_beta_mat
                else:
                    beta_direction = -beta_newton_step.reshape(beta.shape)

                beta_dir_vec = beta_direction.reshape(-1)
                beta_dir_deriv = float(np.dot(g_beta_vec, beta_dir_vec))
                if (not np.isfinite(beta_dir_deriv)) or beta_dir_deriv >= 0.0:
                    beta_direction = -g_beta_mat
                    beta_dir_vec = beta_direction.reshape(-1)
                    beta_dir_deriv = -float(np.dot(g_beta_vec, g_beta_vec))

                beta_dir_norm = float(np.linalg.norm(beta_dir_vec))
                if beta_dir_norm > 0.0 and np.isfinite(beta_dir_norm):
                    step_scale = 1.0
                    accepted = False
                    accepted_scale = 0.0
                    beta_candidate = beta_old
                    for _ in range(ls_max_steps):
                        cand = beta_old + step_scale * beta_direction
                        cand_value = beta_augmented_value(cand, gamma, z, u)
                        if np.isfinite(cand_value):
                            if np.isfinite(beta_ref_value):
                                if beta_dir_deriv < 0.0:
                                    rhs = beta_ref_value + ls_c1 * step_scale * (
                                        beta_dir_deriv
                                    )
                                    if cand_value <= rhs:
                                        accepted = True
                                elif cand_value <= beta_ref_value:
                                    accepted = True
                            else:
                                accepted = True
                        if accepted:
                            beta_candidate = cand
                            accepted_scale = step_scale
                            break
                        step_scale *= ls_shrink

                    if accepted:
                        beta = beta_candidate
                        beta_step_norm = accepted_scale * beta_dir_norm
                    else:
                        beta = beta_old
                        beta_step_norm = 0.0
                else:
                    beta = beta_old
                    beta_step_norm = 0.0

                if np.any(~np.isfinite(beta)):
                    invalid_state = True
                    break

                if (
                    beta_step_norm < self.newton_tol
                    and gamma_step_norm < self.newton_tol
                ):
                    break

            if invalid_state:
                stopped_due_to_invalid = True
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
            base_value = safe_base_value(beta, gamma)
            penalty = float(self.lambda_fuse * np.sum(np.abs(d_beta)))
            total_objective = base_value + penalty
            history["objective"].append(total_objective)
            history["neg_loglik"].append(base_value)
            history["primal_residual"].append(primal_residual)
            history["dual_residual"].append(dual_residual)
            history["rho"].append(float(self.rho))
            history["newton_steps"].append(int(newton_steps))
            history["beta_step_norm"].append(beta_step_norm)
            history["gamma_step_norm"].append(gamma_step_norm)

            if np.isfinite(total_objective) and total_objective < best_objective:
                best_objective = float(total_objective)
                best_beta = beta.copy()
                best_gamma = gamma.copy()
                best_z = z.copy()
                best_u = u.copy()
                best_iter = int(admm_iter)

            if (
                primal_residual <= self.admm_tol_primal
                and dual_residual <= self.admm_tol_dual
            ):
                break

        if bool(self.return_best_iterate) and np.isfinite(best_objective):
            beta_out = best_beta
            gamma_out = best_gamma
            z_out = best_z
            u_out = best_u
            used_best_iterate = True
        else:
            beta_out = beta
            gamma_out = gamma
            z_out = z
            u_out = u
            used_best_iterate = False

        history["best_objective"] = (
            float(best_objective) if np.isfinite(best_objective) else None
        )
        history["best_iter"] = int(best_iter) if best_iter >= 0 else None
        history["used_best_iterate"] = used_best_iterate
        history["stopped_due_to_invalid"] = bool(stopped_due_to_invalid)

        return beta_out, gamma_out, z_out, u_out, history
