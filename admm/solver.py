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
        admm_tol_rel: float,
        admm_stagnation_tol: float,
        admm_stagnation_patience: int,
        newton_steps_per_admm: int,
        max_newton_iter: int,
        newton_tol: float,
        line_search_max_steps: int,
        line_search_shrink: float,
        line_search_c1: float,
        return_best_iterate: bool,
        random_state: Optional[int],
        adaptive_rho: bool = False,
        rho_balance_mu: float = 10.0,
        rho_increase_factor: float = 2.0,
        rho_decrease_factor: float = 2.0,
        rho_update_interval: int = 10,
        rho_min: float = 1e-6,
        rho_max: float = 1e6,
    ) -> None:
        # objective: 近似対数尤度の value と β/γ の勾配・ヘッセを提供する目的関数。
        self.objective = objective

        # lambda_fuse: fused lasso（差分の L1）正則化の強さ。
        # 尤度はサンプル和として実装しているため、solve() 内で N 倍して使う。
        self.lambda_fuse = lambda_fuse

        # rho: ADMM のペナルティ係数。大きいと primal を重視しやすいが、数値的に硬くなることがある。
        self.rho = rho

        # max_admm_iter: ADMM 反復の最大回数。
        self.max_admm_iter = max_admm_iter

        # admm_tol_primal/admm_tol_dual: primal/dual residual の収束閾値。
        self.admm_tol_primal = admm_tol_primal
        self.admm_tol_dual = admm_tol_dual
        self.admm_tol_rel = admm_tol_rel
        self.admm_stagnation_tol = admm_stagnation_tol
        self.admm_stagnation_patience = admm_stagnation_patience

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

        # residual balancing による rho の適応更新。
        self.adaptive_rho = adaptive_rho
        self.rho_balance_mu = rho_balance_mu
        self.rho_increase_factor = rho_increase_factor
        self.rho_decrease_factor = rho_decrease_factor
        self.rho_update_interval = rho_update_interval
        self.rho_min = rho_min
        self.rho_max = rho_max

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

        n_samples = int(X_array.shape[0])
        lambda_fuse = float(self.lambda_fuse)
        if lambda_fuse < 0.0:
            raise ValueError("lambda_fuse は 0 以上である必要があります。")
        # objective.value は -log L のサンプル和なので O(N)。
        # (1/N) loss + lambda P と同じ解になるよう、loss + N*lambda P を解く。
        lambda_fuse_effective = float(n_samples) * lambda_fuse
        rho_current = float(self.rho)

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
        if not np.isfinite(rho_current) or rho_current <= 0.0:
            raise ValueError("rho は正の有限値である必要があります。")
        if not (0.0 < float(self.line_search_shrink) < 1.0):
            raise ValueError("line_search_shrink は (0,1) の範囲である必要があります。")
        if not (0.0 < float(self.line_search_c1) < 1.0):
            raise ValueError("line_search_c1 は (0,1) の範囲である必要があります。")
        if float(self.admm_tol_primal) < 0.0:
            raise ValueError("admm_tol_primal は 0 以上である必要があります。")
        if float(self.admm_tol_dual) < 0.0:
            raise ValueError("admm_tol_dual は 0 以上である必要があります。")
        if float(self.admm_tol_rel) < 0.0:
            raise ValueError("admm_tol_rel は 0 以上である必要があります。")
        if float(self.admm_stagnation_tol) < 0.0:
            raise ValueError("admm_stagnation_tol は 0 以上である必要があります。")
        if int(self.admm_stagnation_patience) <= 0:
            raise ValueError("admm_stagnation_patience は正の整数である必要があります。")
        if float(self.rho_balance_mu) <= 1.0:
            raise ValueError("rho_balance_mu は 1 より大きい必要があります。")
        if float(self.rho_increase_factor) <= 1.0:
            raise ValueError("rho_increase_factor は 1 より大きい必要があります。")
        if float(self.rho_decrease_factor) <= 1.0:
            raise ValueError("rho_decrease_factor は 1 より大きい必要があります。")
        if int(self.rho_update_interval) <= 0:
            raise ValueError("rho_update_interval は正の整数である必要があります。")
        if float(self.rho_min) <= 0.0 or float(self.rho_max) < float(self.rho_min):
            raise ValueError("rho_min/rho_max の範囲が不正です。")

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
                base += 0.5 * rho_current * float(np.sum(residual * residual))
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
            # 当該反復後の rho と更新方向。
            "rho_next": [],
            "rho_update": [],
            # Boyd 型の停止判定で使う許容誤差（絶対 + 相対）
            "primal_tolerance": [],
            "dual_tolerance": [],
            # 停滞判定用の objective 相対変化量と連続停滞回数
            "objective_relative_change": [],
            "stagnation_count": [],
            # ADMM 1反復あたりの Newton ステップ数
            "newton_steps": [],
            # β 更新量のノルム（damped Newton のステップ長を含む）
            "beta_step_norm": [],
            # γ 更新量のノルム（damped Newton のステップ長を含む）
            "gamma_step_norm": [],
        }

        d_beta_init = diff_beta(beta)
        init_obj = safe_base_value(beta, gamma)
        init_obj += float(lambda_fuse_effective * np.sum(np.abs(d_beta_init)))
        best_objective = float(init_obj)
        best_beta = beta.copy()
        best_gamma = gamma.copy()
        best_z = z.copy()
        best_u = u.copy()
        best_iter = -1
        stopped_due_to_invalid = False
        stopping_reason = "max_iter"
        previous_objective: Optional[float] = None
        stagnation_count = 0

        newton_steps = max(1, int(self.newton_steps_per_admm))
        newton_steps = min(newton_steps, max(1, int(self.max_newton_iter)))

        ls_max_steps = int(self.line_search_max_steps)
        ls_shrink = float(self.line_search_shrink)
        ls_c1 = float(self.line_search_c1)
        tol_abs_primal = float(self.admm_tol_primal)
        tol_abs_dual = float(self.admm_tol_dual)
        tol_rel = float(self.admm_tol_rel)
        stagnation_tol = float(self.admm_stagnation_tol)
        stagnation_patience = int(self.admm_stagnation_patience)

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
                        g_beta_mat[:, col] += rho_current * dtr[idx]

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
                        h_full[np.ix_(idx, idx)] += rho_current * dtd

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
                z = soft_threshold(d_beta + u, lambda_fuse_effective / rho_current)
                u = u + d_beta - z
            else:
                d_beta = diff_beta(beta)

            # 残差を計算
            if n_penalized > 0 and diff_len > 0:
                primal_residual = float(np.linalg.norm(d_beta - z))
                dual_step = d_transpose(z - z_prev)
                dual_residual = float(rho_current * np.linalg.norm(dual_step))
                primal_scale = max(
                    float(np.linalg.norm(d_beta)),
                    float(np.linalg.norm(z)),
                )
                dual_scale = float(rho_current * np.linalg.norm(d_transpose(u)))
                primal_tolerance = np.sqrt(float(z.size)) * tol_abs_primal
                primal_tolerance += tol_rel * primal_scale
                dual_tolerance = np.sqrt(float(beta.size)) * tol_abs_dual
                dual_tolerance += tol_rel * dual_scale
            else:
                primal_residual = 0.0
                dual_residual = 0.0
                primal_tolerance = 0.0
                dual_tolerance = 0.0

            # 履歴を記録（目的関数は最小化対象として扱う）
            base_value = safe_base_value(beta, gamma)
            penalty = float(lambda_fuse_effective * np.sum(np.abs(d_beta)))
            total_objective = base_value + penalty
            if (
                previous_objective is not None
                and np.isfinite(previous_objective)
                and np.isfinite(total_objective)
            ):
                objective_relative_change = abs(total_objective - previous_objective)
                objective_relative_change /= max(1.0, abs(previous_objective))
            else:
                objective_relative_change = None
            previous_objective = float(total_objective)

            history["objective"].append(total_objective)
            history["neg_loglik"].append(base_value)
            history["primal_residual"].append(primal_residual)
            history["dual_residual"].append(dual_residual)
            history["rho"].append(float(rho_current))
            history["primal_tolerance"].append(float(primal_tolerance))
            history["dual_tolerance"].append(float(dual_tolerance))
            history["objective_relative_change"].append(objective_relative_change)
            history["newton_steps"].append(int(newton_steps))
            history["beta_step_norm"].append(beta_step_norm)
            history["gamma_step_norm"].append(gamma_step_norm)
            history["rho_next"].append(float(rho_current))
            history["rho_update"].append("none")

            if np.isfinite(total_objective) and total_objective < best_objective:
                best_objective = float(total_objective)
                best_beta = beta.copy()
                best_gamma = gamma.copy()
                best_z = z.copy()
                best_u = u.copy()
                best_iter = int(admm_iter)

            # 停滞停止は residual が収束許容値の近くまで来ている場合だけ使う。
            # β/γ が一時的に動かないだけの反復を早止まりしないため。
            if (
                beta_step_norm <= self.newton_tol
                and gamma_step_norm <= self.newton_tol
                and objective_relative_change is not None
                and objective_relative_change <= stagnation_tol
                and primal_residual <= 10.0 * primal_tolerance
                and dual_residual <= 10.0 * dual_tolerance
            ):
                stagnation_count += 1
            else:
                stagnation_count = 0
            history["stagnation_count"].append(int(stagnation_count))

            if (
                primal_residual <= primal_tolerance
                and dual_residual <= dual_tolerance
            ):
                stopping_reason = "residual_converged"
                break
            if stagnation_count >= stagnation_patience:
                stopping_reason = "stagnated"
                break

            # Boyd et al. の residual balancing。scaled dual 変数 u は
            # rho の変更前後で unscaled dual が不変になるよう補正する。
            should_update_rho = (
                bool(self.adaptive_rho)
                and (admm_iter + 1) % int(self.rho_update_interval) == 0
                and (admm_iter + 1) < int(self.max_admm_iter)
            )
            if should_update_rho:
                rho_old = rho_current
                rho_update = "none"
                if primal_residual > float(self.rho_balance_mu) * dual_residual:
                    rho_current = min(
                        rho_old * float(self.rho_increase_factor),
                        float(self.rho_max),
                    )
                    if rho_current > rho_old:
                        rho_update = "increase"
                elif dual_residual > float(self.rho_balance_mu) * primal_residual:
                    rho_current = max(
                        rho_old / float(self.rho_decrease_factor),
                        float(self.rho_min),
                    )
                    if rho_current < rho_old:
                        rho_update = "decrease"
                if rho_current != rho_old:
                    u *= rho_old / rho_current
                history["rho_next"][-1] = float(rho_current)
                history["rho_update"][-1] = rho_update

        terminal_iter = len(history["objective"]) - 1
        converged = bool(
            stopping_reason == "residual_converged"
            and terminal_iter >= 0
            and history["primal_residual"][terminal_iter]
            <= history["primal_tolerance"][terminal_iter]
            and history["dual_residual"][terminal_iter]
            <= history["dual_tolerance"][terminal_iter]
        )

        # 正式収束時は必ず収束反復を返す。未収束時の best iterate は、少なくとも
        # 一度評価済みの反復だけを候補にし、初期値を推定結果として返さない。
        if converged:
            beta_out = beta
            gamma_out = gamma
            z_out = z
            u_out = u
            returned_iter = terminal_iter
            returned_from = "converged_iterate"
            used_best_iterate = False
        elif (
            bool(self.return_best_iterate)
            and best_iter >= 0
            and np.isfinite(best_objective)
        ):
            beta_out = best_beta
            gamma_out = best_gamma
            z_out = best_z
            u_out = best_u
            returned_iter = int(best_iter)
            returned_from = "best_iterate"
            used_best_iterate = True
        else:
            beta_out = beta
            gamma_out = gamma
            z_out = z
            u_out = u
            returned_iter = terminal_iter if terminal_iter >= 0 else None
            returned_from = "last_iterate"
            used_best_iterate = False

        history["best_objective"] = (
            float(best_objective) if np.isfinite(best_objective) else None
        )
        history["lambda_fuse"] = lambda_fuse
        history["lambda_fuse_scale"] = n_samples
        history["lambda_fuse_effective"] = lambda_fuse_effective
        history["best_iter"] = int(best_iter) if best_iter >= 0 else None
        history["used_best_iterate"] = used_best_iterate
        history["returned_iter"] = returned_iter
        history["returned_from"] = returned_from
        history["stopped_due_to_invalid"] = bool(stopped_due_to_invalid)
        if stopped_due_to_invalid:
            stopping_reason = "invalid_state"
        history["stopping_reason"] = stopping_reason
        history["n_admm_iter"] = int(len(history["objective"]))
        history["converged"] = converged
        history["bic_eligible"] = converged
        history["rho_final"] = float(rho_current)

        returned_metric_keys = (
            "objective",
            "neg_loglik",
            "primal_residual",
            "dual_residual",
            "primal_tolerance",
            "dual_tolerance",
            "rho",
        )
        for key in returned_metric_keys:
            values = history[key]
            value = (
                values[returned_iter]
                if returned_iter is not None and 0 <= returned_iter < len(values)
                else None
            )
            history[f"returned_{key}"] = value

        return beta_out, gamma_out, z_out, u_out, history
