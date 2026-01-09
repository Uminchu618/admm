"""fused lasso 付き最適化問題を ADMM で解くソルバ（骨格）。

責務:
    - ADMM 反復（β,γ の更新 → z の prox 更新 → u の双対更新）を回す
    - 収束判定（primal/dual residual）と履歴の記録

設計意図:
    - 目的関数（HazardAFTObjective）と分離し、ソルバが微分式を知らずに済むようにする
    - 将来、(β,γ) 更新を inexact Newton で行う際の枠組みを提供する

注意:
    現状は solve 本体が未実装。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
        # objective: 近似対数尤度の value/grad/hess を提供する目的関数。
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
        """ADMM により (β,γ) を最適化する（未実装）。

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

        想定される例外:
            - 入力 shape が不正な場合の ValueError
            - Newton 更新でヘッセが特異/非正定の場合の数値例外
            - exp(η) の発散による overflow（clip_eta 等で緩和する設計）

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("ADMM solve is not implemented yet.")
