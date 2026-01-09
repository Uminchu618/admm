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
        """目的関数値を返す（未実装）。

        想定:
            - 返す値は -log\tilde{L}（最小化）として実装してもよいし、
              log\tilde{L}（最大化）として実装してもよい。
              ただしソルバ側と符号規約を一致させる必要がある。

        Args:
            beta: 時間区間ごとの回帰係数。形状の一例: (K, p) または (K, p+1)。
            gamma: ベースラインの係数ベクトル。
            X: 特徴量。
            T: 観測時刻。
            delta: イベント指示（1=イベント, 0=打ち切り）。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("Objective value is not implemented yet.")

    def grad_beta(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """β に関する勾配を返す（未実装）。

        Returns:
            g_beta: beta と同形状

        注意:
            実装では η の計算、exp(η) のクリップ、求積による積分近似が関与する。
            形状不一致は実装時に頻出するため、入力検証（shape チェック）が重要。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("Objective gradient (beta) is not implemented yet.")

    def grad_gamma(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """γ に関する勾配を返す（未実装）。

        Returns:
            g_gamma: gamma と同形状

        注意:
            実装では η の計算、exp(η) のクリップ、求積による積分近似が関与する。
            形状不一致は実装時に頻出するため、入力検証（shape チェック）が重要。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("Objective gradient (gamma) is not implemented yet.")

    def hess_beta(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """β に関するヘッセ行列を返す（未実装）。

        Returns:
            - H_bb: β に関するヘッセ（大きくなりやすい）

        注意:
            実用上はブロック対角近似として扱い、
            β-γ の混合ヘッセは別メソッドで管理する前提。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("Objective hessian (beta) is not implemented yet.")

    def hess_gamma(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """γ に関するヘッセ行列を返す（未実装）。

        Returns:
            - H_gg: γ に関するヘッセ

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("Objective hessian (gamma) is not implemented yet.")

    def hess_beta_gamma(
        self,
        beta: ArrayLike,
        gamma: ArrayLike,
        X: ArrayLike,
        T: ArrayLike,
        delta: ArrayLike,
    ) -> ArrayLike:
        """β-γ の混合ヘッセを返す（未実装）。

        Returns:
            - H_bg: β-γ の混合ヘッセ

        注意:
            実装簡略化で無視する場合もあるが、式通りに実装するための枠を用意する。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError(
            "Objective hessian (beta-gamma) is not implemented yet."
        )
