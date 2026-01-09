"""ADMM による fused lasso 付き Hazard-AFT 推定器（sklearn 風 API）。

本モジュールは Estimator（外側の "顔"）を提供する。
設計意図は AGENTS.md の方針に沿い、ハイパーパラメータは __init__ 引数、
学習結果と ADMM 状態は fit 後属性（末尾 '_'）として保持する。

注意:
    現状は skeleton 実装であり、入力検証・初期化・推論・スコアは未実装。
    ただし、コンポーネントの依存関係（objective/solver/baseline など）を組み立てる流れは
    将来の実装の土台になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .baseline import BaselineHazardModel, BSplineBaseline
from .objective import HazardAFTObjective
from .quadrature import QuadratureRule
from .solver import FusedLassoADMMSolver
from .time_partition import TimePartition
from .types import ArrayLike


@dataclass
class _FitComponents:
    """学習（fit）に必要な内部コンポーネントを束ねるコンテナ。

    目的:
        fit() の中で複数の依存オブジェクト（ベースライン・時間分割・求積・目的関数・ソルバ）を
        都度生成するため、戻り値としてまとめて扱えるようにする。

    注意:
        外部 API として公開する意図はなく、内部実装の都合のみに利用する。
    """

    baseline: BaselineHazardModel
    time_partition: TimePartition
    quadrature: QuadratureRule
    objective: HazardAFTObjective
    solver: FusedLassoADMMSolver


class ADMMHazardAFT:
    """ADMM による fused lasso 正則化付き Hazard-AFT モデル（推定器）。

    sklearn 互換の作法:
        - __init__ ではハイパーパラメータを属性に保存するだけ（副作用なし）
        - fit により学習し、coef_ / gamma_ / z_ / u_ / history_ 等を保持する

    主要ハイパーパラメータ（概念）:
        - time_grid: 時間分割（区分一定の係数 β(t) を定める区間端点）
        - lambda_fuse: fused lasso の強さ（区間間差分の L1）
        - rho: ADMM のペナルティ係数
        - clip_eta: exp(η) の発散抑制（数値安定化のためのクリップ幅）

    注意:
        本クラスは現時点では一部未実装のため、predict/score/入力検証などは NotImplementedError。
    """

    def __init__(
        self,
        time_grid: Sequence[float],
        include_intercept: bool = True,
        baseline_basis: str = "bspline",
        n_baseline_basis: int = 10,
        quadrature: Optional[Dict[str, Any]] = None,
        lambda_fuse: float = 1.0,
        rho: float = 1.0,
        max_admm_iter: int = 1000,
        admm_tol_primal: float = 1e-4,
        admm_tol_dual: float = 1e-4,
        newton_steps_per_admm: int = 1,
        max_newton_iter: int = 50,
        newton_tol: float = 1e-6,
        clip_eta: float = 20.0,
        random_state: Optional[int] = None,
    ) -> None:
        # 以降は sklearn 流に「引数をそのまま属性に保存」する。
        # 副作用（乱数生成、I/O、重い計算）は行わない。
        self.time_grid = time_grid
        self.include_intercept = include_intercept
        self.baseline_basis = baseline_basis
        self.n_baseline_basis = n_baseline_basis
        self.quadrature = quadrature
        self.lambda_fuse = lambda_fuse
        self.rho = rho
        self.max_admm_iter = max_admm_iter
        self.admm_tol_primal = admm_tol_primal
        self.admm_tol_dual = admm_tol_dual
        self.newton_steps_per_admm = newton_steps_per_admm
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.clip_eta = clip_eta
        self.random_state = random_state

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ADMMHazardAFT":
        """辞書（設定）から推定器を構築する。

        設定ファイル（TOML/JSON）をそのまま渡して生成できるようにするためのヘルパ。
        quadrature は入れ子辞書になりやすいため、明示的に取り出して __init__ に渡す。

        Args:
            config: ハイパーパラメータ辞書。

        Returns:
            構築された ADMMHazardAFT インスタンス。

        Raises:
            TypeError: config が __init__ 引数と整合しない場合（余計なキー/不足）。
        """

        # Mapping を直接変更しないよう、まず通常の dict にコピーする。
        config_dict = dict(config)

        # quadrature はサブ辞書として持ちやすいので、明示的に分離する。
        quadrature = config_dict.pop("quadrature", None)

        # 残りは **config_dict として __init__ に展開する。
        return cls(quadrature=quadrature, **config_dict)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ADMMHazardAFT":
        """モデルを学習する。

        Args:
            X: 特徴量行列（想定: (n, p)）。型は現時点では ArrayLike（Any）。
            y: 目的変数。推奨は (time, event) の2列（y[:,0]=T, y[:,1]=delta）。

        Returns:
            self（sklearn の規約）。

        Raises:
            NotImplementedError: 入力検証や初期化が未実装のため、現状では途中で失敗する。
        """

        # 入力検証: X と y を整形し、観測時刻 T と打ち切り指示 delta に分解する。
        X, T, delta = self._validate_inputs(X, y)

        # 入力次元（特徴量数）を保持する。
        self.n_features_in_ = int(X.shape[1])

        # time_grid を tuple 化して不変にし、学習後属性として保持する。
        self.time_grid_ = tuple(self.time_grid)

        # 内部コンポーネント（baseline/time_partition/quadrature/objective/solver）を構築する。
        components = self._build_components()

        # パラメータ初期値（β, γ）を用意する。
        # ここでの初期化は数値安定性に影響する可能性がある。
        beta0, gamma0 = self._initialize_params(X, T, delta)

        # ADMM ソルバにより最適化し、推定値と ADMM の補助変数（z,u）と履歴を得る。
        beta, gamma, z, u, history = components.solver.solve(beta0, gamma0, X, T, delta)

        # 学習後属性（末尾 '_'）として結果を保持する。
        self.coef_ = beta
        self.gamma_ = gamma
        self.z_ = z
        self.u_ = u

        # 現状は適応 rho を未実装のため、実効 rho は初期値のまま。
        self.rho_ = self.rho

        # デバッグ・収束確認用の履歴（目的関数値、残差など）を保持する。
        self.history_ = history

        # 依存コンポーネントも学習後に参照できるよう保持する。
        self.baseline_ = components.baseline
        self.time_partition_ = components.time_partition
        self.quadrature_ = components.quadrature
        self.objective_ = components.objective
        return self

    def predict_survival_function(
        self, X: ArrayLike, times: Optional[Sequence[float]] = None
    ) -> ArrayLike:
        """生存関数 S(t|X) を返す（未実装）。

        Args:
            X: 特徴量。
            times: 予測を返す時刻点。None の場合は time_grid 等を利用する想定。

        Raises:
            RuntimeError: fit 前に呼ばれた場合（_check_is_fitted）。
            NotImplementedError: 現時点では未実装。
        """
        self._check_is_fitted()
        raise NotImplementedError("predict_survival_function is not implemented yet.")

    def predict_cumulative_hazard(
        self, X: ArrayLike, times: Optional[Sequence[float]] = None
    ) -> ArrayLike:
        """累積ハザード Λ(t|X) を返す（未実装）。

        Raises:
            RuntimeError: fit 前に呼ばれた場合。
            NotImplementedError: 現時点では未実装。
        """
        self._check_is_fitted()
        raise NotImplementedError("predict_cumulative_hazard is not implemented yet.")

    def predict_risk_score(
        self, X: ArrayLike, time: Optional[float] = None
    ) -> ArrayLike:
        """リスクスコア（η など）を返す（未実装）。

        time を与える場合は η(t|X) のような時刻依存スコアを返す設計が考えられる。

        Raises:
            RuntimeError: fit 前に呼ばれた場合。
            NotImplementedError: 現時点では未実装。
        """
        self._check_is_fitted()
        raise NotImplementedError("predict_risk_score is not implemented yet.")

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """モデルのスコアを返す（未実装）。

        sklearn 互換の流儀では、高いほど良いスコアを返す。
        本プロジェクトでは近似対数尤度 log\tilde{L} を返す（あるいは -loss）ことが自然。

        Raises:
            RuntimeError: fit 前に呼ばれた場合。
            NotImplementedError: 現時点では未実装。
        """
        self._check_is_fitted()
        raise NotImplementedError("score is not implemented yet.")

    def _validate_inputs(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """入力検証と前処理を行い、(X, T, delta) に分解して返す。

        本メソッドは「学習用に最低限必要な形・値の妥当性」を確認する。
        実装は NumPy に依存し、入力を `np.asarray` で配列へ正規化する。

        行う検証・変換:
        - X:
            - 1 次元なら (n, 1) に reshape（単一特徴量を想定）
            - 2 次元でない場合は ValueError
        - y:
            - (n, 2) 形式（time, event）であることを要求
            - time 列は float へ変換可能で、有限（NaN/inf なし）かつ非負であること
            - event 列は 0/1 の二値であること
              - 浮動小数の場合のみ NaN/inf を明示チェック
              - 戻り値では int にキャストして返す

        Args:
            X: 特徴量。形状 (n, p) を想定（1 次元は (n, 1) とみなす）。
            y: 目的変数。形状 (n, 2) を要求（1列目=観測時刻 time, 2列目=イベント指示 event）。

        Returns:
            (X_array, T, delta)
            - X_array: 形状 (n, p) の NumPy 配列
            - T: 形状 (n,) の float 配列（観測時刻）
            - delta: 形状 (n,) の int 配列（0/1）

        Raises:
            ValueError: 形状不正、型変換不能、NaN/inf、負の time、event が 0/1 以外、など。
        """

        # X を NumPy 配列に正規化する（リスト/タプル等も受け取れるようにするため）。
        X_array = np.asarray(X)

        # 1 次元入力は「単一特徴量」とみなし、(n, 1) に整形する。
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim != 2:
            raise ValueError("X は 2 次元配列（n, p）である必要があります。")

        # y も NumPy 配列に正規化し、(n, 2) 形式であることを要求する。
        # （time, event）の2列を前提に分解するため。
        y_array = np.asarray(y)
        if y_array.ndim != 2 or y_array.shape[1] != 2:
            raise ValueError("y は (n, 2) 形式（time, event）である必要があります。")
        # サンプル数（行数）が一致しない場合は、入力対応が壊れているためエラー。
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X と y の行数が一致しません。")
        # time 列を float に変換する。変換不能（文字列等）の場合は例外を握りつぶさず原因を付与する。
        try:
            T = y_array[:, 0].astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y の time 列は数値である必要があります。") from exc
        # time に NaN/inf が混じると尤度計算が破綻するため拒否する。
        if np.any(~np.isfinite(T)):
            raise ValueError("time に無限大または NaN が含まれています。")
        # time は非負を要求する（負の時刻はモデル定義上不自然）。
        if np.any(T < 0):
            raise ValueError("time は非負である必要があります。")
        # event 列（打ち切り指示）を取り出す。
        # ここではまず asarray のみ行い、後段で 0/1 判定後に int へキャストして返す。
        delta = np.asarray(y_array[:, 1])
        # event が浮動小数の場合は、NaN/inf 混入を明示的に検出する。
        # （整数型や bool 型の場合は isfinite が使えない/不要なケースがあるため分岐する。）
        if delta.dtype.kind == "f" and np.any(~np.isfinite(delta)):
            raise ValueError("event に無限大または NaN が含まれています。")
        # event が 0/1 の二値であることを検証する。
        # np.unique を使うことで、全要素の検査より高速に「取り得る値の集合」を確認できる。
        unique = np.unique(delta)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("event は 0/1 の二値である必要があります。")
        # 下流（尤度計算・最適化）で扱いやすいよう、event は int（0/1）へ正規化して返す。
        return X_array, T, delta.astype(int)

    def _initialize_params(
        self, X: ArrayLike, T: ArrayLike, delta: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """(β, γ) の初期値を生成する（未実装）。

        例:
            - β をゼロ初期化
            - γ を小さな乱数/ゼロで初期化

        注意:
            初期値は収束性・局所解に影響する可能性があるため、random_state を考慮する設計が望ましい。

        Raises:
            NotImplementedError: 現時点では未実装。
        """
        raise NotImplementedError("Parameter initialization is not implemented yet.")

    def _build_components(self) -> _FitComponents:
        """学習に必要な内部コンポーネントを構築する。"""

        # ベースライン（基底）モデルを選択して生成する。
        baseline = self._build_baseline_model()

        # 時間分割: time_grid をもとに区間情報や η の計算を担う。
        time_partition = TimePartition(self.time_grid)

        # 求積ルール: 区間積分を Q 点の加重和として近似する。
        quadrature = QuadratureRule(self.quadrature)

        # 目的関数: 近似対数尤度の値/勾配/ヘッセを提供する。
        objective = HazardAFTObjective(
            baseline=baseline,
            time_partition=time_partition,
            quadrature=quadrature,
            clip_eta=self.clip_eta,
        )

        # ADMM ソルバ: fused lasso を含む最適化を反復で解く。
        solver = FusedLassoADMMSolver(
            objective=objective,
            lambda_fuse=self.lambda_fuse,
            rho=self.rho,
            max_admm_iter=self.max_admm_iter,
            admm_tol_primal=self.admm_tol_primal,
            admm_tol_dual=self.admm_tol_dual,
            newton_steps_per_admm=self.newton_steps_per_admm,
            max_newton_iter=self.max_newton_iter,
            newton_tol=self.newton_tol,
            random_state=self.random_state,
        )

        # dataclass でコンポーネントをまとめて返す。
        return _FitComponents(
            baseline=baseline,
            time_partition=time_partition,
            quadrature=quadrature,
            objective=objective,
            solver=solver,
        )

    def _build_baseline_model(self) -> BaselineHazardModel:
        """baseline_basis の指定に従ってベースラインモデルを生成する。"""

        # baseline_basis の分岐は将来の拡張ポイント。
        if self.baseline_basis == "bspline":
            # B-spline を選択した場合、基底数 n_baseline_basis を渡して生成する。
            return BSplineBaseline(self.n_baseline_basis)

        # 未対応の指定は、誤設定を早期発見するため NotImplementedError とする。
        raise NotImplementedError(
            f"baseline_basis={self.baseline_basis!r} is not implemented yet."
        )

    def _check_is_fitted(self) -> None:
        """fit 済みかどうかを検査する。

        sklearn の check_is_fitted 相当の軽量版。
        学習後属性（例: coef_）が存在しない場合は、利用順序の誤りとして RuntimeError。
        """

        # 学習後属性の存在で fit 済みかを判定する。
        if not hasattr(self, "coef_"):
            # predict/score を fit 前に呼んだ、などの利用ミスを明確にする。
            raise RuntimeError("This ADMMHazardAFT instance is not fitted yet.")


def optimize_with_admm(model: ADMMHazardAFT, X: ArrayLike, y: ArrayLike) -> ADMMHazardAFT:
    """与えられた推定器を ADMM で学習する薄いラッパ。

    目的:
        関数 API で使いたい場合に備えて残しているが、基本は model.fit を直接呼べばよい。

    Args:
        model: 学習対象の推定器。
        X: 特徴量。
        y: (time, event) などの目的変数。

    Returns:
        学習済みの model（fit が self を返すため同一インスタンス）。
    """

    # 実体は fit 呼び出しのみ。
    return model.fit(X, y)
