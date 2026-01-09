"""ベースラインハザード（基準ハザード）を表現するための抽象インターフェース。

本プロジェクトでは「ハザード（hazard）」そのもの、または「対数ハザード（log hazard）」を
基底関数展開で表現し、目的関数（近似対数尤度）や ADMM ソルバから利用できる形に切り出す。

設計上の狙い:
- 将来、B-spline 以外（M-spline / I-spline など）へ差し替え可能にする
- 目的関数や求積（Quadrature）側は「基底値とその導関数」を受け取るだけにして依存を最小化する

注意:
- 本ファイルのクラスは現時点では骨格（skeleton）であり、具体的な計算は未実装。
    そのため各メソッドは NotImplementedError を送出する。
"""

from __future__ import annotations

from .types import ArrayLike


class BaselineHazardModel:
    """ベースラインハザード基底のインターフェース。

    目的:
        時刻（または変換後の時刻）x に対して、基底関数値を返す。
        目的関数の勾配・ヘッセのため、必要に応じて 1階/2階導関数も返す。

    想定する入出力:
        - x: 形状 (n,) 相当の配列（スカラーでもよいが、内部ではベクトル化を想定）
        - 戻り値: 形状 (n, M) の行列（M は基底数）

    例外:
        - 具象クラスが未実装の場合は NotImplementedError
        - 実装により、x が単調でない/負値を含む等の不正入力に ValueError を投げることがある
    """

    def basis(self, x: ArrayLike) -> ArrayLike:
        """基底関数値を返す。

        Args:
            x: 評価点。通常は時刻 t（またはスケーリング/変換済みの時刻）。

        Returns:
            基底行列 S(x)。形状は概ね (n, M)。

        Raises:
            NotImplementedError: 具象クラスで未実装の場合。
        """
        raise NotImplementedError("basis is not implemented yet.")

    def basis_deriv(self, x: ArrayLike) -> ArrayLike:
        """基底関数の 1階導関数を返す。

        勾配・ヘッセを厳密に計算する場合に必要になる。

        Args:
            x: 評価点。

        Returns:
            dS/dx。形状は概ね (n, M)。

        Raises:
            NotImplementedError: 具象クラスで未実装の場合。
        """
        raise NotImplementedError("basis_deriv is not implemented yet.")

    def basis_second_deriv(self, x: ArrayLike) -> ArrayLike:
        """基底関数の 2階導関数を返す。

        Newton 法で 2階情報を用いる場合などに利用する。

        Args:
            x: 評価点。

        Returns:
            d^2S/dx^2。形状は概ね (n, M)。

        Raises:
            NotImplementedError: 具象クラスで未実装の場合。
        """
        raise NotImplementedError("basis_second_deriv is not implemented yet.")


class BSplineBaseline(BaselineHazardModel):
    """B-spline によるベースライン表現（骨格）。

    本クラスは「ベースライン（例: log hazard）を B-spline 基底で展開する」ことを想定している。
    ただし、現時点では基底の具体計算は未実装。

    Attributes:
        n_basis: 基底数 M。大きいほど柔軟だが過学習や数値不安定化のリスクが増える。
    """

    def __init__(self, n_basis: int) -> None:
        # n_basis: 基底数（M）を保持する。
        # ここでは値の検証（n_basis>=1 など）は未実装。
        self.n_basis = n_basis
