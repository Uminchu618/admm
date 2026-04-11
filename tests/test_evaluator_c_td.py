from __future__ import annotations

import numpy as np

from admm.evaluator import HazardAFTEvaluator


class _StubModel:
    def __init__(self, survival: np.ndarray) -> None:
        self._survival = np.asarray(survival, dtype=float)

    def _prepare_predict_X(self, X):
        return np.asarray(X, dtype=float)

    def predict_survival_function(self, X, times=None):
        _ = X
        _ = times
        return self._survival


def test_compute_c_td_basic_ordering() -> None:
    # 3症例、eventは i=0 のみ。比較可能ペアは (0,1), (0,2) の2つ。
    # T_0=1.0 時点で S_0 < S_1, S_0 < S_2 なので Ctd=1.0。
    y = np.array(
        [
            [1.0, 1],
            [2.0, 0],
            [3.0, 0],
        ],
        dtype=float,
    )
    X = np.zeros((3, 2), dtype=float)

    # unique(times)=[1,2,3] の列順を前提に survival を与える
    survival = np.array(
        [
            [0.10, 0.09, 0.08],
            [0.60, 0.55, 0.50],
            [0.80, 0.70, 0.60],
        ],
        dtype=float,
    )
    model = _StubModel(survival)
    evaluator = HazardAFTEvaluator()

    c_td = evaluator.compute_c_td(model, X, y)
    assert np.isclose(c_td, 1.0)


def test_compute_c_td_tie_counts_half() -> None:
    # i=0 の比較で j=1 は同値、j=2 は不一致。
    # concordant = 0.5, comparable = 2 => Ctd = 0.25
    y = np.array(
        [
            [1.0, 1],
            [2.0, 0],
            [3.0, 0],
        ],
        dtype=float,
    )
    X = np.zeros((3, 2), dtype=float)
    survival = np.array(
        [
            [0.30, 0.25, 0.20],
            [0.30, 0.25, 0.20],
            [0.10, 0.09, 0.08],
        ],
        dtype=float,
    )
    model = _StubModel(survival)
    evaluator = HazardAFTEvaluator(tie_score=0.5)

    c_td = evaluator.compute_c_td(model, X, y)
    assert np.isclose(c_td, 0.25)


def test_compute_c_td_no_comparable_pairs_returns_nan() -> None:
    y = np.array(
        [
            [1.0, 0],
            [1.0, 0],
        ],
        dtype=float,
    )
    X = np.zeros((2, 1), dtype=float)
    survival = np.array(
        [
            [0.5],
            [0.4],
        ],
        dtype=float,
    )
    model = _StubModel(survival)
    evaluator = HazardAFTEvaluator()

    c_td = evaluator.compute_c_td(model, X, y)
    assert np.isnan(c_td)
