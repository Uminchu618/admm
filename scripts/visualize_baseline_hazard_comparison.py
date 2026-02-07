from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_lsq_spline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admm.baseline import BSplineBaseline


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_estimated_baseline(
    result: Dict[str, Any],
) -> Tuple[BSplineBaseline, np.ndarray, float]:
    time_grid = result.get("time_grid")
    config = result.get("config")
    gamma = result.get("gamma")
    if time_grid is None or config is None or gamma is None:
        raise ValueError("result JSON must contain time_grid, config, and gamma.")

    time_grid = np.asarray(time_grid, dtype=float)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    print("Estimated gamma coefficients:", gamma)
    n_basis = int(config.get("n_baseline_basis", gamma.size))
    clip_eta = float(config.get("clip_eta", 20.0))
    knot_margin = float(config.get("baseline_knot_margin", 1.1))

    if n_basis != gamma.size:
        raise ValueError("n_baseline_basis and gamma length must match.")
    if knot_margin <= 0.0:
        raise ValueError("baseline_knot_margin must be positive.")

    x_min = 0.0
    x_max = float(time_grid[-1]) * knot_margin
    baseline = BSplineBaseline(
        n_basis=n_basis,
        degree=3,
        knots=None,
        knot_range=(x_min, x_max),
        extrapolate=False,
    )
    return baseline, gamma, clip_eta


def weibull_hazard(t: np.ndarray, alpha: float, rho: float) -> np.ndarray:
    t = np.maximum(t, 1e-12)
    return (alpha / rho) * (t / rho) ** (alpha - 1.0)


def estimate_true_gamma(
    baseline: BSplineBaseline,
    t: np.ndarray,
    h_true: np.ndarray,
    clip_eta: float,
) -> Tuple[np.ndarray, BSplineBaseline]:
    log_h = np.log(np.maximum(h_true, 1e-12))
    log_h = np.clip(log_h, -clip_eta, clip_eta)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    fit_baseline = BSplineBaseline(
        n_basis=baseline.n_basis,
        degree=baseline.degree,
        knots=None,
        knot_range=(t_min, t_max),
        extrapolate=False,
    )
    knots = np.asarray(fit_baseline.knots, dtype=float)
    spline = make_lsq_spline(t, log_h, knots, fit_baseline.degree)
    return np.asarray(spline.c, dtype=float), fit_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="main.py の推定結果と真のベースラインハザードを比較可視化"
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("outputs") / "result.json",
        help="main.py の --output で保存した JSON 結果",
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=Path("generation") / "extended_aft_step_generator.config.json",
        help="extended_aft_step_generator の設定ファイル (JSON)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "baseline_hazard_compare.png",
        help="保存する画像パス",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=400,
        help="描画する時間点数",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="描画の上限時刻 (未指定なら generator の t_max か time_grid 最終点)",
    )
    args = parser.parse_args()

    result = load_json(args.result_json)
    baseline, gamma, clip_eta = build_estimated_baseline(result)

    gen_cfg = load_json(args.generator_config)
    baseline_cfg = gen_cfg.get("baseline")
    grid_cfg = gen_cfg.get("grid", {})
    if not baseline_cfg:
        raise ValueError("generator config must include baseline section.")

    alpha = float(baseline_cfg["alpha"])
    rho = float(baseline_cfg["rho"])

    time_grid = np.asarray(result.get("time_grid"), dtype=float)
    t_max = args.t_max
    if t_max is None:
        t_max = float(grid_cfg.get("t_max", time_grid[-1]))

    t = np.linspace(1e-6, t_max, args.n_points, dtype=float)

    S = np.asarray(baseline.basis(t), dtype=float)
    log_h_est = S @ gamma
    log_h_est = np.clip(log_h_est, -clip_eta, clip_eta)
    h_est = np.exp(log_h_est)

    h_true = weibull_hazard(t, alpha, rho)
    gamma_true, fit_baseline = estimate_true_gamma(baseline, t, h_true, clip_eta)
    S_fit = np.asarray(fit_baseline.basis(t), dtype=float)
    log_h_spline = S_fit @ gamma_true
    log_h_spline = np.clip(log_h_spline, -clip_eta, clip_eta)
    h_spline = np.exp(log_h_spline)
    print("Estimated true gamma from Weibull baseline:")
    print(gamma_true)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, h_est, label="estimated baseline", linewidth=2)
    ax.plot(
        t,
        h_spline,
        label="spline approx (true gamma)",
        linestyle="-.",
        linewidth=2,
    )
    ax.plot(t, h_true, label="true Weibull baseline", linestyle="--", linewidth=2)

    for tk in time_grid:
        ax.axvline(tk, color="#CCCCCC", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("time")
    ax.set_ylabel("baseline hazard")
    ax.set_title("Baseline hazard: estimated vs true")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
