from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.extended_aft_step_generator import build_generator, load_config


def parse_float_list(text: str, name: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def parse_str_list(text: str, name: str) -> list[str]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def weibull_hazard(t: np.ndarray, alpha: float, rho: float) -> np.ndarray:
    t = np.maximum(t, 1e-12)
    return (alpha / rho) * (t / rho) ** (alpha - 1.0)


def scale_stepwise_beta(cfg: dict[str, Any], scale: float) -> None:
    step = cfg["stepwise_beta"]
    for key in ("beta1_levels", "beta2_levels", "beta3_levels"):
        step[key] = [float(scale) * float(v) for v in step[key]]


def summarize_time_true(time_true: np.ndarray, item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out.update(
        {
            "n": int(time_true.size),
            "mean": float(np.mean(time_true)),
            "std": float(np.std(time_true)),
            "cv": float(np.std(time_true) / np.mean(time_true)),
            "q10": float(np.quantile(time_true, 0.10)),
            "q50": float(np.quantile(time_true, 0.50)),
            "q90": float(np.quantile(time_true, 0.90)),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一時スクリプト: alpha/betaスケール/x23_dist の影響を可視化"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("generation") / "extended_aft_step_generator.config.json",
        help="生成設定JSON",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="1.2,1.5,2.0,3.0",
        help="比較するalphaをカンマ区切りで指定",
    )
    parser.add_argument(
        "--beta-scales",
        type=str,
        default="0.75,1.0,1.25",
        help="stepwise_beta係数全体に掛けるスケールをカンマ区切りで指定",
    )
    parser.add_argument(
        "--x23-dists",
        type=str,
        default="normal,uniform",
        help="比較するx23分布をカンマ区切りで指定（normal,uniform）",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="サンプルサイズ上書き（未指定ならconfigのn）",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="ヒストグラムのbin数",
    )
    parser.add_argument(
        "--output-alpha",
        type=Path,
        default=Path("outputs") / "tmp_alpha_effect.png",
        help="alpha比較図の出力画像パス",
    )
    parser.add_argument(
        "--output-beta",
        type=Path,
        default=Path("outputs") / "tmp_beta_scale_effect.png",
        help="betaスケール比較図の出力画像パス",
    )
    parser.add_argument(
        "--output-x23",
        type=Path,
        default=Path("outputs") / "tmp_x23_dist_effect.png",
        help="x23_dist比較図の出力画像パス",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs") / "tmp_effect_summary.json",
        help="要約統計の出力JSON（alpha/beta/x23をまとめて保存）",
    )
    args = parser.parse_args()

    cfg = load_config(str(args.config))
    alphas = parse_float_list(args.alphas, "--alphas")
    beta_scales = parse_float_list(args.beta_scales, "--beta-scales")
    x23_dists = parse_str_list(args.x23_dists, "--x23-dists")
    invalid_dists = [v for v in x23_dists if v not in ("normal", "uniform")]
    if invalid_dists:
        raise ValueError(f"invalid x23_dist values: {invalid_dists}")

    if args.n is not None:
        cfg["n"] = int(args.n)

    rho = float(cfg["baseline"]["rho"])
    t_max = float(cfg["grid"]["t_max"])
    t = np.linspace(1e-6, t_max, 400)

    alpha_results: dict[float, np.ndarray] = {}
    beta_results: dict[float, np.ndarray] = {}
    x23_results: dict[str, np.ndarray] = {}
    summary: dict[str, list[dict[str, Any]]] = {
        "alpha": [],
        "beta_scale": [],
        "x23_dist": [],
    }

    for alpha in alphas:
        cfg_alpha = copy.deepcopy(cfg)
        cfg_alpha["baseline"]["alpha"] = float(alpha)
        generator = build_generator(cfg_alpha)
        df = generator.simulate()
        time_true = (
            df.groupby("id", sort=True)["time_true"].first().to_numpy(dtype=float)
        )
        alpha_results[alpha] = time_true
        summary["alpha"].append(summarize_time_true(time_true, {"alpha": float(alpha)}))

    for scale in beta_scales:
        cfg_beta = copy.deepcopy(cfg)
        scale_stepwise_beta(cfg_beta, float(scale))
        generator = build_generator(cfg_beta)
        df = generator.simulate()
        time_true = (
            df.groupby("id", sort=True)["time_true"].first().to_numpy(dtype=float)
        )
        beta_results[scale] = time_true
        summary["beta_scale"].append(
            summarize_time_true(time_true, {"beta_scale": float(scale)})
        )

    for dist in x23_dists:
        cfg_dist = copy.deepcopy(cfg)
        cfg_dist["x23_dist"] = dist
        generator = build_generator(cfg_dist)
        df = generator.simulate()
        time_true = (
            df.groupby("id", sort=True)["time_true"].first().to_numpy(dtype=float)
        )
        x23_results[dist] = time_true
        summary["x23_dist"].append(summarize_time_true(time_true, {"x23_dist": dist}))

    fig_alpha, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for alpha in alphas:
        axes[0].plot(
            t, weibull_hazard(t, alpha, rho), linewidth=2, label=f"alpha={alpha:g}"
        )
    axes[0].set_title("Baseline hazard by alpha")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("hazard")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="best")

    for alpha in alphas:
        axes[1].hist(
            alpha_results[alpha],
            bins=args.bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"alpha={alpha:g}",
        )
    axes[1].set_title("time_true distribution by alpha")
    axes[1].set_xlabel("time_true")
    axes[1].set_ylabel("density")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="best")

    args.output_alpha.parent.mkdir(parents=True, exist_ok=True)
    fig_alpha.tight_layout()
    fig_alpha.savefig(args.output_alpha, dpi=150)
    plt.close(fig_alpha)

    fig_beta, ax_beta = plt.subplots(figsize=(6.4, 4.5))
    for scale in beta_scales:
        ax_beta.hist(
            beta_results[scale],
            bins=args.bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"beta_scale={scale:g}",
        )
    ax_beta.set_title("time_true distribution by beta scale")
    ax_beta.set_xlabel("time_true")
    ax_beta.set_ylabel("density")
    ax_beta.grid(True, linestyle=":", alpha=0.6)
    ax_beta.legend(loc="best")

    args.output_beta.parent.mkdir(parents=True, exist_ok=True)
    fig_beta.tight_layout()
    fig_beta.savefig(args.output_beta, dpi=150)
    plt.close(fig_beta)

    fig_x23, ax_x23 = plt.subplots(figsize=(6.4, 4.5))
    for dist in x23_dists:
        ax_x23.hist(
            x23_results[dist],
            bins=args.bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"x23_dist={dist}",
        )
    ax_x23.set_title("time_true distribution by x23_dist")
    ax_x23.set_xlabel("time_true")
    ax_x23.set_ylabel("density")
    ax_x23.grid(True, linestyle=":", alpha=0.6)
    ax_x23.legend(loc="best")

    args.output_x23.parent.mkdir(parents=True, exist_ok=True)
    fig_x23.tight_layout()
    fig_x23.savefig(args.output_x23, dpi=150)
    plt.close(fig_x23)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved plot (alpha): {args.output_alpha}")
    print(f"Saved plot (beta scale): {args.output_beta}")
    print(f"Saved plot (x23_dist): {args.output_x23}")
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()
