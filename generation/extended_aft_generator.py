import argparse
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class WeibullBaseline:
    alpha: float
    rho: float

    def hazard(self, t: np.ndarray) -> np.ndarray:
        t = np.maximum(t, 1e-12)
        return (self.alpha / self.rho) * (t / self.rho) ** (self.alpha - 1.0)


@dataclass
class TDParams:
    b11: float
    c1: float
    b21: float
    c2: float
    b31: float
    t0: float
    b30: float


@dataclass
class CensoringParams:
    admin_time: float
    random_a: float
    random_b: float


@dataclass
class GridParams:
    dt: float
    epsilon: float
    t_max: float


class ExtendedAFTGenerator:
    def __init__(
        self,
        n: int,
        scenario: int,
        x23_dist: str,
        baseline: WeibullBaseline,
        td_params: TDParams,
        censoring: CensoringParams,
        grid: GridParams,
        seed: int = 42,
    ):
        if scenario not in (1, 2):
            raise ValueError("scenario must be 1 or 2")
        if x23_dist not in ("normal", "uniform"):
            raise ValueError("x23_dist must be 'normal' or 'uniform'")
        self.n = n
        self.scenario = scenario
        self.x23_dist = x23_dist
        self.baseline = baseline
        self.td_params = td_params
        self.censoring = censoring
        self.grid = grid
        self.seed = seed

    def _beta1(self, t: np.ndarray) -> np.ndarray:
        return self.td_params.b11 * np.exp(-self.td_params.c1 * t)

    def _beta2(self, t: np.ndarray) -> np.ndarray:
        return self.td_params.b21 * np.log1p(self.td_params.c2 * t)

    def _beta3(self, t: np.ndarray) -> np.ndarray:
        if self.scenario == 1:
            return self.td_params.b31 * (t - self.td_params.t0) ** 2
        return np.full_like(t, self.td_params.b30, dtype=float)

    def _g2(self, x: np.ndarray) -> np.ndarray:
        return x

    def _g3(self, x: np.ndarray) -> np.ndarray:
        return x

    def generate_covariates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        x1 = rng.binomial(1, 0.5, size=self.n)
        if self.x23_dist == "normal":
            x2 = rng.normal(0.0, 1.0, size=self.n)
            x3 = rng.normal(0.0, 1.0, size=self.n)
        else:
            x2 = rng.uniform(-1.0, 1.0, size=self.n)
            x3 = rng.uniform(-1.0, 1.0, size=self.n)
        return x1, x2, x3

    def _hazard(self, t: np.ndarray, x1: float, g2: float, g3: float) -> np.ndarray:
        eta = self._beta1(t) * x1 + self._beta2(t) * g2 + self._beta3(t) * g3
        scale_t = np.exp(eta) * t
        return np.exp(eta) * self.baseline.hazard(scale_t)

    def _simulate_event_time(self, x1: float, g2: float, g3: float) -> float:
        t_grid = np.arange(0.0, self.grid.t_max + self.grid.dt, self.grid.dt)
        hazard_vals = self._hazard(t_grid[:-1], x1, g2, g3)
        cum_hazard = np.cumsum(hazard_vals) * self.grid.dt
        surv = np.exp(-cum_hazard)
        u = np.random.default_rng().uniform(0.0, 1.0)
        diff = np.abs(surv - u)
        idx = np.where(diff < self.grid.epsilon)[0]
        if idx.size > 0:
            k = idx[0]
        else:
            k = int(np.argmin(diff))
        return t_grid[k + 1]

    def simulate(self) -> pd.DataFrame:
        x1, x2, x3 = self.generate_covariates()
        g2 = self._g2(x2)
        g3 = self._g3(x3)

        t_true = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            t_true[i] = self._simulate_event_time(x1[i], g2[i], g3[i])

        rng = np.random.default_rng(self.seed + 1)
        c1 = np.full(self.n, self.censoring.admin_time)
        c2 = rng.uniform(self.censoring.random_a, self.censoring.random_b, size=self.n)
        t_obs = np.minimum.reduce([t_true, c1, c2])
        event = (t_true <= c1) & (t_true <= c2)

        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "time": t_obs,
                "event": event.astype(int),
                "time_true": t_true,
                "c1": c1,
                "c2": c2,
            }
        )
        return df


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_generator(cfg: Dict) -> ExtendedAFTGenerator:
    baseline_cfg = cfg["baseline"]
    td_cfg = cfg["time_dependence"]
    censor_cfg = cfg["censoring"]
    grid_cfg = cfg["grid"]

    baseline = WeibullBaseline(alpha=baseline_cfg["alpha"], rho=baseline_cfg["rho"])
    td_params = TDParams(
        b11=td_cfg["b11"],
        c1=td_cfg["c1"],
        b21=td_cfg["b21"],
        c2=td_cfg["c2"],
        b31=td_cfg["b31"],
        t0=td_cfg["t0"],
        b30=td_cfg["b30"],
    )
    censoring = CensoringParams(
        admin_time=censor_cfg["admin_time"],
        random_a=censor_cfg["random_a"],
        random_b=censor_cfg["random_b"],
    )
    grid = GridParams(
        dt=grid_cfg["dt"],
        epsilon=grid_cfg["epsilon"],
        t_max=grid_cfg["t_max"],
    )

    return ExtendedAFTGenerator(
        n=cfg["n"],
        scenario=cfg["scenario"],
        x23_dist=cfg["x23_dist"],
        baseline=baseline,
        td_params=td_params,
        censoring=censoring,
        grid=grid,
        seed=cfg.get("seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="拡張AFT型ハザードモデルに基づくデータ生成"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="generation/extended_aft_generator.config.json",
        help="設定ファイル（JSON）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/simulated_data.csv",
        help="出力CSVパス",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    generator = build_generator(cfg)
    df = generator.simulate()
    df.to_csv(args.output, index=False)
    print(f"生成データを {args.output} に保存しました。")


if __name__ == "__main__":
    main()
