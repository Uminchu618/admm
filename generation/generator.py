import argparse
import json
import numpy as np
import pandas as pd


class DataGenerator:
    """
    シンプルなAFTモデルに従うデータ生成器。

    モデル: log T = - X β + W,  W ~ Normal(0, σ^2)
      - X はインターセプトを含む (n, p+1) 行列
      - β は長さ p+1 の係数ベクトル（先頭が切片）

    設定パラメータ（configにて指定可能）
      - n: サンプルサイズ（デフォルト: 200）
      - p: 共変量の個数（デフォルト: 5）
      - sigma: W の標準偏差（デフォルト: 1.0）
            - intercept: 切片の固定値（デフォルト: 1.0）
        - covariate_type: 全列に一括指定する分布 ("normal", "binary", "both")
        - covariate_types: 各列ごとに指定する分布のリスト（長さ p）。例: ["binary", "normal", ...]
      - beta: 係数ベクトル（長さ p+1）。未指定なら N(0,1) から乱数生成
      - seed: 乱数シード（デフォルト: 42）
            - censor_threshold: 打ち切り閾値。T がこれを超えたら打ち切り（デフォルト: 5.0）
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.n = cfg.get("n", 200)
        self.p = cfg.get("p", 5)
        self.sigma = cfg.get("sigma", 1.0)
        self.intercept = cfg.get("intercept", 1.0)
        self.cov_type = cfg.get("covariate_type", "normal")
        self.cov_types = cfg.get("covariate_types")  # optional per-column types
        self.seed = cfg.get("seed", 42)
        self.censor_threshold = cfg.get("censor_threshold", 5.0)

        beta_cfg = cfg.get("beta")
        if beta_cfg is None:
            self.beta = np.random.default_rng(self.seed).normal(0.0, 1.0, self.p + 1)
        else:
            self.beta = np.asarray(beta_cfg, dtype=float)
            if self.beta.shape[0] != self.p + 1:
                raise ValueError("beta の長さは p+1 と一致させてください")

    def generate_X(self, seed=None) -> np.ndarray:
        """共変量行列 X を生成（先頭列はインターセプト）。"""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        X = np.ones((self.n, self.p + 1)) * self.intercept

        # 列ごとのタイプ指定があれば優先
        if self.cov_types is not None:
            if len(self.cov_types) != self.p:
                raise ValueError("covariate_types の長さは p と一致させてください")
            for j, t in enumerate(self.cov_types, start=1):  # j=1 is x1
                if t == "normal":
                    X[:, j] = rng.normal(0.0, 1.0, size=self.n)
                elif t == "binary":
                    X[:, j] = rng.binomial(1, 0.5, size=self.n)
                else:
                    raise ValueError(f"Unsupported covariate type for column {j}: {t}")
        else:
            # 一括指定モード
            if self.cov_type == "normal":
                X[:, 1:] = rng.normal(0.0, 1.0, size=(self.n, self.p))
            elif self.cov_type == "binary":
                X[:, 1:] = rng.binomial(1, 0.5, size=(self.n, self.p))
            elif self.cov_type == "both":
                half = self.p // 2
                X[:, 1 : 1 + half] = rng.normal(0.0, 1.0, size=(self.n, half))
                X[:, 1 + half :] = rng.binomial(1, 0.5, size=(self.n, self.p - half))
            else:
                raise ValueError(f"Unknown covariate_type: {self.cov_type}")

        return X

    def generate_T(self, X: np.ndarray, seed=None) -> np.ndarray:
        """log T = -Xβ + W に従う生存時間を生成。"""
        if X.shape != (self.n, self.p + 1):
            raise ValueError("X の形状は (n, p+1) である必要があります")

        rng = np.random.default_rng(self.seed if seed is None else seed)
        noise = rng.normal(0.0, self.sigma, size=self.n)
        log_T = -X.dot(self.beta) + noise
        return np.exp(log_T)

    def simulate(self, seed=None):
        """X と T を同時生成し、DataFrame を返す。"""
        X = self.generate_X(seed)
        T_true = self.generate_T(X, seed)

        # 打ち切り処理: 観測時間 = min(T_true, threshold), event=1 if 事象観測, 0 if 打ち切り
        threshold = self.censor_threshold
        observed_time = np.minimum(T_true, threshold)
        event = (T_true <= threshold).astype(int)

        df = pd.DataFrame(X[:, 1:], columns=[f"x{k}" for k in range(1, self.p + 1)])
        df["time"] = observed_time
        df["event"] = event
        df["time_true"] = T_true
        return df, {"X": X, "T_true": T_true, "time": observed_time, "event": event}


def load_config(config_path: str) -> dict:
    """JSON設定ファイルの読み込み（beta は数値配列を想定）。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="logT = -Xβ + W に従うシンプルなAFTデータを生成"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="設定ファイル（JSON形式）のパス",
        default="generation/simple_generator.config.json",
    )
    parser.add_argument(
        "--output", type=str, help="出力CSVファイルのパス", default="simulated_data.csv"
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    generator = DataGenerator(config)
    df, _ = generator.simulate()
    df.to_csv(args.output, index=False)
    print(f"シミュレーションデータを {args.output} に保存しました。")


if __name__ == "__main__":
    main()
