# ADMM Hazard-AFT Model

ADMM（交互方向乗数法）を用いた Fused Lasso 正則化付き Hazard-AFT モデルの実装。

## 概要

本リポジトリは、ADMM（Alternating Direction Method of Multipliers）を用いて
fused lasso（時間方向の差分に対する L1 正則化）付きの Hazard-AFT モデルを推定するための
Python 実装です。

時間区分ごとに回帰係数が変化する生存時間分析モデルを、Fused Lasso により時間方向のスパース性を導入しながら推定します。

- **モデル**: Hazard-based AFT（Accelerated Failure Time）
- **正則化**: Fused Lasso（区間間の差分に L1 ペナルティ）
- **最適化**: ADMM（Alternating Direction Method of Multipliers）
- **ベースラインハザード**: B-spline 基底展開

## インストール

```bash
# uvを使った環境構築
uv sync

# または直接実行
uv run main.py --config config.toml --data data/simulated_data.csv
```

## 基本的な使い方

### データ生成（段階的βの拡張AFT）

段階的に係数が変化する拡張AFTデータを生成します。

```bash
uv run generation/extended_aft_step_generator.py \
  --config generation/extended_aft_step_generator.config.json \
  --output data/simulated_data.csv
```

設定は [generation/extended_aft_step_generator.config.json](generation/extended_aft_step_generator.config.json) を参照してください。

### 単一実験の実行

```bash
uv run main.py --config config.toml --data data/simulated_data.csv --output result.json --plot
```

### Lambda並列実験（スパコン環境）

Lambda値を変えながら複数のデータセットで並列実験を行います。

#### 1. Lambda値の設定

`lambda_grid.json` で実験する lambda値を定義：

```json
{
  "description": "Lambda values for parallel experiments (log scale: 0.01 to 10, 10 points)",
  "lambda_values": [0.01, 0.0215, 0.0464, 0.1, 0.215, 0.464, 1.0, 2.15, 4.64, 10.0]
}
```

#### 2. ジョブ投入

```bash
# UGEアレイジョブとして投入（データ100件 × lambda10点 = 1000パターン）
qsub qsub.sh
```

`qsub.sh` は `SGE_TASK_ID` を使って自動的に以下を切り替えます：
- 処理するデータファイル（`data/extended_aft/*.csv`）
- 使用する lambda値（`lambda_grid.json` から選択）

#### 3. 結果の集計

実験完了後、結果を集計：

```bash
uv run scripts/aggregate_lambda_results.py --base-dir outputs/lambda_experiments --output outputs/lambda_summary.csv
```

#### 4. 結果の可視化

```bash
uv run scripts/visualize_lambda_results.py --summary outputs/lambda_summary.csv --output-dir outputs/lambda_plots
```

生成されるプロット：
- `lambda_vs_objective.png`: Lambda値と目的関数の関係
- `lambda_distribution.png`: Lambda値ごとの目的関数分布（箱ひげ図）
- `lambda_vs_convergence.png`: Lambda値と収束状況（primal/dual residual）

集計CSVには `c_td`（time-dependent C-index 相当）も含まれます。

### ローカルでのテスト実行

```bash
# 特定のタスクIDを指定して単一実験を実行
./run_lambda_experiment.sh 1

# データインデックスとlambdaインデックスの対応：
# task_id = data_idx * n_lambda + lambda_idx + 1
# 例: task_id=1 → data=1, lambda=1
#     task_id=11 → data=2, lambda=1
#     task_id=101 → data=11, lambda=1
```

## ディレクトリ構造

```
.
├── admm/                  # コアパッケージ
│   ├── model.py          # ADMMHazardAFT推定器
│   ├── solver.py         # ADMMソルバ
│   ├── objective.py      # 目的関数・勾配・ヘッセ
│   ├── baseline.py       # ベースラインハザード（B-spline）
│   ├── time_partition.py # 時間分割・η計算
│   ├── quadrature.py     # 求積ルール
│   └── logger.py         # WandBロガー
├── scripts/              # ユーティリティスクリプト
│   ├── aggregate_lambda_results.py  # 結果集計
│   └── visualize_lambda_results.py  # 結果可視化
├── data/
│   └── extended_aft/     # データセット（CSV）
├── outputs/
│   ├── lambda_experiments/  # Lambda実験結果
│   │   └── {data_name}/
│   │       └── lambda_{value}/
│   │           ├── config.toml
│   │           └── result.json
│   ├── lambda_summary.csv  # 集計結果
│   └── lambda_plots/       # 可視化プロット
├── config.toml           # デフォルト設定
├── lambda_grid.json      # Lambda並列実験用の値リスト
├── qsub.sh              # UGEジョブスクリプト
├── run_lambda_experiment.sh  # Lambda実験実行スクリプト
└── main.py              # CLIエントリポイント
```

## 設定ファイル（config.toml）

主要なハイパーパラメータ：

```toml
time_grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 時間分割
lambda_fuse = 1.0        # Fused Lasso強度
rho = 1.0                # ADMMペナルティ係数
max_admm_iter = 100      # ADMM最大反復数
clip_eta = 5.0           # exp(η)の発散抑制
n_baseline_basis = 8     # B-spline基底数

[quadrature]
Q = 5                    # 求積点数
rule = "gauss_legendre"  # 求積法
```

## テスト

```bash
# 全テスト実行
uv run pytest tests/

# 特定テスト実行
uv run pytest tests/test_admm_fit_smoke.py -v
```

## WandBロギング（オプション）

```bash
# 環境変数でWandBを有効化
export WANDB_PROJECT=admm-experiments
export WANDB_ENABLED=true

uv run main.py --config config.toml --data data/simulated_data.csv
```

## 実装状況

✅ 実装済み：
- ADMMソルバ本体
- 目的関数（近似対数尤度）の勾配・ヘッセ
- B-splineベースライン
- Fused Lasso正則化
- Lambda並列実験フレームワーク
- 結果集計・可視化スクリプト

🚧 今後の実装：
- Brier score など追加の評価指標
- 適応的ρ調整
- M/I-splineベースライン（積分の解析的計算）

## Ctd評価（time-dependent C-index）

本実装では、以下の定義に基づく `c_td` を評価指標として計算します。

$$
C^{td}
=
\Pr\!\left(
S(T_i \mid X_i(t)) < S(T_i \mid X_j(t))
\;\middle|\;
T_i < T_j,\; D_i = 1
\right)
$$

- 比較対象ペア: $T_i < T_j,\; D_i=1$
- 判定: $S(T_i \mid X_i(t)) < S(T_i \mid X_j(t))$
- 同値（tie）: 0.5 点

`main.py --output` の `result.json` では、`summary.c_td` として出力されます。

## 参考文献

- Pang et al. (2021). "Flexible Extension of the Accelerated Failure Time Model to Account for Nonlinear and Time-Dependent Effects of Covariates on the Hazard." *Statistical Methods in Medical Research*, 30(11), 2526–42.
- Boyd et al. (2011). "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." *Foundations and Trends in Machine Learning*, 3(1), 1–122.
