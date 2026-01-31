# Lambda並列実験の実装ガイド

## 概要

lambda_fuseパラメータを変えながら複数のデータセットで並列実験を行うフレームワークを実装しました。

## 実装したファイル

### 1. Lambda値の管理

- **`lambda_grid.json`**: 実験で使用するlambda値のリスト（対数スケール: 0.01～10の10点）
- **`scripts/generate_lambda_grid.py`**: lambda_grid.jsonを生成するユーティリティ

### 2. 実験実行スクリプト

- **`run_lambda_experiment.sh`**: SGE_TASK_IDに基づいて実験を実行
  - データファイルとlambda値の組み合わせを自動選択
  - 各実験用の一時configを生成（lambda_fuseを上書き）
  - 結果を構造化されたディレクトリに保存

### 3. ジョブ投入スクリプト

- **`qsub.sh`**: UGEアレイジョブスクリプト（修正済み）
  - タスク数を1000に設定（データ100件 × lambda10点）
  - `run_lambda_experiment.sh`を呼び出すように変更

### 4. 結果集計・可視化スクリプト

- **`scripts/aggregate_lambda_results.py`**: 全実験結果を収集してCSVに集約
- **`scripts/visualize_lambda_results.py`**: 集計結果を可視化（3種類のプロット）

### 5. ドキュメント

- **`README.md`**: 全体的な使い方を含む包括的なドキュメント（更新済み）

## 実験パターンの割り当て

SGE_TASK_IDから実験パターンを決定する仕組み：

```bash
# パターン数 = データ数 × lambda数
total_patterns = n_data * n_lambda

# task_idから各インデックスを計算（1-based → 0-based）
task_idx = SGE_TASK_ID - 1
data_idx = task_idx / n_lambda
lambda_idx = task_idx % n_lambda
```

### 例

- データ数: 100
- Lambda数: 10
- 総パターン数: 1000

| SGE_TASK_ID | データインデックス | Lambdaインデックス | データ名 | Lambda値 |
|-------------|-------------------|-------------------|---------|---------|
| 1 | 0 | 0 | data_0001.csv | 0.01 |
| 2 | 0 | 1 | data_0001.csv | 0.0215 |
| 10 | 0 | 9 | data_0001.csv | 10.0 |
| 11 | 1 | 0 | data_0002.csv | 0.01 |
| 101 | 10 | 0 | data_0011.csv | 0.01 |

## ディレクトリ構造

```
outputs/lambda_experiments/
├── data_0001/
│   ├── lambda_0.01/
│   │   ├── config.toml      # lambda_fuse=0.01で上書きしたconfig
│   │   └── result.json      # 実験結果
│   ├── lambda_0.0215/
│   │   ├── config.toml
│   │   └── result.json
│   └── ...
├── data_0002/
│   └── ...
└── ...
```

## 使い方

### 1. Lambda値の準備

```bash
# デフォルト設定で生成（0.01～10の対数10点）
python scripts/generate_lambda_grid.py

# カスタム設定
python scripts/generate_lambda_grid.py --min 0.001 --max 100 --n-points 20
```

### 2. ジョブ投入

```bash
# スパコンにジョブ投入
qsub qsub.sh

# ジョブ状態確認
qstat
```

### 3. ローカルテスト

```bash
# 特定のタスクIDで実行
./run_lambda_experiment.sh 1

# 複数タスクをシーケンシャルに実行
for i in {1..10}; do
  ./run_lambda_experiment.sh $i
done
```

### 4. 結果の集計

```bash
# 全結果を集計してCSV生成
uv run scripts/aggregate_lambda_results.py \
  --base-dir outputs/lambda_experiments \
  --output outputs/lambda_summary.csv
```

### 5. 結果の可視化

```bash
# 可視化プロット生成
uv run scripts/visualize_lambda_results.py \
  --summary outputs/lambda_summary.csv \
  --output-dir outputs/lambda_plots
```

## 生成されるプロット

1. **`lambda_vs_objective.png`**
   - Lambda値と目的関数の関係
   - 各データファイルごとに線グラフ
   - X軸はlogスケール

2. **`lambda_distribution.png`**
   - Lambda値ごとの目的関数分布
   - 箱ひげ図で全データの分布を表示

3. **`lambda_vs_convergence.png`**
   - Lambda値と収束状況の関係
   - Primal residualとDual residualの2つのサブプロット

## 集計結果CSVの列

| 列名 | 説明 |
|------|------|
| data_name | データファイル名 |
| lambda_fuse | Lambda値 |
| n_samples | サンプル数 |
| n_features | 特徴量数 |
| objective_last | 最終目的関数値 |
| primal_residual_last | 最終primal残差 |
| dual_residual_last | 最終dual残差 |
| rho | ADMMペナルティ係数 |
| max_admm_iter | ADMM最大反復数 |
| clip_eta | exp(η)クリップ幅 |
| result_path | 結果JSONの相対パス |

## 設計のポイント

### 1. 再現性の確保

- Lambda値は`lambda_grid.json`で管理（実験後も確認可能）
- 各実験のconfigを保存（全ハイパーパラメータを記録）
- random_stateはconfig.tomlで固定

### 2. 集計の容易性

- 結果は構造化されたディレクトリに保存
- JSONフォーマットで機械可読
- CSVへの集約スクリプトを提供

### 3. スケーラビリティ

- データ数とlambda数を変えるだけで自動的にタスク数が決まる
- SGE_TASK_IDベースの割り当てで衝突なし

### 4. 柔軟性

- Lambda値の範囲・点数は簡単に変更可能
- run.shは既存実験用に残し、新しいスクリプトを追加

## トラブルシューティング

### ジョブが範囲外エラーで失敗する

```bash
# lambda_grid.jsonの値数を確認
jq '.lambda_values | length' lambda_grid.json

# データファイル数を確認
ls data/extended_aft/*.csv | wc -l

# qsub.shのタスク数を調整
# 総タスク数 = データ数 × lambda数
```

### 一部の結果が欠損している

```bash
# 失敗したタスクを特定
uv run scripts/aggregate_lambda_results.py | grep "Warning"

# 個別に再実行
./run_lambda_experiment.sh <task_id>
```

### メモリ不足

```bash
# qsub.shでメモリ要求を増やす
#$ -l s_vmem=8G  # 4G → 8G など
```

## 今後の拡張

- Cross-validation用のデータ分割機能
- ハイパーパラメータ探索（rho、clip_etaなども並列化）
- WandB統合による実験管理
- 評価指標の自動計算・集計
