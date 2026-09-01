# Lambda並列実験の実装ガイド

## 概要

lambda_fuseパラメータを変えながら複数のデータセットで並列実験を行うフレームワークを実装しました。

`lambda_fuse` は平均損失スケールの正則化強度として扱います。尤度部分は実装上
サンプル和 `negative log-likelihood` なので、最適化では
`negative log-likelihood + N * lambda_fuse * ||Dβ||_1` を解きます。
これは `(1/N) * negative log-likelihood + lambda_fuse * ||Dβ||_1` と同じ解です。

## 実装したファイル

### 1. Lambda値の管理

- **`lambda_grid.json`**: 実験で使用するlambda値のリスト（0 + 対数スケール: 1～10の合計10点）
- **`scripts/generate_lambda_grid.py`**: lambda_grid.jsonを生成するユーティリティ

### 2. 実験実行スクリプト

- **`run_lambda_experiment.sh`**: SGE_TASK_IDに基づいて実験を実行
  - データファイルとlambda値の組み合わせを自動選択
  - 同名の独立評価データを `--eval-data` で指定
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
| 1 | 0 | 0 | data_0001.csv | 0.0 |
| 2 | 0 | 1 | data_0001.csv | 1.0 |
| 10 | 0 | 9 | data_0001.csv | 10.0 |
| 11 | 1 | 0 | data_0002.csv | 0.0 |
| 101 | 10 | 0 | data_0011.csv | 0.0 |

## ディレクトリ構造

```
outputs/lambda_experiments/
├── data_0001/
│   ├── lambda_0/
│   │   ├── config.toml      # lambda_fuse=0.0で上書きしたconfig
│   │   └── result.json      # 実験結果
│   ├── lambda_1/
│   │   ├── config.toml
│   │   └── result.json
│   └── ...
├── data_0002/
│   └── ...
└── ...
```

## 使い方

### 0. 学習・独立評価データの生成

```bash
uv run generation/generate_extended_aft_step_datasets.py \
  --output-dir data/extended_aft_step \
  --eval-output-dir data/extended_aft_step_eval
```

両ディレクトリには同名のCSVが作られ、評価データには既定で
`train seed + 100000` を使います。対応する評価CSVがない場合、
`run_lambda_experiment.sh` は学習時評価へフォールバックせずエラー終了します。

### 1. Lambda値の準備

```bash
# デフォルト設定で生成（0 + 1～10の対数9点、合計10点）
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

BIC は、正式な Boyd 型残差判定を満たした `bic_eligible=true` の結果だけで
計算します。`max_iter`、`stagnated`、`invalid_state` は結果JSONへ残しますが、
モデル選択候補には含めません。候補がないデータセットではBIC選択不能として扱います。

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
| lambda_fuse_effective | 最適化で使った実効値（n_samples * lambda_fuse） |
| n_samples | サンプル数 |
| n_eval_samples | 独立評価データのサンプル数 |
| n_features | 特徴量数 |
| objective_last | 最終目的関数値 |
| returned_iter | `coef` と `z_last` が対応する0始まりの反復番号 |
| returned_neg_loglik | 返却反復の負の対数尤度 |
| returned_primal_residual | 返却反復のprimal残差 |
| returned_dual_residual | 返却反復のdual残差 |
| returned_primal_tolerance | 返却反復のprimal許容誤差 |
| returned_dual_tolerance | 返却反復のdual許容誤差 |
| converged | 正式な残差判定を満たしたか |
| bic_eligible | BIC選択候補として使用できるか |
| primal_residual_last | 最終primal残差 |
| dual_residual_last | 最終dual残差 |
| c_td | 独立評価データ上の time-dependent C-index |
| c_td_train | 学習データ上の time-dependent C-index |
| c_td_test | 独立評価データ上の time-dependent C-index |
| n_change_points | `z_last` の非ゼロ要素数 |
| n_params | `n_baseline_basis + n_features + n_change_points` |
| bic | `2 * NLL + n_params * log(n_samples)` |
| rho | ADMMペナルティ係数 |
| max_admm_iter | ADMM最大反復数 |
| clip_eta | exp(η)クリップ幅 |
| result_path | 結果JSONの相対パス |

## パイロット診断実験

既存の `outputs/pilot` と分離して、Oracle・Fine-grid各3 seed、small lambda
9点の54タスクを実行できます。既定設定は適応的rhoと
`newton_steps_per_admm=5`です。

```bash
./scripts/pilot/submit_diagnostic.sh
./scripts/pilot/aggregate_diagnostic.sh
```

出力先は
`outputs/pilot_diagnostic/adaptive_rho_normalized_stagnation_escape_newton5/`
です。
集計後には `check_diagnostic.py` が54件の正式収束、返却残差、BIC候補、
正則化経路の変化を検査し、不合格なら終了コード1を返します。

適応的rhoは主・双対残差を各停止許容誤差で正規化して比較します。
rhoを更新した反復では停滞カウントをリセットし、更新直後の早期停止を防ぎます。
停滞上限へ達した場合は通常の更新周期外でも一度rho balancingを試し、rhoを
変更できた場合は反復を継続します。rhoを変更できない場合だけ停滞停止します。
固定rhoやNewtonステップ数を比較するときは、別のrun名を必ず指定して結果を分離します。

```bash
PILOT_DIAGNOSTIC_RUN=fixed_rho10_newton5 \
DIAGNOSTIC_RHO=10 \
DIAGNOSTIC_ADAPTIVE_RHO=false \
DIAGNOSTIC_NEWTON_STEPS=5 \
SGE_TASK_ID=1 \
./scripts/pilot/run_diagnostic_task.sh
```

## 診断通過後の本パイロット

第3次診断は54/54件で正式収束し、自動ゲートを通過しました。本パイロットは
Oracle、Fine-grid、Off-grid、Small、No-changeを各20反復、9 lambdaで実行するため、
合計タスク数は `5 * 20 * 9 = 900` です。

既定値は、診断で合格した次の条件へ固定されています。

- lambda: `0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25`
- `adaptive_rho = true`
- `newton_steps_per_admm = 5`
- `rho_update_interval = 5`
- 停滞時の周期外rho balancingを有効化

リモートでデータを生成した後、次の順に実行します。

```bash
./scripts/pilot/generate_data.sh
./scripts/pilot/submit.sh
qstat
./scripts/pilot/aggregate.sh
uv run scripts/pilot/visualize_results.py
```

`submit.sh` は100個の学習CSVと100個の独立評価CSV、9個のlambdaを検証し、
SGEアレイ範囲を `1-900` として動的に指定します。出力は旧パイロットを上書きせず、
次へ保存されます。

```text
outputs/pilot/adaptive_rho_normalized_stagnation_escape_newton5/
outputs/pilot/adaptive_rho_normalized_stagnation_escape_newton5_summary.csv
outputs/pilot/adaptive_rho_normalized_stagnation_escape_newton5_gate.json
outputs/pilot/adaptive_rho_normalized_stagnation_escape_newton5_visualizations/
```

別名で再実行する場合は、投入・集計・可視化で同じ名前または明示パスを使います。

```bash
PILOT_RUN_NAME=my_run ./scripts/pilot/submit.sh
PILOT_RUN_NAME=my_run ./scripts/pilot/aggregate.sh
uv run scripts/pilot/visualize_results.py \
  --summary outputs/pilot/my_run_summary.csv \
  --output-dir outputs/pilot/my_run_visualizations
```

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
