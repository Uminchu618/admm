# Real-data cross-validation

実データの CV は qsub のアレイジョブで `lambda_fuse × fold` を並列実行する。
Support2 と Framingham は同じ実行コードを使い、raw CSV から base データを作る部分だけ `scripts/real_cv/datasets.py` で分ける。
`lambda_fuse` は既存の lambda 並列実験と同じく `lambda_grid.json` から読む。
`lambda_fuse` は平均損失スケールの値で、最適化時には train fold のサンプル数 `N` を掛けた
`N * lambda_fuse` が fused lasso 罰則に使われる。

## ディレクトリ

```text
scripts/real_cv/
  datasets.py            # dataset 固有の raw -> base 変換
  common.py              # fold 分割・標準化・long format 化
  make_splits.py         # id 単位の fold 割当を作成
  prepare_fold.py        # 1 fold の train/test CSV と config を作成
  aggregate_results.py   # result.json を集計

data/real/cv/splits/
  support2/support2_5fold_seed1234.csv
  framingham/framingham_5fold_seed1234.csv

outputs/real_cv/
  support2/support2_5fold_seed1234/lambda_1/fold_00/
    config.json
    fold_meta.json
    data/train.csv
    data/test.csv
    result.json
```

## Support2

`lambda_grid.json` が 10 点で `--n-folds 5` の場合、qsub の task 数は `10 * 5 = 50`。
`run_real_cv_experiment.sh` は `SGE_TASK_ID` を次のように割り当てる。

```text
task_idx   = SGE_TASK_ID - 1
lambda_idx = task_idx // n_folds
fold_idx   = task_idx %  n_folds
```

つまり、lambda ごとに fold 0..4 を回す。

```text
SGE_TASK_ID 1  -> lambda_grid[0], fold 0
SGE_TASK_ID 2  -> lambda_grid[0], fold 1
...
SGE_TASK_ID 5  -> lambda_grid[0], fold 4
SGE_TASK_ID 6  -> lambda_grid[1], fold 0
```

```bash
uv run scripts/real_cv/make_splits.py \
  --dataset support2 \
  --input data/real/support/support2.csv \
  --output data/real/cv/splits/support2/support2_5fold_seed1234.csv \
  --n-folds 5 \
  --random-state 1234

qsub qsub_real_cv.sh

uv run scripts/real_cv/aggregate_results.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234
```

集計時には、5 foldすべてが正式収束し、有限な検証 `c_td` を持つlambdaだけを
選択候補とする。平均検証 `c_td` が最大のlambdaを
`selected_lambda.json` に保存する。同点時は大きいlambdaを選ぶ。

CV 結果の可視化:

```bash
uv run scripts/real_cv/visualize_results.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234
```

同じ fold 分割で CoxPH baseline も重ねる場合:

```bash
uv run scripts/real_cv/compute_cox_baseline.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234

uv run scripts/real_cv/visualize_results.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234 \
  --cox-summary outputs/real_cv/support2/support2_5fold_seed1234/cox_summary.csv
```

同じ fold 分割で単純な parametric AFT baseline も比較する場合:

```bash
uv run scripts/real_cv/compute_aft_baseline.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234

uv run scripts/real_cv/visualize_results.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234 \
  --cox-summary outputs/real_cv/support2/support2_5fold_seed1234/cox_summary.csv \
  --aft-summary outputs/real_cv/support2/support2_5fold_seed1234/aft_summary.csv
```

既定では `WeibullAFTFitter`, `LogNormalAFTFitter`, `LogLogisticAFTFitter`
を fold ごとに推定し、`aft_fold_results.csv` と `aft_summary.csv` を保存する。
`--aft-models weibull` のように model を絞ることもできる。

既定では `fold_results.csv`、`summary_by_lambda.csv` と、次の図を
`outputs/real_cv/support2/support2_5fold_seed1234/plots/` に保存する。

- `cv_lambda_vs_c_td.png`: fold 点、平均線、標準誤差つきの test `c_td`
- `cv_train_test_c_td.png`: train/test `c_td` 平均の比較
- `cv_fold_spaghetti.png`: fold 別 test `c_td` の lambda 軌跡
- `cv_convergence_diagnostics.png`: ADMM iteration、残差、停止理由
- `cv_model_comparison.png`: `--aft-summary` 指定時の ADMM/Cox/AFT 比較

`summary_by_lambda.csv` の `cv_eligible` は選択候補としての適格性、
`selected` は最終選択を表す。不完全fold、未収束、非有限Ctdの理由は
`cv_exclusion_reason` に保存する。

ローカルで 1 task だけ確認する場合:

```bash
./run_real_cv_experiment.sh 1
```

## Framingham

```bash
uv run scripts/real_cv/make_splits.py \
  --dataset framingham \
  --input data/real/framingham/framingham.csv \
  --output data/real/cv/splits/framingham/framingham_5fold_seed1234.csv \
  --n-folds 5 \
  --random-state 1234

qsub -v DATASET=framingham qsub_real_cv.sh

uv run scripts/real_cv/aggregate_results.py \
  --base-dir outputs/real_cv/framingham/framingham_5fold_seed1234
```
