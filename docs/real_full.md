# CV選択lambdaによる実データ全体の再学習

5-fold CVで平均検証 `c_td` が最大となったlambdaを使い、各datasetの全
complete-caseデータで1回だけ再推定する。出力は
`outputs/real_full/{dataset}/{experiment_name}/lambda_{value}/` に保存する。

## スパコン実行

先に各datasetのCV集計を実行し、`selected_lambda.json` を作成する。

```bash
uv run scripts/real_cv/aggregate_results.py \
  --base-dir outputs/real_cv/support2/support2_5fold_seed1234
```

既定では Support2 と Framingham の選択lambdaを各1 taskで再学習する。

```bash
qsub qsub_real_full.sh
```

タスク対応は次の通りです。

- `DATASETS=support2,framingham`
- `SGE_TASK_ID=1`: Support2
- `SGE_TASK_ID=2`: Framingham

dataset数を変える場合は、`qsub_real_full.sh` の `#$ -t` も合わせて変更する。

## 片方だけ実行する場合

`#$ -t` を `1-1:1` に変更してから投げる。

```bash
qsub -v DATASETS=support2 qsub_real_full.sh
qsub -v DATASETS=framingham qsub_real_full.sh
```

## 主な環境変数

- `UV_BIN`: uv のパス。既定は `/home/sagara/.local/bin/uv`
- `DATASETS`: カンマ区切り dataset。既定は `support2,framingham`
- `CONFIG_PATH`: ベース config。既定は `config.toml`
- `LAMBDA_SELECTION_MODE`: 既定は `cv`。過去の全lambda実験は `grid`
- `CV_OUTPUT_BASE_DIR`: CV出力root。既定は `outputs/real_cv`
- `N_FOLDS`, `SPLIT_SEED`: 選択JSONの実験名解決に使用
- `EXPERIMENT_NAME`: 出力実験名。既定は `cv_selected_full`
- `OUTPUT_BASE_DIR`: 出力 root。既定は `outputs/real_full`
- `SUPPORT2_INPUT`: Support2 raw CSV
- `FRAMINGHAM_INPUT`: Framingham raw CSV

## 互換用の全lambda/BICモード

過去のBIC比較を再現する場合だけ、`LAMBDA_SELECTION_MODE=grid` を指定する。

```bash
uv run scripts/real_cv/aggregate_full_results.py \
  --base-dir outputs/real_full \
  --output outputs/real_full/full_summary.csv

uv run scripts/real_cv/visualize_full_results.py \
  --summary outputs/real_full/full_summary.csv \
  --output-dir outputs/real_full/plots
```

このモードでは従来どおり、全lambdaの集計とBIC図を生成できる。

- `outputs/real_full/full_summary.csv`
- `outputs/real_full/plots/lambda_vs_bic.png`

BICは主解析のlambda選択には使用しない。
