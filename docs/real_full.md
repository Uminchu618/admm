# 実データ全体での lambda 実験

CV ではなく、各 dataset の全 complete-case データを使って lambda ごとに再推定する手順です。出力は `outputs/real_full/{dataset}/{experiment_name}/lambda_{value}/` に保存します。

## スパコン実行

既定では Support2 と Framingham の両方を、`lambda_grid.json` の 10 点で実行します。

```bash
qsub qsub_real_full.sh
```

タスク対応は次の通りです。

- `DATASETS=support2,framingham`
- `SGE_TASK_ID=1..10`: Support2 の lambda 10 点
- `SGE_TASK_ID=11..20`: Framingham の lambda 10 点

lambda 数や dataset 数を変える場合は、`qsub_real_full.sh` の `#$ -t` も合わせて変更してください。

## 片方だけ実行する場合

`#$ -t` を `1-10:1` に変更してから投げます。

```bash
qsub -v DATASETS=support2 qsub_real_full.sh
qsub -v DATASETS=framingham qsub_real_full.sh
```

## 主な環境変数

- `UV_BIN`: uv のパス。既定は `/home/sagara/.local/bin/uv`
- `DATASETS`: カンマ区切り dataset。既定は `support2,framingham`
- `CONFIG_PATH`: ベース config。既定は `config.toml`
- `LAMBDA_GRID`: lambda grid JSON。既定は `lambda_grid.json`
- `EXPERIMENT_NAME`: 出力実験名。既定は `full_data`
- `OUTPUT_BASE_DIR`: 出力 root。既定は `outputs/real_full`
- `SUPPORT2_INPUT`: Support2 raw CSV
- `FRAMINGHAM_INPUT`: Framingham raw CSV

## 集計と BIC/lambda 図

ジョブ完了後に実行します。

```bash
uv run scripts/real_cv/aggregate_full_results.py \
  --base-dir outputs/real_full \
  --output outputs/real_full/full_summary.csv

uv run scripts/real_cv/visualize_full_results.py \
  --summary outputs/real_full/full_summary.csv \
  --output-dir outputs/real_full/plots
```

生成される主なファイル:

- `outputs/real_full/full_summary.csv`
- `outputs/real_full/plots/lambda_vs_bic.png`

BIC は `2 * neg_loglik_last + n_params * log(n_samples)` で計算します。`n_params` は `z_last` のうち `|z| > 1e-8` の数です。
