# Framingham / SUPPORT2 の fused MCP 実データ CV

`data/real/cv/splits/{framingham,support2}/` にある 5-fold, seed 1234 の分割を使う。
候補 lambda はシミュレーションと同じ `generation/pilot/lambda_grid.json` の9点
（0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25）。
設定は `config_real_mcp.toml`（MCP, gamma=3, rho=1）。

## 標準実行: 全 lambda × fold を独立並列

```bash
./scripts/real_cv/mcp_workflow.sh submit framingham
./scripts/real_cv/mcp_workflow.sh submit support2
```

各 dataset で 9 lambda × 5 fold = 45 array task、合計90 fit。
`qsub` の `-t` は grid の候補数から自動計算する。
既存の Lasso 実験には書き込まず、以下に保存する。

```text
outputs/real_cv/framingham/mcp_5fold_seed1234/lambda_0.25/fold_00/result.json
outputs/real_cv/support2/mcp_5fold_seed1234/lambda_0.25/fold_00/result.json
```

ジョブ完了後、各 dataset で順に実行する。

```bash
./scripts/real_cv/mcp_workflow.sh aggregate framingham
./scripts/real_cv/mcp_workflow.sh baselines framingham
./scripts/real_cv/mcp_workflow.sh visualize framingham

./scripts/real_cv/mcp_workflow.sh aggregate support2
./scripts/real_cv/mcp_workflow.sh baselines support2
./scripts/real_cv/mcp_workflow.sh visualize support2
```

集計時は存在する `result.json` のMCP・lambda設定を検査する。
実行中のタスクが残っていても `fold_results.csv` と `summary_by_lambda.csv` を
暫定出力し、`selected_lambda.json` は `pending_cv_completion` と記録する。
この時点で5 foldすべてが揃い、収束・有限test `c_td` の条件を満たすlambdaだけから
暫定lambdaを選び、`provisional_lambda` に記録する。欠けたfoldを持つlambdaは
暫定選択の候補に含めない。
全45件が揃ったら同じ `aggregate` コマンドを再実行してlambdaを確定する。
設定不整合や想定外の結果ファイルがあれば集計を停止する。
lambda選択は全5 foldが収束し有限の test `c_td` を持つ候補の平均 test `c_td`
最大値を使い、同点なら大きいlambdaを選ぶ。選択に独立評価データやBICは使わない。
`baselines` は同じ fold の train/test CSV で CoxPH を推定する。
未完了でも `visualize` を実行でき、その時点で利用可能なfoldによる暫定図を
`plots_partial/` に保存する。暫定lambdaがあればCV性能図に印を付け、
`plots_partial/selected_beta/` にその係数軌跡を描く。
`visualize` は現在の結果から集計表を更新するため、ジョブの進行中に再実行できる。
全45件が揃ったら `visualize` を再実行し、lambda選択と最終図を `plots/` に作る。

主な生成物:

- `fold_results.csv`, `summary_by_lambda.csv`, `selected_lambda.json`
- `cox_fold_results.csv`, `cox_summary.csv`
- `plots/cv_lambda_vs_c_td.png`: fold点・平均・標準誤差・選択lambda・Cox基準線
- `plots/cv_train_test_c_td.png`, `plots/cv_fold_spaghetti.png`, `plots/cv_convergence_diagnostics.png`
- `plots/beta_by_lambda/`: 各lambdaで5 foldの係数軌跡を重ねた図
- `plots/selected_beta/`: 選択lambdaの係数軌跡

lambda=0を含む広いグリッドは図の横軸をsymlog表示にする。
選択lambdaが最大候補0.25なら、探索範囲の境界に達したと解釈して報告する。

## 選択lambdaで全データ再推定

```bash
./scripts/real_cv/mcp_workflow.sh submit-refit framingham
./scripts/real_cv/mcp_workflow.sh submit-refit support2
# 各1 taskの完了後
./scripts/real_cv/mcp_workflow.sh plot-refit framingham
./scripts/real_cv/mcp_workflow.sh plot-refit support2
```

全データfitは `outputs/real_full/{dataset}/mcp_5fold_seed1234_selected_full/`
に、最終係数図はCV実験の `plots/selected_lambda_full_beta.png` に保存する。

## 後日の初期値感度検証: 降順 warm start

標準実験と別の出力ディレクトリに保存する。

```bash
./scripts/real_cv/mcp_workflow.sh submit-warm framingham
./scripts/real_cv/mcp_workflow.sh submit-warm support2
```

各 dataset で5 array task（1 task = 1 fold）。タスク内では
0.25 → 0.1 → 0.03 → 0.01 → 0.003 → 0.001 → 0.0003 → 0.0001 → 0
の順に前の結果を `--init-result` へ渡す。`result.json` の
`initialization_source` で初期値を確認できる。
出力先は `outputs/real_cv/{dataset}/mcp_5fold_seed1234_warm/`。
再開時は `SKIP_EXISTING=1` を `qsub -v` またはローカル実行時に指定する。

スパコン以外では `UV_BIN=$(command -v uv)` を指定できる。
標準実験の1タスクのみローカルで確認する場合は以下。

```bash
UV_BIN=$(command -v uv) DATASET=framingham \
  CONFIG_PATH="$PWD/config_real_mcp.toml" \
  LAMBDA_GRID="$PWD/generation/pilot/lambda_grid.json" \
  EXPERIMENT_NAME=mcp_5fold_seed1234 \
  ./run_real_cv_experiment.sh 1
```
