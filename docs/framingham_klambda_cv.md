# FraminghamのK・lambda同時探索

既存の `framingham_5fold_seed1234.csv` のid単位5-foldを固定して、
Kとlambdaの全組合せを比較する。罰則の既定は `config.toml` のLasso。
`framingham_klambda_grid.json` の初期候補はK = 1, 3, 6, 12, 24と
lambda = 0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 1, 3, 10。
5 × 12 × 5 = **300 task**。各taskは独立した初期値で学習する。
K=1ではfused罰則が空なのでlambdaを変えても同じモデルとなる。

時間範囲は `[0, 6]` に固定し、8766日を6へスケーリングする。
K区間の等間隔 `time_grid` は実行時に生成する。
各foldの標準化はtrainのみから計算し、1人1行train/testを一度だけ保存する。
全候補が同じ被験者・観測時間・共変量・foldを使う。
既存CSVやLasso/MCP実験結果は書き換えない。

## スパコンでの実行

リポジトリのルートで以下を実行する。Python依存は既存のuv環境を使う。
`UV_BIN` の既定は `/home/sagara/.local/bin/uv`。

```bash
# 一度だけ前処理し、設定・候補・データハッシュをmanifestへ保存
bash scripts/real_cv/klambda_workflow.sh prepare

# 300 taskを投入（最大同時実行数は既定50）
bash scripts/real_cv/klambda_workflow.sh submit --max-concurrent 50

# 実行中も集計可能。全taskが揃うまでは暫定選択のみ
bash scripts/real_cv/klambda_workflow.sh aggregate

# 全task完了後、同じ集計コマンドでK・lambdaを確定
bash scripts/real_cv/klambda_workflow.sh aggregate

# 確定した1組だけを全データで再学習（1ジョブ）
bash scripts/real_cv/klambda_workflow.sh submit-refit
```

`submit` はmanifestから `qsub -t 1-N` を自動計算する。
taskの対応はfoldが最も速く変わり、次にlambda、最後にKが変わる。

```text
task_idx = SGE_TASK_ID - 1
K_idx = task_idx // (n_lambda * n_folds)
lambda_idx = (task_idx // n_folds) % n_lambda
fold_idx = task_idx % n_folds
```

prepareは既存の探索ディレクトリがあると停止する。
別の候補・設定で実験する場合は、新しい `SEARCH_DIR` を指定してprepareから開始する。
実行は保存したmanifestの設定を使うため、元のconfig/gridを後から編集しても
既に準備した探索には反映されない。準備したデータの変更はハッシュで検出する。

MCPを使う場合は独立した探索先へ準備する。

```bash
export SEARCH_DIR="$PWD/outputs/real_cv/framingham/mcp_klambda_5fold_seed1234"
CONFIG_PATH="$PWD/config_real_mcp.toml" bash scripts/real_cv/klambda_workflow.sh prepare
bash scripts/real_cv/klambda_workflow.sh submit
```

候補ファイルは `SEARCH_GRID`、rawデータは `FRAMINGHAM_INPUT`、
fold割当は `SPLITS_FILE`、fold数は `N_FOLDS` で変更できる。
fold数を変える場合は対応するsplitファイルも指定する。

## 集計・選択

```text
outputs/real_cv/framingham/lasso_klambda_5fold_seed1234/
  manifest.json
  prepared/fold_00/train.csv, test.csv, fold_meta.json
  prepared/all.csv, full_meta.json
  K_06/lambda_0.01/fold_00/config.json, result.json, runtime.json
  fold_results.csv
  summary_by_K_lambda.csv
  selected_params.json
  K_lambda_vs_c_td.png
  selected_full/result.json, selection.json
```

全5foldで正式収束し、有限な検証 `c_td` を持つ組合せだけを適格とする。
その中で平均検証 `c_td` 最大の組合せを選ぶ。同点（許容差1e-12）では
小さいK、次に大きいlambdaを優先する。
未完了では `selection_method=pending_cv_completion` とし、
`selected_K`・`selected_lambda` はnull、暫定値は `provisional_*` に記録する。
全taskが揃っても適格候補がなければ `no_eligible_candidate` と記録し、
再学習は停止する。非収束を収束扱いに変更せず、診断して追加実験する。
壊れたJSON、設定の不一致、想定外の結果は集計を停止して知らせる。

`c_td` は各foldの観測時刻で評価する。time_gridの端点で評価時刻を置き換えない。
図は適格候補の平均と標準誤差をKごとに表示する。
この平均はハイパーパラメータ選択用のCV値であり、独立した最終性能評価ではない。
最終性能を報告する場合は別のテストデータかnested CVを用意する。
最大Kやlambda候補の端で最適値が出た場合は、候補範囲を広げた別実験で確認する。

## ローカル確認・再開

```bash
UV_BIN=$(command -v uv) bash scripts/real_cv/klambda_workflow.sh run --task-id 1
UV_BIN=$(command -v uv) bash scripts/real_cv/klambda_workflow.sh run --task-id 1 --skip-existing
```

qsubは `--skip-existing` で実行する。完成したresultは設定を検査してスキップする。
失敗したtaskは同じtask番号で再投入できる。正常終了まで `result.partial.json` を使う。
二重起動は `.running` で防ぐ。強制終了でlockが残った場合は、ジョブが終了したことを
確認してからそのtaskのlockだけを手動で除去する。
