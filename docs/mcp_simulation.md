# Fused MCP 実装とシミュレーション

## 設定

`ADMMHazardAFT` は次の設定で罰則を切り替える。

```toml
fuse_penalty = "mcp" # "lasso" または "mcp"
mcp_gamma = 3.0
rho = 1.0
```

MCP では `mcp_gamma > 1` および `mcp_gamma * rho > 1` が必要である。
適応的 rho がこの曲率条件を破る減少更新は採用しない。`lambda_fuse` は従来と
同じ mean-loss scale で入力し、ソルバ内では `N * lambda_fuse` を MCP の lambda
として firm-thresholding と目的関数の両方に用いる。

結果 JSON には `fuse_penalty`、`mcp_gamma`、MCP を含む正則化目的関数、罰則値、
primal/dual residual、rho、`mcp_convexity_margin = rho - 1/mcp_gamma`、初期値の由来を
保存する。

## 単一 fit と初期値

ゼロ beta（標準初期値）からの MCP fit:

```bash
uv run python main.py \
  --config generation/pilot/mcp_config.toml \
  --data train.csv \
  --eval-data test.csv \
  --output result.json
```

既存の fused lasso 解または直前 lambda の解から開始する場合:

```bash
uv run python main.py \
  --config generation/pilot/mcp_config.toml \
  --data train.csv \
  --eval-data test.csv \
  --init-result previous_result.json \
  --output result.json
```

`--init-result` の `coef` と `gamma` が warm start に使われ、パスは
`initialization_source` に記録される。

## 同一条件での fused lasso / fused MCP 比較

対象シナリオは Oracle Grid、Fine-grid、Off-grid、Small、No-change である。
両手法で `generation/pilot/lambda_grid.json`、同一 seed、同一 fold を使う。

大きい lambda から小さい lambda へ各 fold 内で warm start する推奨手順:

```bash
./scripts/pilot/penalty_comparison.sh submit-cv-warm
./scripts/pilot/penalty_comparison.sh aggregate-cv
./scripts/pilot/penalty_comparison.sh submit-refit-warm
./scripts/pilot/penalty_comparison.sh aggregate-refit
./scripts/pilot/penalty_comparison.sh compare
```

各 lambda を独立 task として実行して初期値依存性を確認する場合、最初のコマンドを
次に置き換える。

```bash
./scripts/pilot/penalty_comparison.sh submit-cv
```

MCP の最大 lambda を同じ fold・lambda の fused lasso 解から開始し、その後 MCP の
直前解で warm start する別 run は、次のように lasso CV 結果を指定する。

```bash
PILOT_INITIAL_RESULT_BASE=outputs/pilot_penalty_comparison/lasso/cv \
PILOT_PENALTY_METHODS=mcp \
PILOT_PENALTY_OUTPUT_ROOT=outputs/pilot_penalty_comparison_lasso_initialized \
./scripts/pilot/penalty_comparison.sh submit-cv-warm
```

標準初期値 run とこの run の `initialization_source`、返却 objective、係数 RMISE、
検出変化点を対応比較することで、非凸解の初期値依存性を監査できる。

既定の出力先は `outputs/pilot_penalty_comparison/{lasso,mcp}/` であり、最終比較は
`outputs/pilot_penalty_comparison/comparison/` に保存される。主要ファイルは以下。

- `penalty_summary_by_scenario.csv`: シナリオ別の収束率、独立評価 `c_td`、係数 RMISE、変化点 precision/recall/F1、偽陽性数
- `penalty_pairs_converged.csv`: 同一シナリオ・seed における MCP − lasso の対応差
- `penalty_fit_metrics_all.csv`: 非収束も含む全 fit の監査表

Small では recall を維持しながら false positive と precision が改善するか、No-change
では `detected` と `false_positive` が減るかを主要評価とする。予測性能は独立評価
`c_td`、係数推定は RMISE で確認する。

## 小規模スモーク

既存のパイロット環境変数をそのまま使える。例えばデータ数と fold 数を絞った専用
ディレクトリを指定し、`PILOT_EXPECTED_DATASETS` も実数に合わせる。

```bash
PILOT_TRAIN_DIR=/path/to/smoke/train \
PILOT_EVAL_DIR=/path/to/smoke/eval \
PILOT_EXPECTED_DATASETS=10 \
PILOT_N_FOLDS=2 \
PILOT_PENALTY_OUTPUT_ROOT=outputs/pilot_penalty_smoke \
./scripts/pilot/penalty_comparison.sh submit-cv-warm
```
