常に日本語で答えてください。

このリポジトリは、時間変動係数を持つ Hazard-AFT モデルを ADMM + fused lasso で推定し、シミュレーション・実データ CV・lambda 並列実験・ブートストラップ信頼区間まで回すための研究実装です。

## 現在の実装状況を優先する

古い設計メモより、現在のコードと最近のドキュメントを優先してください。

- 評価指標の主役は `c_td` です。`ADMMHazardAFT.score()` も現在は近似対数尤度ではなく `c_td` を返します。
- `docs/ctd-index.md` は `c_td` の定義の根拠です。Comparable pair は `(T_i < T_j, D_i = 1)`、concordant 条件は `S(T_i | X_i) < S(T_i | X_j)`、tie は 0.5 点、comparable pair が無ければ `NaN` です。
- lambda 並列実験は `docs/lambda_experiments.md`、実データ CV は `docs/real_cv.md` の運用に合わせます。
- `outputs/` や `data/` には実験生成物・大きな結果が多いので、必要がない限り触らないでください。特に未追跡の `outputs/real_cv/...` はユーザー作業物として扱い、削除・整理しないでください。

## 主要コマンド

- テスト: `uv run pytest`
- 主要な個別テスト:
  - `uv run pytest tests/test_evaluator_c_td.py`
  - `uv run pytest tests/test_admm_fit_smoke.py`
  - `uv run pytest tests/test_main_eval_data_cli.py`
  - `uv run pytest tests/test_real_cv_preprocessing.py`
  - `uv run pytest tests/test_bootstrap_parameter_ci_cli.py`
  - `uv run pytest tests/test_compute_cox_metrics_smoke.py`
- CLI 実行: `uv run main.py --config config.toml --data <long-format.csv> --output <result.json>`
- 依存は `pyproject.toml` と `uv.lock` で管理します。現状の `requires-python` は `>=3.14` です。

## データ形式

`main.py` とブートストラップは long format CSV を前提にします。

- 必須列: `id`, `k`, `time`, `event`
- 追加列: 特徴量列。`id`, `k`, `time`, `event`, `time_true`, `c1`, `c2` は特徴量から除外します。
- 各 `id` は `k = 0..K-1` のちょうど `K` 行を持つ必要があります。
- `main.py` 内では `X.shape == (n_subjects, K, p)`、`y.shape == (n_subjects, 2)` に変換します。
- `y[:, 0]` は観測時刻、`y[:, 1]` は event 0/1 です。
- `<data>.meta.json` に `time_grid` があれば、config の `time_grid` より優先されます。

予測 API は 2D `X.shape == (n, p)` も受け取り、その場合は全区間に同じ特徴量を繰り返します。3D 入力では `coef_` と同じ `K` が必要です。

## 公開 Estimator

### `ADMMHazardAFT`

責務:

- 入力検証、内部コンポーネント構築、ADMM ソルバ呼び出し、予測 API の提供
- `coef_`, `gamma_`, `z_`, `u_`, `rho_`, `history_`, `baseline_`, `time_partition_`, `quadrature_`, `objective_` など学習後属性の保持
- `from_config(config)` による TOML/JSON config からの構築
- `load_params_from_result_json()` による predict-only 用の復元

重要な `__init__` ハイパーパラメータ:

- モデル: `time_grid`, `baseline_basis`, `n_baseline_basis`, `baseline_knot_margin`
- 求積: `quadrature`
- 正則化/ADMM: `lambda_fuse`, `rho`, `max_admm_iter`, `admm_tol_primal`, `admm_tol_dual`, `admm_tol_rel`
- 停滞判定: `admm_stagnation_tol`, `admm_stagnation_patience`
- Newton/line search: `newton_steps_per_admm`, `max_newton_iter`, `newton_tol`, `line_search_max_steps`, `line_search_shrink`, `line_search_c1`
- 数値安定化: `return_best_iterate`, `clip_eta`, `random_state`

公開メソッド:

- `fit(X, y) -> self`
- `predict_survival_function(X, times=None)`
- `predict_cumulative_hazard(X, times=None)`
- `score(X, y) -> float`: 現在は `c_td`
- `predict_risk_score()` は未実装です。使う場合は実装とテストを追加してください。

sklearn 風の方針は維持します。`__init__` はハイパーパラメータ保存だけにし、副作用を入れないでください。

## 内部コンポーネント

クラス境界は次を維持します。

- `ADMMHazardAFT`
  - `FusedLassoADMMSolver`
    - `HazardAFTObjective`
      - `BaselineHazardModel` / `BSplineBaseline`
      - `TimePartition`
      - `QuadratureRule`

### `HazardAFTObjective`

- 近似負対数尤度、β/γ の勾配、ヘッセを提供します。
- ソルバが微分式を知らない構成を守ってください。
- `clip_eta` による `exp(eta)` の発散抑制は重要です。

### `BSplineBaseline`

- 現行の baseline は B-spline です。
- 将来 M/I-spline へ差し替えられるよう、baseline 表現は `BaselineHazardModel` のインターフェース内に閉じ込めます。
- `baseline_knot_margin` と実データの最大時刻から knot range が決まります。

### `TimePartition`

- `time_grid` を保持し、区間 index、区間積分範囲、`eta(beta, X)` を扱います。
- β は `(K, p)`、X は原則 `(n, K, p)` です。

### `QuadratureRule`

- 区間 `[a, b]` に対する求積点と重みを返します。
- 最小実装は Gauss-Legendre 固定でよいですが、既存 API を壊さないでください。

## ADMM ソルバ

`FusedLassoADMMSolver` は fused lasso 付き最適化を担当します。

- `Dβ` は行列を作らず `beta[1:] - beta[:-1]` として扱います。
- `z` 更新は soft-thresholding、`u` は scaled dual update です。
- β/γ 更新はブロック座標で、gamma 更新後に beta 更新を行います。
- Newton step は line search 付きです。特異ヘッセに対して damping/fallback を使っています。
- `return_best_iterate=True` の挙動を壊さないでください。

`history_` はデバッグと集計の契約です。少なくとも次のキーを維持してください。

- `objective`
- `neg_loglik`
- `primal_residual`
- `dual_residual`
- `rho`
- `primal_tolerance`
- `dual_tolerance`
- `objective_relative_change`
- `stagnation_count`
- `newton_steps`
- `beta_step_norm`
- `gamma_step_norm`
- `stopping_reason`
- `n_admm_iter`

収束判定では絶対許容誤差だけでなく `admm_tol_rel` を含む Boyd 型の許容誤差を使います。停滞判定は `admm_stagnation_tol` と `admm_stagnation_patience` に従います。

## 評価指標

### `HazardAFTEvaluator`

- `evaluate(model, X, y, times=None) -> {"c_td": float}`
- `compare(models, X, y, times=None) -> dict`
- `compute_c_td(...)` が定義式の本体です。

`times` を明示する場合は、全 event time を含める必要があります。含まれていない場合は `ValueError` になります。

近似対数尤度や BIC は実験集計では使いますが、`score()` の意味とは分けて扱ってください。

## CLI と result.json

### `main.py`

通常 fit:

```bash
uv run main.py \
  --config config.toml \
  --data data/extended_aft_step/example.csv \
  --output outputs/example_result.json
```

評価用 test data を別に渡す場合:

```bash
uv run main.py \
  --config config.json \
  --data train.csv \
  --eval-data test.csv \
  --output result.json
```

predict-only:

```bash
uv run main.py \
  --config config.toml \
  --data data.csv \
  --load-result outputs/support2_result.json \
  --predict-times 1.0,2.0,3.0 \
  --output prediction.json
```

fit の `result.json` は主に次を持ちます。

- `data_path`, `eval_data_path`
- `n_samples`, `n_eval_samples`, `n_features`, `feature_cols`
- `time_grid`, `coef`, `gamma`, `z_last`
- `history`
- `summary.objective_last`, `summary.neg_loglik_last`
- `summary.primal_residual_last`, `summary.dual_residual_last`
- `summary.stopping_reason`, `summary.n_admm_iter`
- `summary.c_td`, `summary.c_td_train`, `summary.c_td_test`
- `config`

この構造は集計スクリプトが読むため、キー名を変更する場合は集計・可視化・テストも同時に更新してください。

## Lambda 並列実験

`docs/lambda_experiments.md` に従います。

- lambda 値は `lambda_grid.json` の `lambda_values` で管理します。
- 生成は `uv run scripts/generate_lambda_grid.py`。
- 実行は `run_lambda_experiment.sh`。`SGE_TASK_ID` から `data_idx` と `lambda_idx` を決めます。
- qsub は `qsub.sh`。タスク数は `データ数 × lambda数` に合わせます。
- 出力先は `outputs/lambda_experiments/{data_name}/lambda_{value}/result.json`。
- 集計は `uv run scripts/aggregate_lambda_results.py --base-dir outputs/lambda_experiments --output outputs/lambda_summary.csv`。
- 可視化は `uv run scripts/visualize_lambda_results.py --summary outputs/lambda_summary.csv --output-dir outputs/lambda_plots`。

集計 CSV の重要列:

- `data_name`
- `lambda_fuse`
- `objective_last`
- `neg_loglik_last`
- `primal_residual_last`
- `dual_residual_last`
- `c_td`
- `n_params`: `z_last` の非ゼロ数
- `bic`
- `rho`, `max_admm_iter`, `clip_eta`
- `result_path`

BIC は `n_params` と `neg_loglik_last` から計算します。`z_tol` の既定は `1e-8` です。

## Cox 比較

`scripts/compute_cox_metrics.py` は ADMM と比較する CoxPH ベースラインを計算します。

- `lifelines.CoxPHFitter` を使います。
- `HazardAFTEvaluator` と互換の adapter で `c_td_cox` を計算します。
- Harrell の C-index は `c_index_harrell` です。
- `--lambda-summary` を渡すと ADMM の lambda 集計と結合し、`outputs/cox_vs_lambda_c_td.csv` のような比較表を作れます。
- `scripts/visualize_lambda_results.py --cox-summary <csv>` で `lambda_vs_c_td_with_cox.png` を生成します。

## 実データ CV

`docs/real_cv.md` に従います。Support2 と Framingham は同じ実行コードを使い、dataset 固有の raw -> base 変換だけ `scripts/real_cv/datasets.py` に分けます。

主要ファイル:

- `scripts/real_cv/datasets.py`: dataset 固有の前処理定義
- `scripts/real_cv/common.py`: fold 分割、train 基準の標準化、long format 化
- `scripts/real_cv/make_splits.py`: id 単位の fold 割当作成
- `scripts/real_cv/prepare_fold.py`: 1 fold の train/test CSV、config、meta 作成
- `scripts/real_cv/aggregate_results.py`: fold 別・lambda 別の集計
- `run_real_cv_experiment.sh`: `lambda_fuse × fold` の 1 task 実行
- `qsub_real_cv.sh`: SGE array job

運用の重要点:

- split は実行前に 1 回だけ作ります。
- fold 分割は id 単位で行います。必要に応じて event で層化します。
- 連続特徴量の標準化は train fold の平均・標準偏差だけで train/test に適用します。
- Framingham は `time_scale_max=8766.0`、Support2 は train fold の最大 `time_original` を基準に time を `time_grid` 範囲へスケールします。
- `lambda_grid.json` が 10 点、`N_FOLDS=5` なら qsub の task は 50 です。
- `SGE_TASK_ID` の対応は `lambda_idx = task_idx // n_folds`, `fold_idx = task_idx % n_folds` です。
- 出力先は `outputs/real_cv/{dataset}/{experiment_name}/lambda_{value}/fold_{xx}/` です。

代表コマンド:

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

Framingham は `qsub -v DATASET=framingham qsub_real_cv.sh` を使います。

## ブートストラップ信頼区間

`scripts/bootstrap_parameter_ci.py` は long-format CSV を被験者単位でリサンプリングし、`coef` と `gamma` の percentile 信頼区間を出します。

主要仕様:

- `--data`: long-format CSV
- `--config`: TOML/JSON config
- `--base-result`: 既存 `result.json` があれば点推定値を再 fit せず読みます
- `--n-bootstrap`, `--ci-level`, `--random-state`, `--n-jobs`
- 失敗 replicate は `failures` に記録して、成功分があれば集計を継続します
- JSON に NaN/inf を混ぜないよう `None` へ変換します
- CSV は `*_coef_ci.csv` と `*_gamma_ci.csv` を出します

実データ用の入口は `run_real_bootstrap_ci.sh` です。

- Support2 は `data/real/support/prepare_support2_inference.py` を先に実行します。
- `outputs/support2_result.json` や `outputs/framingham_result_bp.json` があれば `--base-result` として利用します。
- 出力例: `outputs/support2_bootstrap_ci.json`, `outputs/framingham_bootstrap_ci.json`

## WandB

WandB は任意依存です。

- `WANDB_PROJECT` または `WANDB_ENABLED=1` がある場合だけ有効化します。
- 未利用・未インストールでも学習は動くようにしてください。
- `history_` と summary metrics を外側から送る設計を維持します。

## 実装時の注意

- 既存の実験結果、未追跡の `outputs/`、raw data を勝手に削除・再生成しないでください。
- `result.json` のキー、`summary` のキー、CSV の列名は実験スクリプト間の契約です。変更する場合は関連スクリプトとテストをまとめて更新してください。
- `score()` は `c_td` です。ログ尤度系の指標を追加する場合は別名の metric として扱ってください。
- `c_td` の仕様変更は `docs/ctd-index.md` と `tests/test_evaluator_c_td.py` を同時に更新してください。
- real CV の dataset 固有処理は `scripts/real_cv/datasets.py` に閉じ込め、共通処理は `common.py` に置いてください。
- 実験スクリプトは SGE 環境とローカル 1 task 実行の両方を壊さないでください。
- `clip_eta`、line search、ADMM 停止判定、`return_best_iterate` は数値安定性のための重要部品です。周辺を変更したら smoke test だけでなく履歴キーと停止理由も確認してください。
- shell script の `UV_BIN` 既定値はスパコン環境向けです。ローカル実行時は `UV_BIN=$(which uv)` などで上書きできます。
