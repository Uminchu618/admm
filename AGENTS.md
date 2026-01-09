## 設計方針
必須
間違っている箇所を除き、slide資料の式そのままに実装する。
docs\slides14.md
docs\slides15.md
高速化のための実装は、まだ実装しない。
まずは丁寧に答えがでるところまで実装してから高速化に取り組む

1. **Estimator API は 1 クラスに集約**
   `fit / predict` を提供し、ハイパーパラメータは `__init__` 引数に置く（sklearn流）。
2. **ADMM の状態（z,u,ρなど）は “学習後属性” として保持**
   sklearn の規約に合わせて末尾 `_` を付ける。
3. **目的関数・微分・積分近似は内部コンポーネントに分離**
   将来、B-spline→M/I-spline（積分不要化の検討）へ差し替えやすくする

---

## 1) 外側：Estimator 本体（sklearn 互換の“顔”）

### `ADMMHazardAFT`（BaseEstimator 互換）

**責務**

* 入力検証、学習ループ呼び出し、推論 API の提供
* `coef_` / `baseline_` / `history_` など学習結果の保持
* sklearn clone / GridSearch に耐える（`__init__` のみがハイパラ）

**主な `__init__` 引数（ハイパーパラメータ）**

* モデル構造：

  * `time_grid`: `(t0,...,tK)`（区間端点。β(t)の区分一定を決める）
  * `include_intercept: bool`（β_{0k} を持つか）
  * `baseline_basis`: `"bspline"` or `"mspline5"`（将来差し替え前提）
  * `n_baseline_basis: int`（B-spline の M）
* 近似：

  * `quadrature: {"Q": int, "rule": "gauss_legendre" | "simpson" ...}`（式(6)のQ）
* 正則化とADMM：

  * `lambda_fuse: float`（式(2)）
  * `rho: float`
  * `max_admm_iter: int`
  * `admm_tol_primal, admm_tol_dual: float`
  * `newton_steps_per_admm: int`（inexact Newton：ADMM内で1回/数回で止める、という方針）
* 収束/数値安定：

  * `max_newton_iter, newton_tol`
  * `clip_eta: float`（exp(η) の発散抑制）
  * `random_state`（初期値用）

**学習後属性（末尾 `_`）**

* `coef_`: shape `(K, p)`（もしくは `(K, p+1)`：intercept込み）
* `gamma_`: baseline スプライン係数（あなたの記法の γ）
* `time_grid_`, `n_features_in_`, `feature_names_in_`（可能なら）
* ADMM 状態：

  * `z_`: fused lasso 用補助変数（各 j の z_j、式(4)）
  * `u_`: scaled dual（各 j の u_j、式(5)）
  * `rho_`: 実効ρ（適応ρをやるなら更新後の値）
* `history_`: 目的関数、primal/dual residual、Newton step など（デバッグ必須）

**公開メソッド**

* `fit(X, y)`

  * y は sklearn 風にするなら `(time, event)` の2列を推奨（例：`y[:,0]=T, y[:,1]=delta`）
* `predict_survival_function(X, times=None)`

  * 生存関数 S(t|X)
* `predict_cumulative_hazard(X, times=None)`

  * Λ(t|X)
* `predict_risk_score(X, time=None)`

  * η_i(t) ないし平均リスク（スコア用途）
* `score(X, y)`

  * 既定は（負の）近似対数尤度 `log \tilde{L}` を返すのが筋（CVで使える）

---

## 2) 内部：目的関数（対数尤度・勾配・ヘッセ）

### `HazardAFTObjective`

**責務**

* あなたの定義する `log \tilde{L}`（区分求積近似）を計算し、勾配・ヘッセを返す

  * 対数尤度（式(6)）
  * β 勾配・ヘッセ（式(12)(13)）
  * γ 勾配・ヘッセ（式(14)(15)）

**持つべき依存（コンストラクタ注入）**

* `baseline: BaselineHazardModel`
* `time_partition: TimePartition`
* `quadrature: QuadratureRule`

**メソッド（内部用）**

* `value(beta, gamma, X, T, delta) -> float`（= -log \tilde{L} でもよい）
* `grad(beta, gamma, ...) -> (g_beta, g_gamma)`
* `hess(beta, gamma, ...) -> (H_bb, H_bg, H_gg)`
  ※ 最小実装なら **ブロック対角近似**（H_bg無視）にして Newton を軽くする選択も現実的

---

## 3) 内部：ベースラインハザードの表現

あなたの現状案は B-spline（log hazard を spline）で、積分は求積で近似する方針。 
将来、M/I-spline（積分を解析的に）へ移行する可能性があるので、**ここはインターフェースを切るのが最重要**です。

### `BaselineHazardModel`（抽象）

**責務**

* `S_m(x)`（基底値）と必要なら `S'_m(x), S''_m(x)` を返す（あなたの微分式に相当）
* 実装により「log hazard」「hazard」をどちらで表すかは隠蔽

**インターフェース**

* `basis(x) -> (n, M)`
* `basis_deriv(x) -> (n, M)`（必要なら）
* `basis_second_deriv(x) -> (n, M)`（必要なら）

### 実装

* `BSplineBaseline`（現行案）

---

## 4) 内部：時間分割と η の扱い

### `TimePartition`

**責務**

* `time_grid` を保持し、各個体の `k(i)` と、区間ごとの積分範囲 `[a_{ik}, b_{ik}]` を生成

  * あなたの記法の `a_{ik}=t_{k-1}`, `b_{ik}=min(T_i,t_k)` に一致
* β を `(K, p)` で持つ前提で、η_{ik} を一括生成

**メソッド**

* `interval_index(T) -> k(i)`
* `iter_intervals(T) -> list[(k, a, b)]`
* `eta(beta, X) -> (n, K)` （ベクトル化が効くと速度が出る）

---

## 5) 内部：求積（Quadrature）

### `QuadratureRule`

**責務**

* 各 `(a,b)` に対し求積点 `v_{ikℓ}` と重み `w_{ikℓ}` を返す（式(6)の近似）

**メソッド**

* `nodes_weights(a, b) -> (v: (Q,), w: (Q,))`

最小実装なら、ガウス・ルジャンドル固定で十分です（外部依存を減らせる）。

---

## 6) 内部：ADMM ソルバ（ここを単体テスト可能にする）

### `FusedLassoADMMSolver`

**責務**

* ADMM 反復（式(3)(4)(5)）を回し、β・γ・z・u を更新する
* `β,γ` 更新は「Objective + inexact Newton」を呼ぶだけにする（ソルバが微分を知らない）

**インターフェース**

* `solve(beta0, gamma0, X, T, delta) -> (beta, gamma, z, u, history)`

**内部の更新**

* `(β, γ)` 更新：

  * 目的関数 `F(β,γ) = -log\tilde{L}(β,γ) + (ρ/2)∑||Dβ_j - z_j + u_j||^2` を最小化 
  * 実装簡略化のため、資料の方針通り **(1) γ 更新 → (2) β 更新**のブロック座標（damped Newton）にすると楽（式の整備もできている）
* `z` 更新（prox）：soft-thresholding（一般化ラッソのproxと同型）
* `u` 更新：scaled dual update（そのまま）

**`D` の扱い**

* `DifferenceMatrix(K)` を別クラスにせず、`D @ beta_j` / `D.T @ v` は「差分」「逆差分」演算として実装すると軽い

  * `Dβ = beta[1:]-beta[:-1]`
  * `D.T r` もO(K)で書ける（行列を作らない）

---

## 7) scikit-learn 風に見せるための最小セット

最小で「それっぽく見える」ために、Estimator 側はこれだけ守れば十分です。

* `__init__` は引数を属性に保存するだけ（副作用なし）
* `fit(X, y)`：

  * `self.n_features_in_`
  * `self.coef_`, `self.gamma_`, `self.z_`, `self.u_`
  * `return self`
* `predict_*` 系は `check_is_fitted` 相当を通す
* `score` は `log \tilde{L}`（もしくは -loss）を返す 

---

## 8) 具体的なクラスツリー（提案）

最小構成で、これ以上割らないのが「シンプル」と「拡張性」の妥協点です。

* `ADMMHazardAFT`（公開 Estimator）

  * uses `FusedLassoADMMSolver`

    * uses `HazardAFTObjective`

      * uses `BaselineHazardModel`（`BSplineBaseline` or `MSpline5Baseline`）
      * uses `TimePartition`
      * uses `QuadratureRule`

---

## 9) 実装上の注意（簡単に詰む箇所だけ）

* **exp(η) が暴れる**：`clip_eta` を必須にする（推定が発散したときの保険）
* **ADMM の stopping**：primal/dual residual を履歴に残す（z,u の収束が見えないとデバッグ不能）

---

## 10) 評価・メトリクス（Metric / Evaluator）と監視

学習アルゴリズム（ADMM）や目的関数の実装とは独立に、**学習結果の定量評価・モデル比較・運用監視**を行うためのクラスを用意する。
Estimator 本体（`ADMMHazardAFT`）は `fit/predict/score` に集中し、評価は別コンポーネントに切り出す。

### `HazardAFTEvaluator`

**責務**

* 学習済みモデル（`ADMMHazardAFT`）に対して、データセット上の評価指標を計算する
* 実験比較（複数モデル/複数ハイパラの横並び評価）に必要な形で結果を集約する
* 運用監視向けに、評価値の時系列（スプリット/期間別）を出力できる形にする

**入力の想定**

* `X`: 特徴量
* `y`: `(time, event)` の2列（Estimator と同じ）
* `times`: 予測を比較する時間点の配列（必要な指標のみ）

**最小インターフェース案**

* `evaluate(model, X, y, times=None) -> dict[str, float]`
  * 代表例：近似対数尤度（`log \\tilde{L}` または `-loss`）、`score` と整合するスカラー
* `compare(models: dict[str, ADMMHazardAFT], X, y, times=None) -> dict[str, dict[str, float]]`
  * モデル名→指標辞書（表として出しやすい）
* `monitor(model, X, y, split_by=None, times=None) -> dict`
  * 期間（例：月次）や属性（例：施設/地域）で分割して指標を返すための器

**指標の方針（実装順）**

* まずは確実に定義がブレない **学習目的と同じ指標**（近似対数尤度）を実装し、再現性のある比較を可能にする
* 追加の予測性能指標（例：ランキング系/確率校正系）は、必要になった時点で「定義（どの時間点・打ち切り扱い）」を明示して追加する

---

## 11) WandB（Weights & Biases）でのロギング

**目的**

* 学習中の収束状況（ADMMの残差・目的関数・ステップ）と、評価指標（上記 Evaluator）を一元的に記録する
* モデル比較・監視のため、ハイパーパラメータと学習後属性（要約）を紐づけて保存する

**設計方針**

* WandB は **任意依存** とし、未インストールでも学習自体は動く（ログだけ無効化）
* ロギング処理は Estimator から分離し、`history_` と Evaluator の結果を「外側」で送る
  * 例：学習ループの各ADMM反復で `history_` を更新し、Logger がそれを引き取って `wandb.log()` する

### `WandBLogger`（任意）

**責務**

* `wandb.init()`（run名、config=ハイパーパラメータ）
* 学習ログ：`history_`（目的関数、primal/dual residual、ρ、Newton step など）を step 付きで記録
* 評価ログ：`HazardAFTEvaluator.evaluate()` の結果を記録（train/valid/test を区別）
* アーティファクト：`coef_`/`gamma_` の要約（例：ノルム、変化点数など）や、必要なら生存曲線の簡易プロット

**最小インターフェース案**

* `start_run(config: dict, name: str | None = None, tags: list[str] | None = None)`
* `log_fit(history_row: dict, step: int)`
* `log_metrics(metrics: dict[str, float], step: int | None = None, prefix: str | None = None)`
* `finish()`

**ログに残す推奨項目（最低限）**

* ハイパーパラ：`lambda_fuse`, `rho`, `time_grid`, `n_baseline_basis`, `quadrature`, `newton_steps_per_admm`, `clip_eta`
* 収束：primal residual, dual residual, 目的関数（または -log\\tilde{L} + penalty）
* 主要評価：`score` と同義の指標（近似対数尤度）

---

## 12) スパコン環境でのパラメータ並列（多数ジョブ）と結果集計の考慮

 **ハイパーパラメータ探索をパラメータ並列**（多数ジョブ）で回し、後から結果をまとめて確認しやすくするための設計上の注意点。
アルゴリズムの高速化や分散学習はまだ実装せず、**実験運用上の再現性・集計容易性**を優先する。

### 12.1 設定（config）の外部化と再現性

* 1ジョブ=1設定（config）を原則とし、config は **JSON/YAML 等でシリアライズ可能**にする
  * `ADMMHazardAFT.__init__` のハイパーパラメータのみを config として保存できる形にする（sklearn clone 互換の思想）
* 乱数の扱い：`random_state` を必須級にし、初期値依存がある箇所は seed を明示して再現できるようにする
* 入力データの同一性を担保するため、データの識別子（ファイルパス、ハッシュ、スプリットseed等）を結果に必ず残す

### 12.2 出力の規約（集計しやすい形式）

* 各ジョブは機械可読な結果を必ず出力する（例：`metrics.json` もしくは 1行1run の `results.jsonl`）
  * 推奨：`metrics`（評価指標）・`config`（ハイパーパラ）・`summary`（学習後属性の要約）・`runtime`（学習時間など）
* ファイル/ディレクトリ命名：run を一意に識別できる `run_id`（UUIDやハッシュ）を付与し、
  `outputs/{experiment_name}/{run_id}/` のように衝突しない構成にする
* `history_` は肥大化しやすいので、
  * （A）最後の値のみを `summary` に入れる（primal/dual residual、目的関数など）
  * （B）全履歴は `history.jsonl` 等で別出力
  のように分離して集計を軽くする

### 12.3 まとめ確認（比較・集計の導線）

* `HazardAFTEvaluator.compare()` が返す「モデル名→指標辞書」は、並列実験の集計でもそのまま使える形（dict of dict）に保つ
* 多数runの集計は「外側のスクリプト」で行う前提とし、最低限次を満たす
  * `results.jsonl` を読み込み、指標でソートして上位を表示できる
  * config と metrics を結合して表にできる（CSV/Parquetに落とすなど）
