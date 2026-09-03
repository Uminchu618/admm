# シミュレーションの5-fold CVによるlambda選択

主解析の `lambda_fuse` は、各シミュレーション学習データ内の5-fold CVで選択する。
独立評価データはlambda選択には使用せず、選択後に全学習データで再推定したモデルの
最終評価にだけ使用する。

## 選択規則

各候補lambdaについて、被験者ID単位・event層化の5-fold CVを行う。
5個の検証 `c_td` の単純平均が最大のlambdaを選ぶ。

候補lambdaは次の条件をすべて満たす必要がある。

- fold 0から4までが重複なく存在する
- 5 foldすべてが正式収束している
- 5 foldすべての検証 `c_td` が有限値である

平均値が `1e-12` 以内で同点なら、より強く融合する大きいlambdaを選ぶ。
BICはソルバ診断および感度分析用には残すが、主解析のlambda選択には使わない。

## パイロット実験

学習データと独立評価データを生成する。

```bash
./scripts/pilot/generate_data.sh
```

CVを投入する。100データセット、9 lambda、5 foldの場合は4,500 taskになる。

```bash
./scripts/pilot/submit.sh
```

全task完了後、データセットごとのlambdaを選択する。

```bash
./scripts/pilot/aggregate.sh
```

各データセットのディレクトリに次を保存する。

- `fold_results.csv`
- `summary_by_lambda.csv`
- `selected_lambda.json`

全選択結果は `cv_selections.csv` にまとめる。

次に、選択lambdaだけで全学習データを再学習し、独立評価データ上の `c_td` を計算する。

```bash
./scripts/pilot/submit_refit.sh
./scripts/pilot/aggregate_refit.sh
```

結果を可視化する。

```bash
uv run scripts/pilot/visualize_cv_results.py \
  --cv-selections outputs/pilot_cv/<run>/cv_selections.csv \
  --refit-summary outputs/pilot_cv_refit/<run>/refit_summary.csv \
  --output-dir outputs/pilot_cv_refit/<run>/visualizations
```

係数関数と真値を比較する場合は次を実行する。

```bash
uv run scripts/pilot/plot_cv_selected_beta.py \
  --cv-selections outputs/pilot_cv/<run>/cv_selections.csv \
  --refit-summary outputs/pilot_cv_refit/<run>/refit_summary.csv \
  --output-dir outputs/pilot_cv_refit/<run>/beta_comparison
```

## 小規模確認

`PILOT_N_FOLDS`、学習データディレクトリ、lambda gridを差し替えることで、
少数データ・少数lambdaのスモーク実験を先に実行できる。
既存の `outputs/pilot/` は上書きせず、CV結果は `outputs/pilot_cv/`、
再学習結果は `outputs/pilot_cv_refit/` に分離する。

## 粗いCV選択値周辺の局所fine grid

既存9点CVの選択値を中心に、上下の粗い候補との間をそれぞれ10分割し、
データセットごとに21点の局所gridを作る。正のlambdaは対数間隔、
lambda 0を含む区間は線形間隔とする。粗いCVで上限0.25を選んだ場合は
探索上限を0.75まで広げる。

既存結果は `outputs/pilot_cv/` から再利用し、未計算、非収束、または検証
`c_td` が非有限の組だけを `outputs/pilot_cv_refined_additions/` へ投入する。

### スモーク実験

各シナリオ2 seedに限定してmanifestを作り、追加taskを投入する。

```bash
PILOT_REFINED_DATA_NAMES="oracle_seed_42,oracle_seed_43,fine_grid_seed_42,fine_grid_seed_43,off_grid_seed_42,off_grid_seed_43,small_seed_42,small_seed_43,no_change_seed_42,no_change_seed_43" \
  ./scripts/pilot/submit_refined_cv.sh
```

### 全データセット

```bash
./scripts/pilot/submit_refined_cv.sh
```

同じコマンドを再実行すると、収束判定を満たし有限な検証 `c_td` を持つ
結果は再利用され、不足taskだけが新しいmanifestに入る。追加出力に使用できない
`result.json` がある場合は、削除せず `result.before_retry_*.json` として保存してから
再実行する。

全task完了後、既存結果と追加結果を統合する。

```bash
./scripts/pilot/aggregate_refined_cv.sh
```

5 foldが揃い、収束判定を満たし、検証 `c_td` が有限な候補だけからlambdaを選ぶ。
利用できない候補は選択対象から外し、件数を `refined_cv_audit.csv` に記録する。
選択lambdaの隣接候補を利用できない場合は `selection_neighbor_ineligible=true` として
感度確認の対象にする。利用可能な候補が1つもないデータセットがある場合だけ集計を
停止する。

選択後の全学習データ再学習と集計は次で行う。

```bash
./scripts/pilot/submit_refined_refit.sh
./scripts/pilot/aggregate_refined_refit.sh
```

粗いCVと局所fine-grid CVを同一seedで比較する。

```bash
uv run scripts/pilot/compare_coarse_refined_cv.py \
  --coarse-selections outputs/pilot_cv/<run>/cv_selections.csv \
  --coarse-refit-summary outputs/pilot_cv_refit/<run>/refit_summary.csv \
  --refined-selections outputs/pilot_cv_refined/<run>/cv_selections.csv \
  --refined-refit-summary outputs/pilot_cv_refined_refit/<run>/refit_summary.csv \
  --output-dir outputs/pilot_cv_refined_refit/<run>/analysis
```

比較対象は選択lambda、独立評価 `c_td`、係数RMISE、変化点数、
precision、recall、F1である。局所grid端が選ばれたデータセットは
`refined_cv_audit.csv` の `selected_at_local_boundary` で判定し、
該当データセットだけ探索範囲を追加拡張する。
