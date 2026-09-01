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
