## 概要

本リポジトリは、ADMM（Alternating Direction Method of Multipliers）を用いて
fused lasso（時間方向の差分に対する L1 正則化）付きの Hazard-AFT モデルを推定するための
Python 実装です。

現状は「Estimator（sklearn 風 API）と内部コンポーネントの分離」「設定ファイルからの初期化」など、
実装の骨格と責務分割を優先して整備しており、尤度・勾配・ヘッセ・ADMM 反復の本体は未実装です。

## 目的

- `fit/predict/score` を提供する推定器 `ADMMHazardAFT` を中心に、
  目的関数（近似対数尤度）、時間分割、求積、ベースライン基底、ADMM ソルバを疎結合に保つ
- 将来的な差し替え（例: B-spline → M/I-spline、求積ルール変更）を容易にする
- 実験運用を想定し、設定（ハイパーパラメータ）を TOML/JSON で外部化する

## 使い方（現状）

推定器の初期化は、設定ファイルを通して確認できます（学習処理自体は未実装）。

1) 設定ファイルを用意（例: `config.toml`）

2) CLI を実行

```bash
uv run main.py --config config.toml
```

想定される例外:
- 設定ファイルが存在しない: `FileNotFoundError`
- TOML を Python < 3.11 で読み込もうとした: `RuntimeError`
- JSON/TOML の構文が不正: パーサ由来の例外

## Module Layout（構成）

```
admm/
  __init__.py        公開 API のエクスポート
  model.py           ADMMHazardAFT 推定器 + fit フロー
  solver.py          ADMM ソルバ（fused lasso）
  objective.py       近似対数尤度 + 勾配 + ヘッセ
  baseline.py        ベースラインハザード基底（B-spline 等）
  time_partition.py  time_grid と区間ユーティリティ
  quadrature.py      数値積分（求積）ルール
  config.py          設定ローダ（TOML/JSON）
  types.py           共有の型エイリアス
main.py              CLI エントリポイント（設定読み込み）
config.toml          設定例
```
