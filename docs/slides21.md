---
marp: true
theme: academic
paginate: true
mathjax: {
  output: 'svg',
  loader: {load: ['ui/lazy']}
}
style: |
  text {
    font-family: 'Noto Sans JP';
  }

---

# 進捗報告（2026/06/10）

---
<!-- _header: 本日の内容 -->

- 前回会議の振り返り
- 継続タスクの確認
- 前回会議で追加されたタスクの確認
- 本日追加する内容

---
<!-- _header: 前回会議の振り返り -->

- Framingham / SUPPORT2 の実データ解析結果を確認した
- 提案法と Cox 回帰の $C^{td}$ は大きく変わらず，予測性能の大幅改善は主張しにくい
- 提案法の主張は，予測性能よりも時間変動係数の推定と解釈性に置く方針となった
- SUPPORT2 では fold 間の係数軌跡のばらつきが大きく，初期イベント集中や後半区間のイベント数不足が論点になった

---
<!-- _header: 前回会議での重要な指摘 -->

## $\lambda$ のスケーリング

- 現在の目的関数で，尤度部分と正則化項のサンプルサイズ $N$ に関するオーダーが揃っていない可能性がある
- 通常は

$$
\frac{1}{N}\ell(\theta) + \lambda P(\theta)
$$

または同等に

$$
\ell(\theta) + N\lambda P(\theta)
$$

の形で扱う
- 正則化項に $N$ が掛かっていない場合，実データでは正則化が十分に効いていない可能性がある

---
<!-- _header: 継続タスク -->

## 論文化に向けた整理

- Introduction の整理
  - AFT モデルの位置づけ
  - Pang 法の位置づけ
  - change-point / fused lasso 系の関連研究
  - 本研究の新規性
- Methods の整理
  - 記号の統一
  - 時間分割の定義
  - 目的関数と ADMM の詳細
- 実データ解析の方針整理
  - Framingham / SUPPORT2 の解析結果の位置づけ
  - 時間変動係数の解釈を中心に据える
- Overleaf を川野先生側でプロジェクトを作成し，共同編集する方向

---
<!-- _header: 前回会議で追加されたタスク -->

- 正則化項の実装を確認し，$\lambda$ の $N$ スケーリングを修正する
- 修正後に Framingham / SUPPORT2 の解析を再実行する
- 正則化後に係数が適切に fuse / merge されるか確認する
- AFT モデルを比較対象に追加する
- penalty なしモデルを比較対象に追加する
- 可能であれば Pang 法を比較対象に追加する
- MIMIC-IV 申請

---
<!-- _header: 本日の内容 -->


---
