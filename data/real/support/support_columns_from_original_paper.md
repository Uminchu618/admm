# SUPPORT2 CSV 列定義メモ

対象ファイル: `support2.csv`  
主な典拠: SUPPORT 原著論文  
**A Controlled Trial to Improve Care for Seriously Ill Hospitalized Patients: The Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments (SUPPORT), JAMA, 1995**

## 前提

この CSV の変数名は、原著論文にそのまま載っているラベルではなく、**SUPPORT 研究で使われた概念・略語を短縮したデータセット変数名**です。  
そのため、以下の説明は

1. 原著論文に明示されている測定対象・アウトカム・患者背景  
2. 論文中の脚注や本文で定義されている指標  
3. SUPPORT で一般的に使われる臨床略語

を突き合わせて整理したものです。  
**論文中で厳密な変数名定義が確認できない列については、その旨を明記**しています。

以下は `support2.csv` の列定義メモです。

---

## 列ごとの意味

### 基本属性・転帰

| 列名 | 意味 | 備考 |
|---|---|---|
| `age` | 年齢 | 連続値。論文 Table 2 に年齢あり。 |
| `death` | 追跡期間中の死亡フラグ | 0/1。`d.time` と対で使う生存解析用の指標とみるのが自然。 |
| `sex` | 性別 | `male` / `female`。 |
| `hospdead` | 入院中死亡フラグ | 0/1。論文では hospital mortality が報告されている。 |
| `slos` | 入院期間（study/hospital length of stay） | 在院日数。論文では length of stay を収集。 |
| `d.time` | 登録から死亡または打ち切りまでの日数 | `death` と組で解釈するのが妥当。6か月死亡やその後の追跡を扱う SUPPORT の生存時間変数。 |

### 疾患分類

| 列名 | 意味 | 備考 |
|---|---|---|
| `dzgroup` | 詳細な疾患群 | 例: `ARF/MOSF w/Sepsis`, `CHF`, `COPD`, `Coma`, `Colon Cancer`, `Lung Cancer`。論文の対象疾患 9 群に対応。 |
| `dzclass` | 大分類の疾患クラス | 例: `ARF/MOSF`, `COPD/CHF/Cirrhosis`, `Coma`, `Cancer`。論文 Table 2 の disease class に対応。 |
| `num.co` | 併存疾患数（number of comorbidities） | 論文では comorbidities を収集。整数値。 |

### 社会経済属性

| 列名 | 意味 | 備考 |
|---|---|---|
| `edu` | 教育年数 | 論文 Table 2 では high school 以上かを報告。CSV では年数で入っている。 |
| `income` | 年収カテゴリ | 例: `under $11k`, `$11-$25k`, `$25-$50k`, `>$50k`。論文 Table 2 の annual income に対応。 |
| `race` | 人種・民族カテゴリ | 例: `white`, `black`, `hispanic`, `asian`, `other`。 |

### 重症度・予後関連列

| 列名 | 意味 | 備考 |
|---|---|---|
| `sps` | SUPPORT physiological score、またはそれに準ずる生理学的重症度スコア | 値域から、複数の生理指標を要約した連続スコアとみるのが自然。**原著本文だけでは略語の厳密展開は確認しきれない。** |
| `aps` | APACHE 系の acute physiology score | SUPPORT 論文でも APACHE III/APS が重症度評価に使われている。整数値。 |
| `surv2m` | 2か月生存確率の予測値 | 0〜1 の連続値。SUPPORT の予後予測モデルに整合的。 |
| `surv6m` | 6か月生存確率の予測値 | 0〜1 の連続値。2か月版と対になる予測値。 |
| `prg2m` | 2か月時点の機能予後・予後良好確率の予測値 | 0〜1 の連続値。`surv2m` とは別に、機能状態や予後区分の推定値を持っている可能性が高い。**厳密定義は補助資料が必要。** |
| `prg6m` | 6か月時点の機能予後・予後良好確率の予測値 | 0〜1 の連続値。`prg2m` の6か月版とみられる。 |

### 既往・臨床経過列

| 列名 | 意味 | 備考 |
|---|---|---|
| `hday` | 入院後何日目の評価/登録かを表す病日 | 値の取り方から hospital day と解釈するのが自然。 |
| `diabetes` | 糖尿病の有無 | 0/1。併存疾患フラグ。 |
| `dementia` | 認知症の有無 | 0/1。併存疾患フラグ。 |
| `ca` | がんの状態 | `no`, `yes`, `metastatic` など。単なる有無ではなく進行度を含むカテゴリ。 |
| `dnr` | DNR（Do Not Resuscitate）指示の状態 | `no dnr`, `dnr before sadm`, `dnr after sadm` などのカテゴリ。`sadm` は study admission とみられる。 |
| `dnrday` | DNR 指示が出た病日 | 数値列。DNR がない場合は `NA` のことがある。 |

### 重症度・生理学的指標

| 列名 | 意味 | 備考 |
|---|---|---|
| `scoma` | coma score / 昏睡関連スコア | 論文は coma を対象疾患に含み、重症度指標を収集。値の取り方から Glasgow coma 系ではなく SUPPORT/APACHE 系の coma 指標の可能性が高い。**原著本文だけでは厳密定義は確認しきれない。** |
| `avtisst` | 平均 TISS スコア（average Therapeutic Intervention Scoring System） | 論文では resource use を「平均 TISS × 在院日数」で定義。 |
| `meanbp` | 平均血圧 | physiological indicators の 1 つ。 |
| `wblc` | 白血球数（white blood cell count） | 生理学的指標。 |
| `hrt` | 心拍数（heart rate） | 生理学的指標。 |
| `resp` | 呼吸数（respiratory rate） | 生理学的指標。 |
| `temp` | 体温 | 生理学的指標。 |
| `pafi` | PaO2 / FiO2 比 | 呼吸不全評価でよく使う酸素化指標。 |
| `alb` | アルブミン | 血液生化学。 |
| `bili` | 総ビリルビン | 血液生化学。 |
| `crea` | クレアチニン | 腎機能指標。 |
| `sod` | 血清ナトリウム | 電解質。 |
| `ph` | 血液 pH | 酸塩基平衡。 |
| `glucose` | 血糖 | 生化学検査。 |
| `bun` | BUN（blood urea nitrogen） | 腎機能・代謝指標。 |
| `urine` | 尿量 | 生理学的/臨床管理指標。 |

### 費用・資源消費

| 列名 | 意味 | 備考 |
|---|---|---|
| `charges` | 病院請求額 | 論文 Table 2 に median hospital charges がある。 |
| `totcst` | 総コスト（total cost） | 請求額ではなくコスト推定値とみられる。 |
| `totmcst` | 総 Medicare コスト、またはコストの別集計 | 変数名からはその解釈が自然。**原著本文だけでは厳密定義は確認不可。** |

### ADL・機能状態

| 列名 | 意味 | 備考 |
|---|---|---|
| `adlp` | 患者報告の ADL 障害スコア | `p` は patient と解釈するのが自然。0-7 の整数。 |
| `adls` | 代諾者/家族等（surrogate）報告の ADL 障害スコア | `s` は surrogate と解釈するのが自然。0-7 の整数。 |
| `adlsc` | 統合 ADL スコア / 補完後 ADL スコア | 患者または surrogate 情報を統合した連続値とみられる。論文では「患者面接不能なら surrogate 回答で代用」とある。 |
| `sfdm2` | 2か月時点機能状態に関する分類または欠測理由コード | 値に `SIP>=30`, `Coma or Intub`, `<2 mo. follow-up` などがある。**2か月後機能予後に関連する派生変数**とみるのが妥当。原著本文では phase II 介入で「2か月時点の機能障害予測」を扱っているが、この列の厳密なコード表は本文だけでは出ていない。 |

---

## 原著論文と対応づけて読める主要列

### 1. 患者背景
- `age`
- `sex`
- `race`
- `income`
- `edu`
- `dzgroup`
- `dzclass`
- `num.co`
- `diabetes`
- `dementia`
- `ca`

### 2. 重症度・生理学
- `scoma`
- `sps`
- `aps`
- `meanbp`
- `wblc`
- `hrt`
- `resp`
- `temp`
- `pafi`
- `alb`
- `bili`
- `crea`
- `sod`
- `ph`
- `glucose`
- `bun`
- `urine`

### 3. 資源利用
- `slos`
- `hday`
- `avtisst`
- `charges`
- `totcst`
- `totmcst`

### 4. 生存・死亡・予後
- `death`
- `hospdead`
- `d.time`
- `surv2m`
- `surv6m`
- `prg2m`
- `prg6m`
- `dnr`
- `dnrday`

### 5. 機能状態
- `adlp`
- `adls`
- `adlsc`
- `sfdm2`

---

## この CSV を使うときの注意

1. `death` と `hospdead` は別物  
   - `hospdead` は**その入院中に死亡したか**
   - `death` は**追跡期間中に死亡したか**  
   生存解析では `death` と `d.time` を組で使うのが基本です。

2. `surv2m` / `surv6m` と `prg2m` / `prg6m` は別系列の予測値  
   前者は生存確率、後者は機能予後や広義の prognosis を表す派生予測値である可能性が高く、同一視しない方が安全です。

3. `charges` と `totcst` は同じではない  
   論文でも **charges（請求額）** と **resource use / cost** は区別されています。

4. `adlp`, `adls`, `adlsc` はそのまま混ぜない  
   患者自己申告、代理回答、補完後スコアが混在している可能性があります。

5. `sfdm2`, `totmcst`, `scoma`, `sps`, `prg2m`, `prg6m` は原著論文だけでは厳密定義が落ちない  
   ここは SUPPORT 関連の補助論文やデータ辞書がないと完全確定できません。

6. `dnr` と `dnrday` は重要だが、時点の解釈に注意  
   `dnr before sadm` / `dnr after sadm` のような値があり、単純な 0/1 ではなく「いつ DNR が成立したか」を含むカテゴリ列です。

---

## 実務上の最小データ辞書

```text
age      : 年齢
death    : 追跡期間中死亡フラグ
sex      : 性別
hospdead : 入院中死亡フラグ
slos     : 在院日数
d.time   : 死亡/打ち切りまでの日数
dzgroup  : 詳細疾患群
dzclass  : 疾患大分類
num.co   : 併存疾患数
edu      : 教育年数
income   : 年収カテゴリ
scoma    : 昏睡/意識レベル関連スコア
sps      : 生理学的重症度要約スコア
aps      : APACHE系 acute physiology score
surv2m   : 2か月生存確率予測値
surv6m   : 6か月生存確率予測値
hday     : 病日 / hospital day
diabetes : 糖尿病フラグ
dementia : 認知症フラグ
ca       : がん状態カテゴリ
prg2m    : 2か月予後予測値
prg6m    : 6か月予後予測値
dnr      : DNR状態
dnrday   : DNR決定病日
charges  : 病院請求額
totcst   : 総コスト
totmcst  : コスト別集計（厳密定義は要補足資料）
avtisst  : 平均TISSスコア
race     : 人種・民族
meanbp   : 平均血圧
wblc     : 白血球数
hrt      : 心拍数
resp     : 呼吸数
temp     : 体温
pafi     : PaO2/FiO2比
alb      : アルブミン
bili     : 総ビリルビン
crea     : クレアチニン
sod      : 血清ナトリウム
ph       : 血液pH
glucose  : 血糖
bun      : BUN
urine    : 尿量
adlp     : 患者報告ADL障害スコア
adls     : surrogate報告ADL障害スコア
sfdm2    : 2か月機能状態関連コード
adlsc    : 統合/補完後ADLスコア
```

---

## 典拠メモ

- SUPPORT 原著論文では、対象患者、疾患群、死亡、DNR、ICU日数、人工呼吸、痛み、TISS を用いた resource use、APACHE III/APS、ADL、2か月後機能障害予測などが説明されている。
- ただし **CSV の短縮変数名そのものの完全なコードブックは原著論文本文には載っていない**。  
  したがって、この Markdown は**原著ベースで確実に言える部分**と、**変数名から合理的に復元した部分**を分けて記述した。
