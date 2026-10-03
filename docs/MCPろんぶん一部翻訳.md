# 枝分かれ · 枝分かれ · Introduction日本語訳と論文概要

**User:** Anonymous  
**Created:** 9/6/2026 23:00:20  
**Updated:** 9/16/2026 17:20:31  
**Exported:** 9/16/2026 17:21:06  
**Link:** [https://chatgpt.com/g/g-p-6a9c166062a48191830d34e54a0fcca1/c/6a9d71e2-bd44-83ee-9327-fdd206b43283?no_universal_links=1](https://chatgpt.com/g/g-p-6a9c166062a48191830d34e54a0fcca1/c/6a9d71e2-bd44-83ee-9327-fdd206b43283?no_universal_links=1)  

## Prompt:
2026/9/5 23:29:12

1. Introductionに関して日本語に翻訳してください。
出てくる引用論文に関しては、その概要を示してください

## Response:
2026/9/5 23:29:14

> **翻訳方針を整理中**
>
> 論文の導入部を自然な日本語へ訳し、そこに挙げられた主要研究を各1〜2文で要約します。引用情報が不足する研究は、タイトルと一般的な研究内容に基づく簡潔な説明にとどめます。
>
> **整理した翻訳方針**
>
> 
>
> **第1章序論と引用文献の概要を翻訳・整理**
>
> 
>
> **要約方針を確定中**
>
> 導入部を自然な日本語に訳し、本文で引用される研究を個別に簡潔に整理します。原文にない詳細は補わず、各研究の目的、提案手法、位置づけが分かる範囲でまとめます。
>
> **要約方針を確定した**
>
> 
>
> **引用研究を整理中**
>
> 導入部の翻訳に加え、引用文献ごとに目的・手法・位置づけを簡潔に示します。引用一覧だけでは詳細不足のため、外部の書誌情報や要旨も照合し、原文の主張と補足説明を区別します。
>
> **引用研究を整理した**
>
> 
>
> **12件のウェブサイトを検索しました**
>
> 
>
> 45s考えました

以下は、Breheny & Huang (2011) の **1. Introduction** の日本語訳です。原文の論理構成と統計学上の用語をできるだけ維持しています。Introduction はPDFの1-3ページにまたがっています。`nihms332857.pdf`

## 1. Introduction 日本語訳

回帰分析において、**変数選択（variable selection）**は重要な問題である。一般に、予測因子とアウトカムとの間に存在する重要な関係を見落とさないよう、多数の候補予測変数について測定が行われる。近年では、自動化されたデータ収集・保存手法が低コストで容易に利用できるようになったため、この傾向はさらに強まっており、**標本サイズに比べて変数の数が非常に多い問題**が数多く生じている。

推定のばらつきを減らし、より解釈しやすいモデルを得るためには、通常、重要な変数だけからなるより小さな部分集合を求める。しかし、候補となる予測変数の部分集合を探索して適切な小規模モデルを見つける方法は、**不安定になり得る**［Breiman (1996)］うえ、変数の次元がそれほど大きくなくても、計算量の観点から実行不可能となり得る。

このような欠点を避けるため、近年では、部分集合選択を離散的な組合せ探索ではなく**連続的な形で行う罰則付き回帰（penalized regression）**が数多く提案されている。罰則付き回帰では、一部の回帰係数を0方向へ縮小すると同時に、一部の係数を**正確に0**とすることで変数選択を行う。したがって、0にならずに残った係数に対応する変数が選択されることになる。

最も広く用いられている罰則付き回帰法は **lasso** ［Tibshirani (1996)］である。Lassoには多くの望ましい性質がある一方で、lassoによって導入される縮小のため、**大きな回帰係数についても0方向への無視できないバイアスが生じる**。

このバイアスを軽減することを目的として、別の罰則関数も提案されている。その代表例が、

- **smoothly clipped absolute deviation（SCAD）罰則**［Fan and Li (2001)］
- **minimax concave penalty（MCP）**［Zhang (2010)］

である。

SCADおよびMCPを提案した研究では、これらを用いた回帰モデルが、いわゆる**オラクル性（oracle property）**を有することが示されている。これは漸近的には、「どの係数が0であり、どの係数が0でないかを解析者があらかじめ知っていた場合」と同等の性能を達成できる、という性質である。

ただし、SCADとMCPの罰則関数は**非凸（nonconvex）**であり、このことがモデルの推定に数値計算上の困難をもたらす。一方、lassoの罰則は凸である。Lassoについては、**least angle regression（LARS）**［Efron et al. (2004)］を用いることにより、一連のlasso解、すなわち正則化パラメータに沿った解パス全体を、通常の最小二乗法と同程度の計算時間で非常に効率よく求めることができる。

非凸罰則については、Zou and Li (2008) が罰則関数を**局所線形近似（local linear approximation; LLA）**する方法を提案している。この近似によって得られた目的関数は、LARSアルゴリズムを用いて最適化することができる。

さらに近年、lasso罰則付きモデルを推定するための**座標降下法（coordinate descent）**が、特に高次元問題においてLARSに匹敵する性能を持つことが示されている［Friedman et al. (2007); Wu and Lange (2008); Friedman, Hastie and Tibshirani (2010)］。

本論文では、罰則関数が非凸である **SCAD回帰およびMCP回帰に対して座標降下法を適用すること**を検討する。また、これらのアルゴリズムは、公開Rパッケージ **`ncvreg`** として実装している。

高次元回帰および変数選択の手法は、多くの科学分野、とりわけ**ハイスループットな生物医学研究**に応用できる。本論文では、そのような研究のうち、

**遺伝的関連解析（genetic association study）**と  
**遺伝子発現解析（gene expression study）**

という2種類の研究にこれらの手法を適用する。両者では、スパース回帰モデルを使用する動機が異なっている。

遺伝的関連解析では、対象となる表現型と関連する遺伝マーカーはごく少数であると考えられている。したがって、基礎となるデータ生成過程は**非常にスパース**である可能性が高く、スパース回帰法が良好に機能すると考えられる。

遺伝子発現研究についても、基礎となる表現がスパースである場合がある。しかし、仮に比較的「密（dense）」な構造であったとしても、別の理由からスパース回帰モデルが望まれることがある。

たとえば、マイクロアレイ実験のデータは、バイオマーカーの発見や診断アッセイの設計に利用できる。こうしたアッセイを実際の臨床現場で使用可能なものとするためには、**使用するプローブの数を少数に抑える必要がある**［Yu et al. (2007)］。

本論文の構成は以下の通りである。第2節では、MCPおよびSCADによって罰則化された**線形回帰モデル**を推定するアルゴリズムを説明し、その収束性について議論する。第3節では、これらのアルゴリズムを**ロジスティック回帰**に適用するための修正について述べる。

安定性、局所凸性、およびこれらを調べるための診断指標、さらにチューニングパラメータの選択については第4節で議論する。提案アルゴリズムの数値計算上の効率性については第5節で検討し、LLAアルゴリズムと比較する。また、lasso、SCAD、MCPの統計的性質についてもシミュレーションデータを用いて第5節で比較し、生物医学データへの適用を第6節で行う。

※ 原文には “**mimimax concave penalty**” と記載されている箇所がありますが、正式名称は **minimax concave penalty (MCP)** です。

---

# Introductionで引用されている論文の概要

以下では単なる書誌情報ではなく、**各論文が何を提案・主張したものなのか**と、**Breheny & Huang (2011) がIntroductionでなぜ引用しているのか**を対応させます。引用論文の内容については、各原著論文の要旨等も参照して補足しています。

| 引用論文 | 主な内容 | このIntroductionでの役割 |
|---|---|---|
| **Breiman (1996), “Heuristics of instability and stabilization in model selection”** | モデル選択法の**不安定性**を体系的に議論した論文。データをわずかに変化させただけで選択されるモデルや予測器が大きく変化する場合、その手法をunstableと定義する。不安定な選択法では予測性能にも損失が生じ得ることを示し、データを摂動して得た複数の予測器を平均化する「安定化」の考え方も論じる。([SciSpace](https://scispace.com/papers/heuristics-of-instability-and-stabilization-in-model-2ji36pw9zj?utm_source=chatgpt.com)) | best subset selectionのような**離散的な変数選択が不安定になり得る**という主張の根拠。 |
| **Tibshirani (1996), “Regression Shrinkage and Selection Via the Lasso”** | **lassoを提案した原著論文**。残差平方和を、係数の絶対値和（$L_1$ノルム）に制約を課して最小化する。$L_1$制約によって係数の一部が厳密に0となるため、**推定と変数選択を同時に行える**。subset selectionの解釈性とridge regressionの安定性の双方を持つことを示した。([Royal Statistical Society](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1996.tb02080.x?utm_source=chatgpt.com)) | 罰則付き変数選択の代表的方法としてlassoを導入。ただし後続の議論では「大きな係数まで縮小するためバイアスが大きい」という弱点が問題になる。 |
| **Fan & Li (2001), “Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties”** | **SCAD罰則の代表的原著論文**。良い罰則関数に必要な性質として、sparsity、unbiasedness、continuityなどを議論し、非凹罰則による変数選択法を構築した。適切な正則化パラメータのもとで、正しいモデルを既知とした場合と漸近的に同等になる**oracle property**を示した。([Penn State](https://pure.psu.edu/en/publications/variable-selection-via-nonconcave-penalized-likelihood-and-its-or/?utm_source=chatgpt.com)) | 「lassoの大きな係数に対する縮小バイアスを減らしつつ、スパース性を維持したい」というSCAD導入の理論的背景。 |
| **Zhang (2010), “Nearly Unbiased Variable Selection under Minimax Concave Penalty”** | **MCP（minimax concave penalty）**と、それを用いるMC+法を提案。Lassoのバイアスを軽減しつつ、高次元 $p\gg n$ における変数選択の整合性などを理論的に研究している。MCPは、一定以上の大きさを持つ係数について罰則の微分を0にすることで、大きな真の係数をほぼ縮小しないよう設計されている。([DOI](https://doi.org/10.1214%2F09-aos729?utm_source=chatgpt.com)) | SCADと並ぶ主要な**非凸罰則**としてMCPを導入し、そのoracle的・低バイアスな性質の根拠としている。 |
| **Efron et al. (2004), “Least Angle Regression”** | **LARS（Least Angle Regression）**を提案。forward selectionをより連続的・効率的にしたアルゴリズムで、簡単な修正によって**lassoの正則化パス全体を非常に効率よく計算できる**ことを示した。([arXiv](https://arxiv.org/abs/math/0406456?utm_source=chatgpt.com)) | 「凸問題であるlassoには極めて効率的な解法が存在する」という比較対象。SCAD/MCPでは目的関数が非凸なので、そのままLARSを利用できない。 |
| **Zou & Li (2008), “One-step Sparse Estimates in Nonconcave Penalized Likelihood Models”** | SCADなどの非凹罰則付き尤度の計算を容易にするため、罰則関数を各反復で線形化する**LLA（Local Linear Approximation）アルゴリズム**を提案。良い初期値があれば**one-step LLA**にもoracle propertyがあることを示した。([PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2759727/?utm_source=chatgpt.com)) | Breheny & Huangが提案するcoordinate descentの主要な**比較対象となる非凸最適化法**。この論文では後に「LLAはMCP/SCADの解パス計算では非効率になり得る」と比較される。 |
| **Friedman et al. (2007), “Pathwise Coordinate Optimization”** | 1変数ずつ更新する**coordinate-wise descent**をlasso等の凸最適化問題に適用し、大規模lassoではLARSと非常に競争力があることを示した。elastic netなどにも適用可能であることを論じた。([Stanford University](https://web.stanford.edu/~hastie/Papers/pathwise.pdf?utm_source=chatgpt.com)) | 「座標降下法は単純だが、高次元lassoでは非常に高速」という、本論文の方法論を動機づける重要な先行研究。 |
| **Wu & Lange (2008), “Coordinate Descent Algorithms for Lasso Penalized Regression”** | Lasso回帰に対する高速なcoordinate descent法を検討。cyclic coordinate descentやgreedy coordinate descentを扱い、$p$ が $n$ を大きく上回る問題にも適用できることを示す。収束性についても理論的に検討している。([arXiv](https://arxiv.org/abs/0803.3876?utm_source=chatgpt.com)) | coordinate descentがlassoに対して実用的・高速であることを裏付けるもう一つの先行研究。 |
| **Friedman, Hastie & Tibshirani (2010), “Regularization Paths for Generalized Linear Models via Coordinate Descent”** | 現在の **`glmnet`** の基礎となる研究。線形回帰、二値ロジスティック回帰、多項ロジスティック回帰などについて、lasso、ridge、elastic netの正則化パスを**cyclic coordinate descent**で高速に計算する方法を提示。大規模・スパースデータにも適する。([PubMed](https://pubmed.ncbi.nlm.nih.gov/20808728/?utm_source=chatgpt.com)) | 「coordinate descentは特に高次元でLARSに匹敵、あるいはそれ以上に実用的」という背景。本論文はこの発想を**非凸なSCAD/MCPへ拡張**する。 |
| **Yu et al. (2007), “Feature Selection and Molecular Classification of Cancer Using Genetic Programming”** | マイクロアレイ遺伝子発現データから少数の特徴遺伝子を選択し、genetic programmingを用いてがんを分類する研究。しばしば**5遺伝子以下**の比較的小さな分類器でも高い分類性能を達成し、独立研究・異なるプラットフォームへの一般化も検討した。([PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1854845/?utm_source=chatgpt.com)) | 臨床診断アッセイでは、「予測精度だけでなく**少数の遺伝子・プローブで実現できること自体が重要**」というスパースモデルの実用的動機を示している。 |

## Introduction全体の論理

このIntroductionが組み立てている問題設定を数式的・統計的に圧縮すると、

**best subset selection**
→ スパースだが、組合せ最適化で計算困難・不安定

**lasso**
→ 凸最適化なので高速かつ安定、変数選択も可能  
→ しかし $L_1$ 罰則のため**大きな真の係数まで縮小され、バイアスが残る**

**SCAD / MCP**
→ 大きな係数への罰則を弱めることでlassoのバイアスを軽減し、oracle propertyも持つ  
→ しかし**非凸最適化問題**になる

というトレードオフです。

そこでBreheny & Huangの中心的な問いは、

> **「lassoで成功しているcoordinate descentを、非凸なSCAD/MCPにも適用して、高速かつ安定に計算できないか？」**

となっています。

この問いが、その後の第2節以降の **MCP/SCADに対するcoordinate descent → 収束性 → 局所凸性診断 → logistic regressionへの拡張** という流れにつながっています。`nihms332857.pdf`

---

**Sources:**

- [(PDF) Heuristics of instability and stabilization in model selection (1996) | Leo Breiman | 1313 Citations](https://scispace.com/papers/heuristics-of-instability-and-stabilization-in-model-2ji36pw9zj?utm_source=chatgpt.com)
- [Regression Shrinkage and Selection Via the Lasso - Tibshirani - 1996 - Journal of the Royal Statistical Society: Series B (Methodological) - Wiley Online Library](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1996.tb02080.x?utm_source=chatgpt.com)
- [Variable selection via nonconcave penalized likelihood and its oracle properties - Penn State](https://pure.psu.edu/en/publications/variable-selection-via-nonconcave-penalized-likelihood-and-its-or/?utm_source=chatgpt.com)
- [Nearly unbiased variable selection under minimax concave penalty](https://doi.org/10.1214%2F09-aos729?utm_source=chatgpt.com)
- [Least Angle Regression](https://arxiv.org/abs/math/0406456?utm_source=chatgpt.com)
- [One-step Sparse Estimates in Nonconcave Penalized Likelihood Models - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2759727/?utm_source=chatgpt.com)
- [The Annals of Applied Statistics](https://web.stanford.edu/~hastie/Papers/pathwise.pdf?utm_source=chatgpt.com)
- [Regularization Paths for Generalized Linear Models via Coordinate Descent - PubMed](https://pubmed.ncbi.nlm.nih.gov/20808728/?utm_source=chatgpt.com)

## Prompt:
2026/9/6 23:00:17

2. Linear regression with nonconvex penalties 
省略なく　日本語に翻訳してください。

## Response:
2026/9/6 23:00:20

> **PDF第2節の原文抽出**
>
>
> **数式を確認する**
>
> 式中の \(\lambda_{\max}\) が絶対値なしで定義されている点は、ソフトしきい値処理の通常の形と合わないように見えます。表示上の誤植やOCRの問題も考えられますが、原文を補わず、そのまま注記する方針です。
>
> **数式を確認した**
>
> 
>
> **PDF第2節を抽出し画像解析と正則化パラメータ式を確認**
>
> 
>
> 2m 10s考えました

以下、論文 **Breheny & Huang (2011), “Coordinate Descent Algorithms for Nonconvex Penalized Regression, with Applications to Biological Feature Selection”** の **Section 2. Linear regression with nonconvex penalties** を、**2.1-2.4、数式、脚注、Lemma、Propositionを含めて省略せず**日本語に翻訳します。数式は原文の形を保持します。`nihms332857.pdf`

なお、2.4 の $\lambda_{\max}$ に関する記述には、数式・文章上やや不自然な箇所があります。ここでは勝手に修正せず原文どおり訳し、最後に注記します。

---

# 2. 非凸罰則を用いた線形回帰
## 2. Linear regression with nonconvex penalties

$n$ 個の観測があるとする。それぞれの観測には、応答変数

$$
y_i
$$

と、$p$ 個の特徴量

$$
\{x_{i1},\ldots,x_{ip}\}
$$

の測定値が含まれている。

一般性を失うことなく、特徴量は次を満たすように標準化されているものと仮定する。

$$
\sum_{i=1}^{n}x_{ij}=0,
$$

$$
n^{-1}\sum_{i=1}^{n}x_{ij}^{2}=1,
$$

さらに、

$$
\sum_{i=1}^{n}y_i=0.
$$

この標準化により、罰則がすべての共変量に対して**等変的（equivariant）な形で等しく適用される**ことが保証され、また切片を導入する必要がなくなる。

これは正則化推定における標準的な方法である。罰則付きモデルを当てはめた後、推定値は元の尺度に変換され、その段階で切片が導入される。

---

応答変数の期待値が、共変量に対して次の線形関数を通じて依存するモデルを考える。

$$
E(\mathbf y)=\boldsymbol{\eta}=X\boldsymbol{\beta}.
$$

ここで問題となるのは、回帰係数ベクトル

$$
\boldsymbol{\beta}
$$

を推定することである。

罰則付き回帰法では、**損失関数と罰則関数の和から構成される目的関数 $Q$** を最小化することによって、この推定を行う。

この節では、損失関数として**二乗誤差損失（squared error loss）**を用いる。

$$
\boxed{
Q_{\lambda,\gamma}(\boldsymbol{\beta})
=
\frac{1}{2n}
\sum_{i=1}^{n}
(y_i-\eta_i)^2
+
\sum_{j=1}^{p}
p_{\lambda,\gamma}(|\beta_j|)
}
\tag{2.1}
$$

ここで

$$
p_{\lambda,\gamma}(\cdot)
$$

は回帰係数に対する関数であり、パラメータ $\lambda$ によって添字付けされている。

$\lambda$ は、損失関数と罰則関数との間のトレードオフを制御する。また、この罰則関数の形状は、1つまたは複数のチューニングパラメータ $\gamma$ によって決定される場合がある。

この方法では、$\lambda$ の値に応じて一連の解が得られる。この種の方法はしばしば**正則化法（regularization methods）**と呼ばれ、$\lambda$ は**正則化パラメータ（regularization parameter）**と呼ばれる。

---

式 (2.1) を最適化する $\boldsymbol{\beta}$ を求めるために、LLAアルゴリズムでは、まず罰則関数を線形近似し、その後LARSアルゴリズムを用いて解を計算する。

この過程を、$\lambda$ のグリッド上の各値について、収束するまで反復する。¹

アルゴリズムおよびその実装の詳細については Zou and Li (2008) を参照されたい。

LLAアルゴリズムは、回帰係数の更新を得るために**解パスを追跡するLARSアルゴリズム**を用いるという点で、ある程度本質的に非効率である。

例えば、$\lambda$ の値を100個取ったグリッドを考え、各点で収束まで平均10回の反復を必要とするとする。

このときLLAアルゴリズムは、MCPまたはSCADの解パスを1本近似するために、

$$
100\times10=1000
$$

本のlasso解パスを計算しなければならない。

---

LLAの代替手段として、**座標降下法（coordinate descent）**を用いることができる。

座標降下アルゴリズムは、一度に1個のパラメータについて目的関数を最適化し、すべてのパラメータについて順番に反復していき、収束するまでこれを繰り返す。

考え方は単純であるが効率的であり、全パラメータを1回走査するために必要な演算量はわずか

$$
O(np)
$$

である。

反復回数が $p$ より小さければ、QR分解によって線形回帰問題を解くために必要な

$$
np^2
$$

回の演算よりも少ない計算負荷で解に到達できる。

さらに、計算負荷は $p$ に対して線形にしか増加しないため、座標降下アルゴリズムは非常に高次元の問題にも適用することができる。

---

### 脚注1

Zou and Li (2008) では、ある条件を満たす初期推定値から開始する**one-step版のLLA**についても議論している。

これは興味深い話題ではあるが、本論文では、指定されたMCP/SCAD目的関数そのものを最小化するアルゴリズムに焦点を当てる。

これによって、別途初期推定量を必要とすることなく、Fan and Li (2001) および Zhang (2010) が示した**oracle property**を有することになる。

---

座標降下アルゴリズムは、**1次元では単純な閉形式解を持つが、高次元では閉形式解を持たない問題**に特に適している。

座標降下アルゴリズムの基本構造は単純である。

各

$$
j\in\{1,\ldots,p\}
$$

について、$\boldsymbol{\beta}$ の残りの要素を直近に更新された値に固定したまま、$\beta_j$ に関して $Q$ を部分的に最適化する。

LLAアルゴリズムと同様に、座標降下アルゴリズムも収束するまで反復する。

そしてこの過程を、$\lambda$ のグリッド上の各値について繰り返すことで、**解パス（path of solutions）**を得る。

---

座標降下アルゴリズムの効率性には2つの理由がある。

1. 各更新を非常に高速に計算できる。
2. 連続的な解パスを計算している場合（Section 2.4参照）、初期値が解から大きく離れることがなく、少数回の反復だけで済む。

高速な更新が可能である理由は、$\beta_j$ に関する $Q$ の最小化が、現在の残差

$$
\mathbf r
=
\mathbf y-X\boldsymbol{\beta}
$$

を $x_j$ に対して単変量回帰することで得られるからである。

その計算コストは

$$
O(n)
$$

である。

これらの更新式の具体的な形は、用いる罰則関数、および線形回帰を行うかロジスティック回帰を行うかに依存する。それぞれについて、対応する節でさらに説明する。

本節では、SCADおよびMCPによって罰則化された最小二乗回帰に対する座標降下アルゴリズムを説明するとともに、これらのアルゴリズムの収束性について検討する。

---

# 2.1 MCP

Zhang (2010) は、区間

$$
[0,\infty)
$$

上で次のように定義されるMCPを提案した。

$$
\boxed{
p_{\lambda,\gamma}(\theta)
=
\begin{cases}
\lambda\theta-\dfrac{\theta^2}{2\gamma},
& \theta\leq\gamma\lambda,\\[6pt]
\dfrac{1}{2}\gamma\lambda^2,
& \theta>\gamma\lambda,
\end{cases}
}
$$

およびその微分は

$$
\boxed{
p'_{\lambda,\gamma}(\theta)
=
\begin{cases}
\lambda-\dfrac{\theta}{\gamma},
& \theta\leq\gamma\lambda,\\[6pt]
0,
& \theta>\gamma\lambda.
\end{cases}
}
\tag{2.2}
$$

ここで

$$
\lambda\geq0,
\qquad
\gamma>1
$$

である。

この罰則の考え方は、その**微分**を見ることで理解できる。

MCPはまず、lassoと同じ強さで罰則を課し始める。しかしその罰則の強さを連続的に弱めていき、

$$
\theta>\gamma\lambda
$$

となると、罰則の増加率は

$$
0
$$

になる。

この罰則の形状は Figure 1 に示されている。

---

MCPの考え方は、その**単変量問題における解**を考えることによっても理解できる。

$\mathbf y$ を $\mathbf x$ に回帰する単純線形回帰を考える。

罰則を課さない最小二乗解を

$$
z=n^{-1}\mathbf x^\prime\mathbf y
$$

とする。

ここで、$\mathbf x$ は

$$
\mathbf x^\prime\mathbf x=n
$$

となるよう標準化されていることを思い出されたい。

この単純線形回帰問題について、MCP推定量は次の閉形式を持つ。

$$
\boxed{
\hat{\beta}
=
f_{\mathrm{MCP}}(z,\lambda,\gamma)
=
\begin{cases}
\dfrac{S(z,\lambda)}
{1-1/\gamma},
&
|z|\leq\gamma\lambda,
\\[10pt]
z,
&
|z|>\gamma\lambda.
\end{cases}
}
\tag{2.3}
$$

ここで $S$ は Donoho and Johnstone (1994) による**soft-thresholding operator**であり、$\lambda\geq0$ に対して次のように定義される。

$$
\boxed{
S(z,\lambda)
=
\begin{cases}
z-\lambda,
& z>\lambda,
\\
0,
& |z|\leq\lambda,
\\
z+\lambda,
& z<-\lambda.
\end{cases}
}
\tag{2.4}
$$

---

$S(z,\lambda)$ がlassoの単変量問題における解であることに注意すると、両者を比較することによって、MCPはlassoの解を、$\gamma$ に依存する量だけ**罰則なしの解の方向へ戻すようにスケーリングしている**ことが分かる。

$$
\gamma\rightarrow\infty
$$

のとき、MCPとlassoの解は一致する。

一方、

$$
\gamma\rightarrow1
$$

のとき、MCPの解はhard-thresholding推定量

$$
zI_{\{|z|>\lambda\}}
$$

となる。

したがって、単変量の意味では、MCPは Gao and Bruce (1997) の

**firm shrinkage estimator**

を与える。

---

計画行列（design matrix）が直交規格化されている（orthonormal）という特殊な場合には、

- subset selection は hard-thresholding と等価
- lasso は soft-thresholding と等価
- MCP は firm-thresholding と等価

である。

したがって、lassoはある種の**多変量soft-thresholding**を行う方法、subset selectionは**多変量hard-thresholding**を行う方法、MCPは**多変量firm-thresholding**を行う方法と考えることができる。

---

MCPのこの単変量解を、座標降下アルゴリズムにおいて目的関数の**各座標方向の最小化解**を得るために利用する。

ただし、この場合、「罰則なしの解」の役割を果たすのは、$x_j$ に関する**partial residual（部分残差）**を $x_j$ に罰則なしで回帰した結果である。

これを

$$
z_j
$$

と表す。

$j$ 番目の列または要素を除去した残りの部分を表すために、添字

$$
-j
$$

を導入する。

$x_j$ に関する部分残差は

$$
\mathbf r_{-j}
=
\mathbf y-X_{-j}\boldsymbol{\beta}_{-j}
$$

である。

ここで

$$
\boldsymbol{\beta}_{-j}
$$

は、$\boldsymbol{\beta}$ の直近に更新された値である。

したがって、反復 $m$ の第 $j$ ステップでは、次の3つの計算を行う。

### 1. $z_j$ を計算する

$$
\boxed{
z_j
=
n^{-1}\mathbf x_j^\prime\mathbf r_{-j}
=
n^{-1}\mathbf x_j^\prime\mathbf r
+
\beta_j^{(m)}
}
$$

### 2. $\beta_j$ を更新する

$$
\boxed{
\beta_j^{(m+1)}
\leftarrow
f_{\mathrm{MCP}}(z_j,\lambda,\gamma)
}
$$

### 3. 残差を更新する

$$
\boxed{
\mathbf r
\leftarrow
\mathbf r
-
\left(
\beta_j^{(m+1)}
-
\beta_j^{(m)}
\right)\mathbf x_j
}
$$

最後のステップによって、$\mathbf r$ が常に**現在の回帰係数に対応した最新の残差**を保持することが保証される。

$z_j$ は、$x_j$ を部分残差に回帰することによっても、現在の残差に回帰することによっても得ることができる。

しかし、現在の残差を用いる方が計算上効率的である。

なぜなら、各更新のたびに部分残差を再計算する必要がないからである。

---

# 2.2 SCAD

Fan and Li (2001) が区間

$$
[0,\infty)
$$

上で定義したSCAD罰則は、次のように与えられる。

$$
\boxed{
p_{\lambda,\gamma}(\theta)
=
\begin{cases}
\lambda\theta,
&
\theta\leq\lambda,
\\[6pt]
\dfrac{
\gamma\lambda\theta
-
0.5(\theta^2+\lambda^2)
}{
\gamma-1
},
&
\lambda<\theta\leq\gamma\lambda,
\\[12pt]
\dfrac{
\lambda^2(\gamma^2-1)
}{
2(\gamma-1)
},
&
\theta>\gamma\lambda.
\end{cases}
}
$$

その微分は

$$
\boxed{
p'_{\lambda,\gamma}(\theta)
=
\begin{cases}
\lambda,
&
\theta\leq\lambda,
\\[6pt]
\dfrac{\gamma\lambda-\theta}{\gamma-1},
&
\lambda<\theta\leq\gamma\lambda,
\\[10pt]
0,
&
\theta>\gamma\lambda.
\end{cases}
}
\tag{2.5}
$$

ここで

$$
\lambda\geq0,
\qquad
\gamma>2.
$$

この罰則の考え方はMCPと似ている。

どちらの罰則も、まずlassoと同じ割合で罰則を課し始め、$\theta$ が0から離れるにつれてその罰則率を低下させ、最終的に0にする。

両者の違いは、その**遷移の仕方**にある。

SCAD罰則についても Figure 1 に図示されている。

---

SCAD罰則付き回帰係数の単変量解は以下のとおりである。

ここでも $z$ は罰則なし線形回帰の解である。

$$
\boxed{
\hat{\beta}
=
f_{\mathrm{SCAD}}(z,\lambda,\gamma)
=
\begin{cases}
S(z,\lambda),
&
|z|\leq2\lambda,
\\[8pt]
\dfrac{
S\!\left(z,\dfrac{\gamma\lambda}{\gamma-1}\right)
}{
1-\dfrac{1}{\gamma-1}
},
&
2\lambda<|z|\leq\gamma\lambda,
\\[14pt]
z,
&
|z|>\gamma\lambda.
\end{cases}
}
\tag{2.6}
$$

この解は、MCPの解、すなわちfirm shrinkage推定量と似ているが、同一ではない。

両方の罰則とも、soft-thresholdingによる解を、罰則なしの解の方向へ再スケーリングする。

しかし、SCADでは

$$
\gamma\rightarrow\infty
$$

のときsoft-thresholdingが極限となる一方、

$$
\gamma\rightarrow2
$$

としても、hard-thresholdingが極限とはならない。

この単変量解は、SCADに対する座標降下アルゴリズムにおいて、MCPの場合の

$$
f_{\mathrm{MCP}}(z_j,\lambda,\gamma)
$$

と同じ役割を果たす。

---

# 2.3 収束
## 2.3. Convergence

次に、SCADおよびMCPに対する座標降下アルゴリズムの収束について考える。

まず、次の補題から始める。

---

### 補題1（Lemma 1）

式 (2.1) で定義した目的関数 $Q$ を、$\boldsymbol{\beta}$ の残りの要素を固定したうえで、単一の変数 $\beta_j$ の関数として見たものを

$$
Q_{j,\lambda,\gamma}(\boldsymbol{\beta})
$$

とする。

SCAD罰則について

$$
\gamma>2
$$

であり、MCPについて

$$
\gamma>1
$$

であるならば、

$$
Q_{j,\lambda,\gamma}(\boldsymbol{\beta})
$$

は、すべての $j$ について、$\beta_j$ の**凸関数**である。

---

この補題から、SCADおよびMCPに対する座標降下アルゴリズムについて、次の収束性を示すことができる。

---

### 命題1（Proposition 1）

SCADおよびMCPに対する座標降下アルゴリズムの各反復で得られる回帰係数の系列を

$$
\{\boldsymbol{\beta}^{(k)}\}
$$

とする。

すべての

$$
k=0,1,2,\ldots
$$

について、

$$
\boxed{
Q_{\lambda,\gamma}
\left(
\boldsymbol{\beta}^{(k+1)}
\right)
\leq
Q_{\lambda,\gamma}
\left(
\boldsymbol{\beta}^{(k)}
\right)
}
$$

が成立する。

さらに、この系列は、$Q_{\lambda,\gamma}$ の

- **局所最小点（local minimum）**
- **座標ごとの意味での大域的最小点（global coordinate-wise minimum）**

の両方である点へ収束することが保証される。

---

SCADおよびMCPの罰則関数は凸ではない。

したがって、本論文で提案する座標降下アルゴリズムもLLAアルゴリズムも、一般には**大域的最小値へ収束することは保証されない**。

しかし、目的関数 $Q$ が非凸な罰則項を含んでいたとしても、$\boldsymbol{\beta}$ に関して目的関数全体が凸になることはあり得る。

特に、

$$
c_*
$$

を

$$
n^{-1}X^\prime X
$$

の最小固有値とすると、MCPの目的関数は

$$
\boxed{
\gamma>\frac{1}{c_*}
}
$$

ならば凸であり、SCADの目的関数は

$$
\boxed{
\gamma>1+\frac{1}{c_*}
}
$$

ならば凸である［Zhang (2010)］。

この条件が満たされている場合、座標降下アルゴリズムは**大域的最小値**へ収束する。

Section 4 では、この問題について高次元設定との関連からさらに議論する。

---

# 2.4 パスに沿った最適化と初期値
## 2.4. Pathwise optimization and initial values

通常、我々が関心を持つのは、単一の $\lambda$ に対する

$$
\hat{\boldsymbol{\beta}}
$$

だけではない。

すべての罰則付き係数が0になる最大値

$$
\lambda_{\max}
$$

から、

$$
\lambda=0
$$

または、モデルが過度に大きくなる、あるいは識別不能となる最小値

$$
\lambda_{\min}
$$

までの、一連の $\lambda$ の値について $\hat{\boldsymbol{\beta}}$ を得たい。

目的関数が**狭義凸（strictly convex）**である場合、推定された回帰係数は

$$
\lambda\in[\lambda_{\min},\lambda_{\max}]
$$

に関して連続的に変化し、$\lambda$ によって正則化された**解パス**を形成する。

このようなパスの例は Figures 2 および 5 に示されている。

---

目的関数が狭義凸であれば係数パスは連続であるため、初期値を選ぶ合理的な方法は、解パスの一方の端から開始し、直前の $\lambda$ の値で得られた推定値

$$
\hat{\boldsymbol{\beta}}
$$

を、次の $\lambda$ の値における初期値として使用することである。

MCPおよびSCADについては、

**すべての罰則付き回帰係数が0になる最小の $\lambda$**

である

$$
\lambda_{\max}
$$

を容易に求めることができる。

式 (2.3) および (2.6) より、

$$
\boxed{
\lambda_{\max}=z_{\max}
}
$$

であることが分かる。ここで原文では

$$
\boxed{
z_{\max}
=
\max_j
\left\{
n^{-1}\mathbf x_j^\prime\mathbf y
\right\}
}
$$

と書かれている。

ロジスティック回帰の場合（Section 3）も、

$$
\lambda_{\max}
=
\max_j\{z_j\}
$$

である。

ただし、この場合 $z_j$ は式 (3.4) によって定義され、二次近似は**切片のみを含むモデル（intercept-only model）**を基準として行われる。

したがって、原文では、

$$
\lambda_{\min}
$$

から

$$
\boldsymbol{\beta}^{(0)}=\mathbf 0
$$

として開始し、

$$
\lambda_{\max}
$$

の方向へ進むことによって、初期値が解から大きく離れることが決してないようにできる、と述べている。

---

本論文のすべての数値計算結果では、Friedman, Hastie and Tibshirani (2010) の方法に従い、対数尺度上で等間隔に配置した

$$
100
$$

個の $\lambda$ のグリッドに沿って解を計算する。

---

## 原文上の注意点

2.4には少なくとも2点、数理的には不自然な記述があります。ただし上では**原文をそのまま翻訳**しました。

第一に、本文では

$$
z_{\max}
=
\max_j\{n^{-1}x_j^\prime y\}
$$

と印刷されていますが、soft-thresholdingの条件から通常期待されるのは絶対値を含む

$$
\max_j|n^{-1}x_j^\prime y|
$$

です。

第二に、本文には

> starting at $\lambda_{\min}$ with $\beta^{(0)}=0$ and proceeding toward $\lambda_{\max}$

とあります。しかし、直前では「$\lambda_{\max}$ が全係数を0にする値」と定義されているため、warm start の通常の論理では

$$
\lambda_{\max}
\longrightarrow
\lambda_{\min}
$$

と進む方が自然です。

したがって、ここは**論文原文の誤植である可能性があります**。ただし、これは原文外の数学的判断であり、翻訳本文には反映せず区別しました。

## Prompt:
2026/9/6 23:09:22

> 座標降下アルゴリズム

LLAと座標効果アルゴリズムを比較して
それぞれのアルゴリムについて説明してください

## Response:
2026/9/6 23:09:27

> **比較を整理する**
>
> この論文では、座標降下法が各係数を個別更新し、LLAが非凸罰則を重み付きLASSOへ近似してLARSで解いています。前者は高速、後者は一括更新で、実験では両者が同様の解へ収束しています。
>
> **比較を整理した**
>
> 
>
> 16s考えました

ここでいう「座標効果アルゴリズム」は、論文中の **座標降下法（coordinate descent; CD）** のこととして説明します。

この論文における LLA と座標降下法の最大の違いは、

$$
\boxed{\text{LLA：非凸罰則そのものを近似して、lasso問題として解く}}
$$

のに対して、

$$
\boxed{\text{CD：非凸罰則を近似せず、係数を1個ずつ直接最適化する}}
$$

という点です。Breheny & Huang は、SCAD/MCPに対して後者を使うことで大幅な高速化が可能だと主張しています。`nihms332857.pdf`

---

# 1. そもそも何を解いているのか

線形回帰では、両アルゴリズムとも同じ目的関数

$$
Q_{\lambda,\gamma}(\boldsymbol\beta)
=
\frac{1}{2n}
\|\mathbf y-X\boldsymbol\beta\|_2^2
+
\sum_{j=1}^p
p_{\lambda,\gamma}(|\beta_j|)
$$

を小さくすることを目的とします。

ここで、

- $n$：標本数
- $p$：説明変数数
- $X\in\mathbb R^{n\times p}$：デザイン行列
- $\boldsymbol\beta=(\beta_1,\dots,\beta_p)^\top$：回帰係数
- $p_{\lambda,\gamma}$：SCADやMCPの罰則
- $\lambda$：罰則の強さ
- $\gamma$：非凸性・罰則形状を決めるパラメータ

です。

問題は、SCADやMCPの場合、

$$
p_{\lambda,\gamma}
$$

が**非凸**であることです。

したがって、lassoのような単純な凸最適化問題にはなりません。

---

# 2. LLAとは何か

LLA は

**Local Linear Approximation**
（局所線形近似）

です。

Zou & Li (2008) が提案した方法で、この論文では座標降下法の比較対象になっています。`nihms332857.pdf`

## 基本アイデア

SCAD/MCPの非凸罰則

$$
p_{\lambda,\gamma}(|\beta_j|)
$$

を、現在の推定値 $\beta_j^{(m)}$ の周辺で**一次近似**します。

Taylor展開的に、

$$
p_{\lambda,\gamma}(|\beta_j|)
\approx
p_{\lambda,\gamma}(|\beta_j^{(m)}|)
+
p'_{\lambda,\gamma}(|\beta_j^{(m)}|)
\left(
|\beta_j|-|\beta_j^{(m)}|
\right).
$$

$\beta$ に依存しない定数項を捨てると、

$$
p_{\lambda,\gamma}(|\beta_j|)
\approx
p'_{\lambda,\gamma}(|\beta_j^{(m)}|)
|\beta_j|.
$$

したがって目的関数は、

$$
Q(\boldsymbol\beta)
\approx
\frac{1}{2n}
\|\mathbf y-X\boldsymbol\beta\|^2
+
\sum_{j=1}^p
w_j^{(m)}|\beta_j|,
$$

ただし

$$
w_j^{(m)}
=
p'_{\lambda,\gamma}
\left(
|\beta_j^{(m)}|
\right)
$$

となります。

これは何かというと、

$$
\boxed{\text{重み付きlasso}}
$$

です。

---

# 3. LLAアルゴリズムの流れ

現在の推定値を

$$
\boldsymbol\beta^{(m)}
$$

とします。

まずSCAD/MCP罰則の傾きを計算します。

$$
w_j^{(m)}
=
p'_{\lambda,\gamma}
(|\beta_j^{(m)}|).
$$

次に、

$$
\boldsymbol\beta^{(m+1)}
=
\arg\min_\beta
\left\{
\frac{1}{2n}
\|\mathbf y-X\beta\|^2
+
\sum_jw_j^{(m)}|\beta_j|
\right\}
$$

を解きます。

つまり、元々の非凸問題を直接解かず、

> 現在地で非凸罰則を直線に置き換える  
> ↓  
> weighted lasso を解く  
> ↓  
> 新しい $\beta$ を得る  
> ↓  
> そこで再び罰則を線形化する

という操作を繰り返します。

イメージとしては

$$
\text{SCAD/MCP}
\rightarrow
\text{局所的に直線近似}
\rightarrow
\text{weighted lasso}
\rightarrow
\text{新しい推定値}
\rightarrow\cdots
$$

です。

この論文で扱っているLLA実装では、このweighted lasso問題を **LARS** を使って解きます。`nihms332857.pdf`

---

# 4. MCPで具体的に見ると

MCPの微分は

$$
p'_{\lambda,\gamma}(\theta)
=
\begin{cases}
\lambda-\dfrac{\theta}{\gamma},
&
0\leq\theta\leq\gamma\lambda,\\[6pt]
0,
&
\theta>\gamma\lambda.
\end{cases}
$$

です。

したがってLLAでは、

$$
w_j^{(m)}
=
\begin{cases}
\lambda-\dfrac{|\beta_j^{(m)}|}{\gamma},
&
|\beta_j^{(m)}|
\leq
\gamma\lambda,
\\[6pt]
0,
&
|\beta_j^{(m)}|>\gamma\lambda.
\end{cases}
$$

となります。

つまり、

### $\beta_j^{(m)}$ が小さい

$$
w_j^{(m)}\approx\lambda
$$

なのでlassoに近い強い罰則。

### $\beta_j^{(m)}$ が大きい

$$
w_j^{(m)}
$$

が小さくなる。

### 十分大きい

$$
|\beta_j^{(m)}|>\gamma\lambda
$$

なら

$$
w_j^{(m)}=0.
$$

つまりその係数にはほぼ罰則をかけません。

これによってlassoの「大きな係数まで縮めてしまう」というバイアスを減らしています。

---

# 5. LLAの問題点

この論文で問題にしているのは、主に**計算量**です。

LLAでは各反復でweighted lasso問題を解かなければなりません。

しかも論文の実装では、その更新のために**解パスを追跡するLARS**を使います。

例えば、

- $\lambda$ を100個
- 各 $\lambda$ で平均10回反復

とすると、

$$
100\times10=1000
$$

回のlasso path計算が必要になる、と論文は説明しています。`nihms332857.pdf`

ここがかなり重い。

注意点として、これは特に**この論文で比較しているLLA＋LARS実装**についての話です。LLAという考え方自体が必ずLARSを使わなければならないわけではありません。

---

# 6. 座標降下法とは何か

次にCoordinate Descentです。

発想はLLAとは全く異なります。

$$
\boldsymbol\beta
=
(\beta_1,\beta_2,\ldots,\beta_p)
$$

を全部一度に最適化するのではなく、

$$
\boxed{\beta_1\rightarrow\beta_2\rightarrow\cdots\rightarrow\beta_p}
$$

と、**1変数ずつ最適化する**方法です。

たとえば $\beta_j$ を更新するときには、

$$
\beta_1,\ldots,\beta_{j-1},
\beta_{j+1},\ldots,\beta_p
$$

をすべて固定します。

そして

$$
\min_{\beta_j}
Q(\beta_1,\ldots,\beta_j,\ldots,\beta_p)
$$

という**1次元問題**だけを解きます。

---

# 7. なぜSCAD/MCPと相性がいいのか

SCAD/MCPには非常に都合のよい性質があります。

多変量では閉形式解がないのに、**1変量問題なら閉形式解が存在する**からです。

これがこの論文の核心です。

MCPなら、

$$
\hat\beta
=
f_{\mathrm{MCP}}(z,\lambda,\gamma)
$$

を直接計算できます。

具体的には

$$
f_{\mathrm{MCP}}(z,\lambda,\gamma)
=
\begin{cases}
\dfrac{S(z,\lambda)}
{1-1/\gamma},
&
|z|\leq\gamma\lambda,
\\[10pt]
z,
&
|z|>\gamma\lambda,
\end{cases}
$$

ここで

$$
S(z,\lambda)
=
\operatorname{sgn}(z)(|z|-\lambda)_+
$$

はsoft-thresholdingです。

したがって、

$$
\boxed{\text{1座標だけならMCPを厳密に解ける}}
$$

ということになります。

LLAのように「MCPをlassoで近似する」必要がありません。

---

# 8. 座標降下法での $z_j$

ここが重要です。

$\beta_j$ 以外を固定します。

第 $j$ 変数以外による予測を取り除いた**部分残差**を

$$
\mathbf r_{-j}
=
\mathbf y
-
X_{-j}\boldsymbol\beta_{-j}
$$

とします。

すると、

$$
\mathbf r_{-j}
\approx
\mathbf x_j\beta_j+\text{noise}
$$

です。

そこで

$$
z_j
=
\frac1n
\mathbf x_j^\top\mathbf r_{-j}
$$

を計算します。

$x_j^\top x_j=n$ と標準化しているので、これは

$$
\mathbf r_{-j}
$$

を

$$
\mathbf x_j
$$

に単回帰したときの**無罰則最小二乗推定値**です。

その $z_j$ をMCPの単変量解に突っ込むだけです。

$$
\boxed{
\beta_j
\leftarrow
f_{\mathrm{MCP}}(z_j,\lambda,\gamma)
}
$$

SCADなら

$$
\boxed{
\beta_j
\leftarrow
f_{\mathrm{SCAD}}(z_j,\lambda,\gamma)
}
$$

です。

---

# 9. 座標降下法の1ステップ

論文のMCPの場合、反復 $m$ で $j$ 番目の係数を更新するとき、

まず

$$
z_j
=
\frac1n x_j^\top r_{-j}
$$

を計算。

次に

$$
\beta_j^{(m+1)}
=
f_{\mathrm{MCP}}
(z_j,\lambda,\gamma)
$$

とします。

そして残差を更新します。

$$
r
\leftarrow
r-
\left(
\beta_j^{(m+1)}
-
\beta_j^{(m)}
\right)x_j.
$$

実装上は、毎回

$$
r_{-j}
$$

をゼロから計算する必要はありません。

現在の残差

$$
r=y-X\beta
$$

を保持しておけば、

$$
z_j
=
\frac1n x_j^\top r+\beta_j
$$

と計算できます。

この工夫が高速化に重要です。`nihms332857.pdf`

---

# 10. アルゴリズム全体

座標降下法は概念的には次のようになります。

$$
\beta^{(0)}=(0,\ldots,0)
$$

から始めます。

その後、

$$
j=1,2,\ldots,p
$$

について順番に

$$
z_j
=
\frac1n x_j^\top r+\beta_j
$$

を計算し、

$$
\beta_j
\leftarrow
f(z_j,\lambda,\gamma)
$$

で更新し、

$$
r
\leftarrow
r-(\beta_j^{new}-\beta_j^{old})x_j
$$

とします。

$j=1,\ldots,p$ を一周したら、

$$
\beta_1\rightarrow\beta_2\rightarrow\cdots\rightarrow\beta_p
$$

再び最初から回ります。

$$
\text{収束するまで反復}
$$

です。

---

# 11. LLAとCDの本質的な違い

かなり重要なので表にします。

| | LLA | Coordinate Descent |
|---|---|---|
| 正式名 | Local Linear Approximation | Coordinate Descent |
| 基本戦略 | **罰則を近似** | **パラメータを1個ずつ最適化** |
| SCAD/MCPをどう扱うか | 局所的にweighted $L_1$ に置換 | SCAD/MCPをそのまま使う |
| 1回の更新 | weighted lasso問題を解く | 1個の $\beta_j$ を閉形式で更新 |
| この論文で使う内部アルゴリズム | LARS | soft/firm/SCAD thresholding |
| 一度に動かす係数 | 原則として全係数 | 1係数 |
| 罰則の近似 | あり | なし |
| 高次元での計算効率 | 重い | 非常に軽い |
| 1 sweepの計算量 | LARSに依存 | $O(np)$ |
| 非凸問題の大域解保証 | 一般にはなし | 一般にはなし |
| CDについての収束保証 | - | coordinate-wise minimumへ収束 |
| 本論文での評価 | 比較対象 | 提案手法 |

---

# 12. 「全部同時に解くLLA」と「1つずつ解くCD」

直感的には、LLAは

$$
(\beta_1,\beta_2,\ldots,\beta_p)
$$

について、

> 今の位置でSCAD/MCPの地形を簡単な凸問題に描き直す  
> → その近似問題全体を解く

方法です。

一方CDは、

> 元のSCAD/MCPの地形を変形せず、  
> $\beta_1$ 方向だけ最小化  
> → 次に$\beta_2$方向  
> → …  
> → 何周もする

方法です。

つまり、

$$
\boxed{
\text{LLA = objective function側を簡単にする}
}
$$

対して

$$
\boxed{
\text{CD = optimizationの次元を簡単にする}
}
$$

と理解するとよいです。

---

# 13. なぜCDは速いのか

論文は理由を2つ挙げています。

### 理由1：1回の更新が $O(n)$

1個の係数について必要なのは主として

$$
x_j^\top r
$$

です。

これは長さ $n$ の内積なので

$$
O(n)
$$

です。

$p$ 個全部について更新しても、

$$
O(np).
$$

---

### 理由2：warm startを使える

正則化パラメータについて

$$
\lambda_1>\lambda_2>\lambda_3>\cdots
$$

と少しずつ変えていくと、

$$
\hat\beta(\lambda_k)
$$

と

$$
\hat\beta(\lambda_{k+1})
$$

は、局所凸な領域では通常かなり近い。

したがって、

$$
\hat\beta(\lambda_k)
$$

を次の問題の初期値にします。

つまり、

$$
\boxed{
\beta^{(0)}_{\lambda_{k+1}}
=
\hat\beta_{\lambda_k}
}
$$

です。

これを**warm start**といいます。

最初から解く必要がないので、数回のcoordinate sweepで収束することがあります。

---

# 14. 収束についてはCDにも弱点がある

ここは重要です。

「CDは高速だから大域最適解が必ず得られる」という意味ではありません。

SCAD/MCPは非凸なので、

$$
Q(\beta)
$$

全体も非凸になることがあります。

したがって論文は、

> CDもLLAも一般にはglobal minimumへの収束は保証されない

と明記しています。`nihms332857.pdf`

ただしCDについては、この論文では次の性質を証明しています。

MCPなら

$$
\gamma>1,
$$

SCADなら

$$
\gamma>2
$$

のとき、他の係数を固定した

$$
Q_j(\beta_j)
$$

は各座標方向について凸です。

そのためCDは

$$
Q(\beta^{(k+1)})
\leq
Q(\beta^{(k)})
$$

と目的関数を単調に低下させ、

最終的には

- coordinate-wise global minimum
- local minimum

である点に収束します。

ただし、

$$
\boxed{
\text{local minimum}
\neq
\text{global minimum}
}
$$

の場合があります。

---

# 15. 目的関数全体が凸ならどうなるか

もしMCPについて

$$
\gamma>\frac{1}{c_*},
$$

SCADについて

$$
\gamma>1+\frac{1}{c_*},
$$

ただし

$$
c_*=
\lambda_{\min}
\left(
\frac1nX^\top X
\right)
$$

なら、目的関数全体が凸になります。

この場合CDは

$$
\boxed{\text{global minimum}}
$$

へ収束します。

したがって非凸罰則だからといって、目的関数全体が必ず非凸とは限りません。

---

# 16. 実際どれくらい速いのか

この論文の Section 5.1 ではLLAとCDを直接比較しています。

シミュレーションでは、比較したすべての設定で、収束精度の範囲内において

$$
\text{LLAとCDは同じ解パス}
$$

に収束しました。

しかし計算時間には大きな差があります。

Figure 3では、

$$
\boxed{\text{CDはLLAより最大約1000倍高速}}
$$

であり、調べた全設定で

$$
\boxed{\text{少なくとも100倍高速}}
$$

でした。

強い相関

$$
\rho=0.9
$$

がある場合にはCD自体は無相関の場合より2〜5倍遅くなりますが、それでもLLAより少なくとも100倍高速だったと報告されています。`nihms332857.pdf`

実データの遺伝的関連解析でも、

$$
\text{LLA}=17\text{分}
$$

に対し、

$$
\text{CD}=2.5\text{秒}
$$

で1本のMCP解パスを計算しています。`nihms332857.pdf`

---

# 17. ではLLAは何のためにあるのか

LLAの利点は、**非凸罰則問題を既存のlasso計算技術に変換できる**点です。

つまりSCAD/MCP専用の解法を一から構築しなくても、

$$
\text{非凸罰則}
\rightarrow
\text{weighted lasso}
$$

と変換して、すでに確立されていたlassoソルバーを再利用できます。

特にZou & Liでは、良い初期推定量が得られるなら**one-step LLA**という考え方にも理論的意味があります。

一方、本論文の目的は、

> 近似を介さずにSCAD/MCPの目的関数そのものを効率よく最小化する

ことです。

そのためCDを採用しています。

---

# 18. 一言で比較すると

最も重要なのはここです。

### LLA

$$
\boxed{
\text{SCAD/MCPを毎回lassoに近似して、lassoソルバーで解く}
}
$$

したがって、

$$
\text{外側の反復}
\times
\text{lasso問題}
$$

になります。

### Coordinate Descent

$$
\boxed{
\text{SCAD/MCPの1次元閉形式解を直接使って、係数を1つずつ更新する}
}
$$

したがって、

$$
\text{安価な1次元更新}
\times
p
\times
\text{少数のsweep}
$$

になります。

この論文がCDを採用する理由は、単に「座標降下法が流行っていたから」ではありません。

**SCAD/MCPでは1変量問題が解析的に解けるため、その構造をcoordinate descentが非常にうまく利用できるから**です。これがアルゴリズム上の本質です。`nihms332857.pdf`

## Prompt:
2026/9/6 23:31:00

> design matrix

ってなんだっけ？直交きかくかされているとはどういうことでどういう意味か？

## Response:
2026/9/6 23:31:10

ここでいう **design matrix（デザイン行列）** と、**直交規格化（orthonormal）** は別々に押さえると分かりやすいです。

## 1. デザイン行列 $X$ とは

線形回帰

$$
\mathbf y = X\boldsymbol\beta+\boldsymbol\varepsilon
$$

における $X$ がデザイン行列です。

観測数が $n$、説明変数が $p$ 個なら、

$$
X=
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p}\\
x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \vdots & & \vdots\\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}
\in\mathbb R^{n\times p}.
$$

列

$$
\mathbf x_j=
\begin{pmatrix}
x_{1j}\\
\vdots\\
x_{nj}
\end{pmatrix}
$$

が第 $j$ 説明変数です。

たとえば、

- 列1：身長
- 列2：体重
- 列3：年齢

という感じです。

---

# 2. 「標準化」と「直交規格化」は違う

この論文ではまず各列を

$$
\sum_{i=1}^n x_{ij}=0
$$

かつ

$$
\frac1n\sum_{i=1}^n x_{ij}^2=1
$$

となるように標準化しています。`nihms332857.pdf`

つまり、

$$
\mathbf x_j^\top \mathbf x_j=n.
$$

これは単に、

- 平均0
- 分散1

にしているだけです。

この時点では、異なる説明変数どうしは相関していて構いません。

つまり一般には

$$
\mathbf x_j^\top\mathbf x_k\neq0
\qquad(j\neq k)
$$

です。

---

# 3. 直交しているとは

2つの列ベクトル

$$
\mathbf x_j,\mathbf x_k
$$

が直交するとは、

$$
\boxed{
\mathbf x_j^\top\mathbf x_k=0
}
$$

ということです。

幾何学的には「90度」です。

2次元なら、

$$
\mathbf x_1=
\begin{pmatrix}
1\\0
\end{pmatrix},
\qquad
\mathbf x_2=
\begin{pmatrix}
0\\1
\end{pmatrix}
$$

なら

$$
\mathbf x_1^\top\mathbf x_2
=
1\cdot0+0\cdot1=0.
$$

なので直交しています。

---

# 4. 回帰での「直交」の意味

説明変数が中心化されているとき、

$$
\frac1n\mathbf x_j^\top\mathbf x_k
$$

は essentially 共分散です。

さらに両方分散1に標準化されていれば、

$$
\frac1n\mathbf x_j^\top\mathbf x_k
$$

は相関係数です。

したがって、

$$
\mathbf x_j^\top\mathbf x_k=0
$$

は、

$$
\boxed{\text{説明変数 }j,k\text{ が無相関}}
$$

という意味になります。

注意すると、「統計的独立」とまでは言っていません。直交は基本的には**線形相関が0**という意味です。

---

# 5. 「規格化」とは

ベクトルの長さを揃えることです。

通常の数学では「単位長」にするなら

$$
\|\mathbf x_j\|_2=1
$$

つまり

$$
\mathbf x_j^\top\mathbf x_j=1
$$

とします。

これが通常の意味での normalized です。

そして、

$$
\mathbf x_j^\top\mathbf x_k
=
\begin{cases}
1,&j=k\\
0,&j\neq k
\end{cases}
$$

なら

$$
X^\top X=I
$$

となり、これを **orthonormal** と呼びます。

---

# 6. ただしこの論文ではスケーリングが少し違う

この論文では各列について

$$
\frac1n\mathbf x_j^\top\mathbf x_j=1
$$

つまり

$$
\mathbf x_j^\top\mathbf x_j=n
$$

という規格化を使っています。

したがって、この論文のスケールで「直交規格化されたデザイン行列」と言えば、

$$
\boxed{
\frac1nX^\top X=I_p
}
$$

すなわち

$$
\boxed{
X^\top X=nI_p
}
$$

です。

成分で書くと、

$$
\frac1n\mathbf x_j^\top\mathbf x_k
=
\begin{cases}
1,& j=k\\
0,& j\neq k.
\end{cases}
$$

つまり、

- 各変数の分散は1
- 異なる変数間の相関は0

です。

---

# 7. 具体例

たとえば $n=4,\ p=2$ として、

$$
\mathbf x_1=
\begin{pmatrix}
1\\
1\\
-1\\
-1
\end{pmatrix},
\qquad
\mathbf x_2=
\begin{pmatrix}
1\\
-1\\
1\\
-1
\end{pmatrix}.
$$

まず長さは、

$$
\mathbf x_1^\top\mathbf x_1=4,
$$

$$
\mathbf x_2^\top\mathbf x_2=4.
$$

つまり $n=4$ なので

$$
\frac14\mathbf x_j^\top\mathbf x_j=1.
$$

また、

$$
\mathbf x_1^\top\mathbf x_2
=
1-1-1+1
=
0.
$$

したがって、

$$
X^\top X
=
\begin{pmatrix}
4&0\\
0&4
\end{pmatrix}
=
4I.
$$

これは、この論文のスケーリングでは直交規格化されています。

---

# 8. なぜ直交すると問題が簡単になるのか

これがMCP論文では非常に重要です。

最小二乗部分を展開すると、

$$
\frac1{2n}\|\mathbf y-X\boldsymbol\beta\|^2
$$

です。

展開すると、

$$
=
\frac1{2n}
\left(
\mathbf y^\top\mathbf y
-
2\boldsymbol\beta^\top X^\top\mathbf y
+
\boldsymbol\beta^\top X^\top X\boldsymbol\beta
\right).
$$

一般には、

$$
\boldsymbol\beta^\top X^\top X\boldsymbol\beta
$$

の中に、

$$
\beta_j\beta_k\mathbf x_j^\top\mathbf x_k
$$

という**変数間の交差項**があります。

つまり、

$$
\beta_1,\beta_2,\ldots,\beta_p
$$

が互いに絡み合っています。

---

しかし直交規格化されていれば、

$$
X^\top X=nI.
$$

なので、

$$
\boldsymbol\beta^\top X^\top X\boldsymbol\beta
=
n\boldsymbol\beta^\top\boldsymbol\beta
=
n\sum_j\beta_j^2.
$$

交差項が全部消えます。

したがって目的関数が

$$
Q(\boldsymbol\beta)
=
\sum_{j=1}^p Q_j(\beta_j)+\text{constant}
$$

のように、**各係数ごとの独立な1次元問題へ完全に分離**できます。

これが極めて大きい。

---

# 9. だから thresholding と一致する

この論文が

> orthonormal design matrix の特殊ケースでは、subset selection は hard-thresholding、lasso は soft-thresholding、MCP は firm-thresholding

と述べている理由がここです。`nihms332857.pdf`

直交なら各 $j$ について

$$
z_j=\frac1n\mathbf x_j^\top\mathbf y
$$

だけを見ればよくなります。

### Lasso

$$
\hat\beta_j
=
S(z_j,\lambda)
$$

というsoft thresholding。

### MCP

$$
\hat\beta_j
=
f_{\mathrm{MCP}}(z_j,\lambda,\gamma)
$$

というfirm thresholding。

### subset selection

ある閾値を超えればそのまま採用、超えなければ0、というhard thresholding。

つまり、多変量回帰なのに

$$
\boxed{
p\text{個の独立した単回帰問題}
}
$$

として解けてしまいます。

---

# 10. 一般のデザイン行列ではそうならない

現実には普通、

$$
X^\top X\neq nI
$$

です。

例えば説明変数1と2が強く相関していれば、

$$
\mathbf x_1^\top\mathbf x_2
$$

は大きくなります。

そうすると、

$$
\beta_1
$$

を変えたときの最適値が

$$
\beta_2
$$

にも依存します。

だから多変量MCPを一発の式で解けない。

そこでcoordinate descentでは、

$$
\beta_2,\ldots,\beta_p
$$

を一旦固定して、

$$
\beta_1
$$

だけを解く。

次に、

$$
\beta_1,\beta_3,\ldots
$$

を固定して、

$$
\beta_2
$$

だけを解く。

これを繰り返します。

---

# 11. 一番重要な整理

混同しやすいので、ここだけ覚えるとよいです。

$$
\boxed{
\text{標準化}
\quad\Longrightarrow\quad
\frac1n x_j^\top x_j=1
}
$$

は**各列自身のスケールを揃えるだけ**。

一方、

$$
\boxed{
\text{直交}
\quad\Longrightarrow\quad
x_j^\top x_k=0,\quad j\neq k
}
$$

は**異なる列どうしが無相関**。

両方を合わせると、

$$
\boxed{
\frac1nX^\top X=I
}
$$

となり、これがこの論文で言っている **orthonormal design の特殊ケース**です。

そしてこの条件の意味は、

$$
\boxed{
\text{説明変数同士が干渉しないので、
多変量問題が1変数ずつに完全分離する}
}
$$

ということです。これが lasso = soft thresholding、MCP = firm thresholding と直接言える理由です。

## Prompt:
2026/9/6 23:37:55

> アルゴリズム全体

今回の私の手法ではADMM Fused　Lassoを用いているのですが、それをADMM　Fused　MCP　にしたい。どのように変わる？数式で検討して

## Response:
2026/9/6 23:38:05

> **組み立て直す**
>
> 融合ラッソでは差分にL1ペナルティを置きますが、融合MCPでは各差分にMCPを適用します。係数自体のL1を残すか、そこもMCPに替えるかで更新式が変わるため、まず定式化を分けて示します。
>
> **組み立て直した**
>
> 
>
> 38s考えました

はい。今回の変更はかなり明確に数式化できます。

結論から言うと、現在の ADMM Fused Lasso が

$$
\lambda_F\|D\beta\|_1
$$

という **L1 fusion penalty** を使っているなら、ADMM Fused MCP ではこれを

$$
\sum_k p_{\lambda_F,\gamma}(|(D\beta)_k|)
$$

に置き換えます。

重要なのは、ADMMで適切に変数分割すれば、

$$
\boxed{\beta\text{-update は基本的にそのまま}}
$$

で、

$$
\boxed{\text{差分変数の soft-thresholding だけを MCP の firm-thresholding に置き換える}}
$$

という構造になることです。

以下、導出します。

---

# 1. 現在の ADMM Fused Lasso

まず標準的な fused lasso を

$$
\min_{\beta\in\mathbb R^p}
\frac{1}{2n}\|y-X\beta\|_2^2
+
\lambda_1\|\beta\|_1
+
\lambda_F\|D\beta\|_1
\tag{1}
$$

とします。

ここで $D$ は差分行列です。

例えば1次元系列なら

$$
D=
\begin{pmatrix}
-1&1&0&\cdots&0\\
0&-1&1&\cdots&0\\
\vdots&&\ddots&\ddots&\vdots\\
0&\cdots&0&-1&1
\end{pmatrix},
$$

なので

$$
D\beta
=
\begin{pmatrix}
\beta_2-\beta_1\\
\beta_3-\beta_2\\
\vdots\\
\beta_p-\beta_{p-1}
\end{pmatrix}.
$$

したがって

$$
\lambda_F\|D\beta\|_1
=
\lambda_F
\sum_{j=1}^{p-1}
|\beta_{j+1}-\beta_j|.
$$

これは「隣り合う係数を同じ値にしよう」という罰則です。

---

# 2. ADMM用に変数分割する

次の変数を導入します。

$$
z=\beta,
\qquad
d=D\beta.
$$

すると (1) は

$$
\min_{\beta,z,d}
\frac{1}{2n}\|y-X\beta\|^2
+
\lambda_1\|z\|_1
+
\lambda_F\|d\|_1
$$

subject to

$$
\beta-z=0,
\qquad
D\beta-d=0.
$$

scaled dual variable を

$$
u,\quad v
$$

とし、ADMM penalty parameter を

$$
\rho_1,\quad \rho_F
$$

とすると、scaled augmented Lagrangian は本質的に

$$
\begin{aligned}
L
=&
\frac1{2n}\|y-X\beta\|^2
+\lambda_1\|z\|_1
+\lambda_F\|d\|_1\\
&+
\frac{\rho_1}{2}
\|\beta-z+u\|^2
+
\frac{\rho_F}{2}
\|D\beta-d+v\|^2.
\end{aligned}
$$

定数項は省いています。

---

# 3. Fused Lasso のADMM更新

## $\beta$-update

$$
\beta^{k+1}
=
\arg\min_\beta
\left\{
\frac1{2n}\|y-X\beta\|^2
+
\frac{\rho_1}{2}
\|\beta-z^k+u^k\|^2
+
\frac{\rho_F}{2}
\|D\beta-d^k+v^k\|^2
\right\}.
$$

微分を0にすると、

$$
\left(
\frac1nX^\top X
+
\rho_1I
+
\rho_FD^\top D
\right)\beta^{k+1}
$$

$$
=
\frac1nX^\top y
+
\rho_1(z^k-u^k)
+
\rho_FD^\top(d^k-v^k).
$$

したがって

$$
\boxed{
\beta^{k+1}
=
\left(
\frac1nX^\top X+\rho_1I+\rho_FD^\top D
\right)^{-1}
\left[
\frac1nX^\top y
+\rho_1(z^k-u^k)
+\rho_FD^\top(d^k-v^k)
\right]
}
\tag{2}
$$

です。

---

## $z$-update

$$
z^{k+1}
=
\arg\min_z
\left\{
\lambda_1\|z\|_1
+
\frac{\rho_1}{2}
\|\beta^{k+1}-z+u^k\|^2
\right\}.
$$

したがってsoft-thresholding

$$
\boxed{
z^{k+1}
=
S\left(
\beta^{k+1}+u^k,
\frac{\lambda_1}{\rho_1}
\right)
}
\tag{3}
$$

です。

---

## $d$-update

同様に

$$
d^{k+1}
=
\arg\min_d
\left\{
\lambda_F\|d\|_1
+
\frac{\rho_F}{2}
\|D\beta^{k+1}-d+v^k\|^2
\right\}
$$

なので、

$$
\boxed{
d^{k+1}
=
S\left(
D\beta^{k+1}+v^k,
\frac{\lambda_F}{\rho_F}
\right)
}
\tag{4}
$$

となります。

ここが **Fused Lasso のfusion部分**です。

---

# 4. Fused MCP に変える

ここからが本題です。

MCPを

$$
p_{\lambda,\gamma}(\theta)
=
\begin{cases}
\lambda\theta-\dfrac{\theta^2}{2\gamma},
&
0\leq\theta\leq\gamma\lambda,
\\[6pt]
\dfrac{\gamma\lambda^2}{2},
&
\theta>\gamma\lambda
\end{cases}
\tag{5}
$$

とします。

これは今回読んでいるBreheny-Huang論文のMCPそのものです。`nihms332857.pdf`

Fused Lasso の

$$
\lambda_F\|D\beta\|_1
$$

を

$$
\boxed{
\sum_{j=1}^{m}
p_{\lambda_F,\gamma}
\left(
|(D\beta)_j|
\right)
}
\tag{6}
$$

に置き換えます。

したがってモデルは

$$
\boxed{
\min_\beta
\frac1{2n}\|y-X\beta\|^2
+
\lambda_1\|\beta\|_1
+
\sum_j
p_{\lambda_F,\gamma}
(|(D\beta)_j|)
}
\tag{7}
$$

となります。

これをここでは **Fused MCP** と呼びます。

---

# 5. ADMMで何が変わるか

同じく

$$
z=\beta,\qquad d=D\beta
$$

と分割すれば、

$$
\min_{\beta,z,d}
\frac1{2n}\|y-X\beta\|^2
+
\lambda_1\|z\|_1
+
\sum_jp_{\lambda_F,\gamma}(|d_j|)
$$

subject to

$$
z=\beta,\qquad d=D\beta.
$$

ここで重要なのは、

$$
\boxed{\beta\text{-updateは(2)から全く変わらない}}
$$

ということです。

なぜならMCPの非凸性を全部 $d$ 側へ分離したからです。

$z$-updateも、$\beta$ 自身にはlassoを残すなら変わりません。

唯一大きく変わるのは

$$
\boxed{d\text{-update}}
$$

です。

---

# 6. Fused MCP の $d$-update

各成分について、

$$
a_j
=
(D\beta^{k+1}+v^k)_j
$$

と置きます。

すると、

$$
d_j^{k+1}
=
\arg\min_d
\left[
p_{\lambda_F,\gamma}(|d|)
+
\frac{\rho_F}{2}(d-a_j)^2
\right].
\tag{8}
$$

これを解けばよい。

---

# 7. MCP proximal operatorを導出する

まず

$$
0<d\leq\gamma\lambda_F
$$

の場合を考えます。

MCPは

$$
p_{\lambda_F,\gamma}(d)
=
\lambda_Fd-\frac{d^2}{2\gamma}.
$$

したがって

$$
h(d)
=
\lambda_Fd
-\frac{d^2}{2\gamma}
+
\frac{\rho_F}{2}(d-a)^2.
$$

微分すると、

$$
h'(d)
=
\lambda_F
-\frac{d}{\gamma}
+
\rho_F(d-a).
$$

0とおけば、

$$
\lambda_F
-\frac{d}{\gamma}
+
\rho_Fd-\rho_Fa=0,
$$

したがって

$$
\left(
\rho_F-\frac1\gamma
\right)d
=
\rho_Fa-\lambda_F.
$$

よって

$$
d
=
\frac{\rho_Fa-\lambda_F}
{\rho_F-1/\gamma}.
$$

分子・分母を $\rho_F$ で割ると、

$$
\boxed{
d
=
\frac{
a-\lambda_F/\rho_F
}{
1-1/(\rho_F\gamma)
}
}
$$

です。

負の場合も対称性を使えばsoft-thresholdingで書けます。

$$
\boxed{
d
=
\frac{
S(a,\lambda_F/\rho_F)
}{
1-1/(\rho_F\gamma)
}
}
\tag{9}
$$

です。

---

# 8. 完全なMCP thresholding

したがって、

$$
\rho_F\gamma>1
$$

を仮定すると、

$$
\boxed{
T_{\mathrm{MCP}}
(a;\lambda_F,\gamma,\rho_F)
=
\begin{cases}
0,
&
|a|\leq\dfrac{\lambda_F}{\rho_F},
\\[10pt]
\dfrac{
S(a,\lambda_F/\rho_F)
}{
1-\dfrac{1}{\rho_F\gamma}
},
&
\dfrac{\lambda_F}{\rho_F}
<
|a|
\leq
\gamma\lambda_F,
\\[14pt]
a,
&
|a|>\gamma\lambda_F.
\end{cases}
}
\tag{10}
$$

したがってFused MCPの更新は

$$
\boxed{
d_j^{k+1}
=
T_{\mathrm{MCP}}
\left(
(D\beta^{k+1}+v^k)_j;
\lambda_F,\gamma,\rho_F
\right)
}
\tag{11}
$$

です。

これがFused Lassoとの最大の違いです。

---

# 9. LassoとMCPを直接比較する

Fused Lassoでは、

$$
d=S(a,\lambda_F/\rho_F).
$$

つまり

$$
d=
\begin{cases}
0,
&
|a|\leq\lambda_F/\rho_F,
\\
\operatorname{sign}(a)
\left(
|a|-\lambda_F/\rho_F
\right),
&
|a|>\lambda_F/\rho_F.
\end{cases}
$$

大きな差分に対しても常に

$$
\lambda_F/\rho_F
$$

だけ縮めます。

一方MCPは

$$
|a|>\gamma\lambda_F
$$

なら

$$
\boxed{d=a}
$$

です。

つまり**大きなジャンプを全く縮小しない**。

これが Fused MCP を使う統計的な意味です。

---

# 10. change point の観点では何が変わるか

例えば真の係数が

$$
\beta=
(1,1,1,5,5,5)
$$

だったとします。

真のジャンプは

$$
D\beta
=
(0,0,4,0,0).
$$

Fused Lassoはジャンプ $4$ に対しても

$$
|4| \mapsto |4|-\text{threshold}
$$

と縮めます。

したがって境界前後の水準差が過小評価されやすい。

MCPならジャンプが十分大きく

$$
4>\gamma\lambda_F
$$

なら、

$$
4\mapsto4.
$$

したがって、

$$
\boxed{
\text{小さな差は0にするが、
大きな差はほぼそのまま残す}
}
$$

という挙動になります。

これはまさにMCPの「large coefficient biasを減らす」という思想を、**係数そのものではなく差分 $D\beta$** に適用したものです。

---

# 11. KKT条件を見るとさらに明確

Fused Lassoでは、差分

$$
d_j=(D\beta)_j
$$

について $d_j\neq0$ なら罰則の微分は

$$
\frac{\partial}{\partial d_j}
\lambda_F|d_j|
=
\lambda_F\operatorname{sign}(d_j).
$$

つまり $d_j$ がどれだけ大きくても

$$
\boxed{
\lambda_F
}
$$

のfusion forceがかかり続けます。

一方MCPでは、

$$
p'_{\lambda_F,\gamma}(|d_j|)
\operatorname{sign}(d_j)
$$

なので、

$$
s_j=
\begin{cases}
\lambda_F\operatorname{sign}(d_j)
-\dfrac{d_j}{\gamma},
&
0<|d_j|<\gamma\lambda_F,
\\[8pt]
0,
&
|d_j|>\gamma\lambda_F.
\end{cases}
$$

そして $d_j=0$ では

$$
s_j\in[-\lambda_F,\lambda_F].
$$

したがってstationarityは

$$
\boxed{
\frac1nX^\top(X\beta-y)
+
\lambda_1 q
+
D^\top s
=
0
}
\tag{12}
$$

です。

ここでlasso部分について

$$
q_j=
\begin{cases}
\operatorname{sign}(\beta_j),
&\beta_j\neq0,\\
[-1,1],
&\beta_j=0.
\end{cases}
$$

MCPでは大きな差分について

$$
s_j=0.
$$

つまり、

$$
\boxed{
\text{大きなchange pointにはfusion penaltyから力がかからない}
}
$$

ということです。

これはFused Lassoとの非常に大きな違いです。

---

# 12. ADMMアルゴリズム全体

したがって、Fused MCP ADMMは

$$
\boxed{
\begin{aligned}
\beta^{k+1}
&=
\left(
\frac1nX^\top X+\rho_1I+\rho_FD^\top D
\right)^{-1}\\
&\quad\times
\left[
\frac1nX^\top y
+
\rho_1(z^k-u^k)
+
\rho_FD^\top(d^k-v^k)
\right],
\\[4pt]
z^{k+1}
&=
S\left(
\beta^{k+1}+u^k,
\frac{\lambda_1}{\rho_1}
\right),
\\[4pt]
d_j^{k+1}
&=
T_{\mathrm{MCP}}
\left(
(D\beta^{k+1}+v^k)_j;
\lambda_F,\gamma,\rho_F
\right),
\\[4pt]
u^{k+1}
&=
u^k+\beta^{k+1}-z^{k+1},
\\
v^{k+1}
&=
v^k+D\beta^{k+1}-d^{k+1}.
\end{aligned}
}
\tag{13}
$$

つまり実装上はかなり美しい変更です。

現在のコードがADMM Fused Lassoなら、

$$
\boxed{
\texttt{soft\_threshold(D beta + v)}
}
$$

の部分を

$$
\boxed{
\texttt{MCP\_threshold(D beta + v)}
}
$$

へ変更するのが中心です。

---

# 13. ただし重要：$\rho_F\gamma>1$

MCPではここがFused Lassoと異なります。

$d$-subproblemのMCP領域では二階微分が

$$
-\frac1\gamma+\rho_F.
$$

したがって、

$$
\boxed{
\rho_F>\frac1\gamma
}
$$

つまり

$$
\boxed{
\rho_F\gamma>1
}
\tag{14}
$$

なら、$d$-subproblemは凸になります。

これによって式 (10) の更新が一意に扱いやすくなります。

逆に

$$
\rho_F\gamma\leq1
$$

だと、ADMMの $d$-update 自体が非凸になり、更新が不連続・多値的になり得ます。

したがって実装するなら、少なくとも

$$
\boxed{\rho_F\gamma>1}
$$

を保つ設計を強く勧めます。

たとえば

$$
\gamma=3
$$

なら、

$$
\rho_F>\frac13.
$$

---

# 14. ADMMの収束保証も変わる

ここはかなり重要です。

Fused Lassoでは目的関数は凸なので、標準的な条件の下でADMMの収束理論をそのまま使えます。

しかしFused MCPは

$$
p_{\lambda_F,\gamma}
$$

が非凸なので、

$$
\boxed{\text{通常の凸ADMMの大域収束保証は失われる}}
$$

ことになります。

これは今回読んでいるMCP論文でも同様で、MCP/SCADでは一般にglobal minimumへの保証はありません。論文のcoordinate descentでも、一般にはlocal minimum / coordinate-wise minimumへの収束です。`nihms332857.pdf`

ADMMの場合も、

$$
\text{stationary point}
$$

や局所解への収束を考える必要があります。

したがって実験では、

- 複数初期値
- warm start
- primal residual
- dual residual
- objective value
- $\gamma$ に対する安定性

を見るべきです。

---

# 15. もう一つの定義：係数自身もMCPにする

ここまで私は

$$
\lambda_1\|\beta\|_1
+
\text{MCP}(D\beta)
$$

を考えました。

つまり

$$
\boxed{\text{Lasso + Fused MCP}}
$$

です。

しかし「Fused MCP」を

$$
\boxed{
\sum_jp_{\lambda_1,\gamma_1}(|\beta_j|)
+
\sum_kp_{\lambda_F,\gamma_F}(|(D\beta)_k|)
}
\tag{15}
$$

と定義することも可能です。

つまり、

- sparsity penalty も MCP
- fusion penalty も MCP

です。

この場合でも $\beta$-updateは同じです。

ただし $z$-updateも

$$
z_j^{k+1}
=
T_{\mathrm{MCP}}
\left(
\beta_j^{k+1}+u_j^k;
\lambda_1,\gamma_1,\rho_1
\right)
$$

になります。

つまり、

$$
\boxed{
\begin{array}{ccc}
&\text{Fused Lasso}&\text{Full Fused MCP}\\
z\text{-update}
&
\text{soft}
&
\text{MCP firm}
\\
d\text{-update}
&
\text{soft}
&
\text{MCP firm}
\end{array}
}
$$

です。

---

# 16. あなたの手法ならどちらを変えるべきか

ここはモデルの目的によります。

もし現在Fused Lassoを使っている主目的が、

> 近接するパラメータを同じクラスタ・同じ区間にまとめたい。ただし本当に大きな差まで縮めたくない

であれば、まず変更すべきなのは

$$
\boxed{
\lambda_F\|D\beta\|_1
\rightarrow
\sum_jp_{\lambda_F,\gamma}(|D_j\beta|)
}
$$

です。

つまり **fusion部分だけMCP化**するのが自然です。

この変更だと、

- 小さな差 → 0へ
- 中程度の差 → 縮小するがLassoより弱い
- 大きな差 → 縮小しない

となります。

一方、$\beta_j$ 自身のvariable selectionについてもLasso biasを問題にしているのであれば、

$$
\lambda_1|\beta_j|
$$

もMCPに置き換える価値があります。

---

# 17. Fused Lasso → Fused MCP の本質

最も簡潔にまとめると、

Fused Lasso：

$$
\beta_i-\beta_j
\quad
\overset{\text{soft threshold}}{\longrightarrow}
\quad
0
\text{ or 縮小された差}
$$

Fused MCP：

$$
\beta_i-\beta_j
\quad
\overset{\text{firm threshold}}{\longrightarrow}
\quad
\begin{cases}
0 & \text{小さい差}\\
\text{弱く縮小} & \text{中程度}\\
\text{そのまま} & \text{大きい差}
\end{cases}
$$

です。

つまり

$$
\boxed{
\text{Fused Lassoは「すべてのジャンプを縮める」}
}
$$

のに対し、

$$
\boxed{
\text{Fused MCPは「小さいジャンプだけ消し、大きなジャンプは尊重する」}
}
$$

というモデルになります。

あなたの現在のADMM Fused Lassoの具体的な目的関数と更新式を貼ってもらえれば、次はそれを基準にして、**現在の各更新式を1行ずつ Fused MCP に置換して、実装可能なアルゴリズムまで完全に導出**できます。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)