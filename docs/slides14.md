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
<!-- _header: 時間拡張AFTモデル -->



観測開始時点を$t_0 < t_1 < t_2 < \dots<t_K$

区間$[t_{k-1}, t_{k})$ $1 < k \le K$における


$p$ 次元の時間変動有りパラメータ $\boldsymbol{\beta}_k^{\top} = (\beta_{k0}, \beta_{k1}, \beta_{k2} \dots, \beta_{kp})$


$T_i$: $i$ 番目の個体の観測された生存時間

$\delta_i$: 打ち切り指示変数

観測数 : $n$

指示関数：$I(\cdot)$





---
<!-- _header: 方針 -->


- クラシックなAFTで時間依存する回帰係数を持つことは難しい
**ハザードからモデリングするほうが適切**
直感的な解釈：ハザードは瞬間的な状態から算出されるものなので、時間方向の積分がでてこない？

- 共変量と回帰係数、両方とも時間依存とするのは難しい？
目的に立ち戻ると、回帰係数の時間変動を捉えることが目的であり
共変量の方はおまけ

- Pang et al. (2021) のスプラインモデルをFused Lasso化する方針
 
> Pang, Menglan, Robert W Platt, Tibor Schuster, and Michal Abrahamowicz. “Flexible Extension of the Accelerated Failure Time Model to Account for Nonlinear and Time-Dependent Effects of Covariates on the Hazard.” Statistical Methods in Medical Research 30, no. 11 (2021): 2526–42. https://doi.org/10.1177/09622802211041759.


---
<!-- _header: モデル式(修正) -->

**修正方針**
変化点スパースな 時間依存acceleration factorを推定することで、解釈しやすいモデルを提案
滑らかさより変化点がはっきりでるほうが実務的には使いやすい
- モデルの骨格は Pang et al. (2021) の hazard-based AFT をそのまま使う
- $β_j(t)$ のスプライン展開部分だけ、区分一定＋Fused Lasso に差し替える
- ベースラインハザードは、スプライン近似
- 共変量のdose-response functionは線形として考える

> $g_j(X_j)$の非線形的モデル化も考えられるが、あとで
> Coxでtime-varying β(t)＋fused lasso の論文はあるが、hazard-based AFT で同じことをやっている論文は無さそう

---
<!-- _header: モデル式(修正) -->
・共変量のdose-response function　 log acceleration factor
$$
g_j(X_j) = X_j \tag{2}
$$
$$
\beta_j(t) = \sum_{k=1}^{K} \beta_{jk} I(t_{k-1}\le t \lt t_{k}) \tag{3}
$$
・ ベースラインハザード
$$
 \quad \lambda_0(z) =\exp\left(\sum_{m=1}^M\gamma_m S_m(z)\right) \qquad S_m(z) :\text{B-spline basis functions} \tag{4}
$$
・hazard-based AFT モデル式 個体  $i$ ・共変量ベクトル  $X_i$  に対し
$$
\eta_i(t)  =\sum_{k=1}^K\eta_{ik}I(t_{k-1}\le t \lt t_{k}), \quad \eta_{ik}=\sum_{j=0}^p\beta_{jk}X_{ij}
$$

$$
\lambda(t\mid X_i) =\exp(\eta_i(t))\; \lambda_0\big(\exp(\eta_i(t))t\big) \quad \tag{5}

$$

---
<!-- _header:   対数尤度 -->
$t_{k'-1}\le t \lt t_{k'}$を満たす区間内の場合、$\eta_i(t)=\eta_{ik'}$であるため
$$
\begin{align*}
\lambda(t\mid X_i,\; t_{k'-1}\le t \lt t_{k'}) &=\exp\left(\eta_{ik'}\right)\; \lambda_0\big(\exp(\eta_{ik'})t\big) \\
&=\exp\left(\eta_{ik'}\right)\; \exp\left(\sum_{m=1}^M\gamma_m S_m(\exp(\eta_{ik'})t)\right) \\ 
 &=\exp\Big\{ \eta_{ik'} +\sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{ik'})t\big) \Big\} \qquad \tag{6} 

\end{align*}
$$



> 前回の資料では、曖昧だった条件をつけた

---
<!-- _header: モデル式(修正)-->

観測  $(T_i,\delta_i,X_i), i=1,\dots,n$  に対し、対数尤度はPang et al. (2021)と同じく

$$
\log L =\sum_{i=1}^n \left[ \delta_i\log\lambda(T_i\mid X_i) -\int_0^{T_i}\lambda(u\mid X_i)\,du \right]  
$$
第一項がイベント発生の項、第二項が累積ハザードの項。其々で計算する。
イベントの項に関して 式6の $\lambda(t\mid X_i,\; t_{k'-1}\le t \lt t_{k'})$  を代入する



$T_i$  が属する区間番号を$k(i)$とする　つまり、$t_{k(i)-1} \le T_i \lt t_{k(i)}$


$$
\begin{aligned} \log\lambda(T_i\mid X_i) &=\eta_{i,k(i)} +\sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{i,k(i)})T_i\big). \end{aligned}
$$

よって、イベントの項は

$$
\delta_i\log\lambda(T_i\mid X_i) = \delta_i\left[ \eta_{i,k(i)}+ \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{i,k(i)})T_i\big) \right] \tag{7}
$$

---
<!-- _header:   対数尤度 -->

累積ハザードの項について、区間を分けて積分する。
観測開始時点を$t_0<t_1<\cdots<t_K$とすると、
$$
\begin{aligned} 
\int_0^{T_i}\lambda(u\mid X_i)\,du 
&= \int_{t_0}^{t_1}\lambda(u\mid X_i)\,du
+ \int_{t_1}^{t_2}\lambda(u\mid X_i)\,du
+ \cdots
+ \int_{t_{k(i)-1}}^{T_i}\lambda(u\mid X_i)\,du \\
&= \sum_{k=1}^{k(i)} \int_{t_{k-1}}^{\min(T_i,t_k)} \lambda(u\mid X_i,\; t_{k-1}\le u \lt t_{k})\,du \\
&=\sum_{k=1}^{k(i)}  \int_{t_{k-1}}^{\min(T_i,t_k)}  \exp\left\{\eta_{ik} + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{ik})u\big) \right\}du 
\end{aligned} \tag{8}
$$

> 積分区間を分割したらシンプルな形にできる
---
<!-- _header:   対数尤度 -->
式(7)、(8)より　hazard-based AFT モデルの対数尤度は

$$
\begin{aligned}
\log L(\{\beta_{jk}\},\{\gamma_m\})
&= \sum_{i=1}^n \left[
  \delta_i\left\{
    \eta_{i,k(i)} 
    + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{i,k(i)})T_i\big)
  \right\} \right. \\
&\qquad\left.
  - \sum_{k=1}^{k(i)} \int_{t_{k-1}}^{\min(T_i,t_k)}
      \exp\left\{
        \eta_{ik} + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{ik})u\big)
      \right\}
  \,du
\right]. 
\end{aligned} \tag{9}
$$



Fused Lasso 正則化を加えると、対数尤度は

$$
\ell_{\text{lasso}}(\{\beta_{jk}\},\{\gamma_m\})
= \log L(\{\beta_{jk}\},\{\gamma_m\})
- \lambda_{\text{fuse}} \sum_{j=1}^p \left\| D\boldsymbol{\beta}_j \right\|_1 \tag{10}
$$

---
<!-- _header:   対数尤度 -->

ここで $\boldsymbol{\beta}_j = (\beta_{j1},\dots,\beta_{jK})^{\top}$、$D$ は一次差分行列
$$
D = \begin{pmatrix}
-1 & 1 & 0 & \cdots & 0 \\
0 & -1 & 1 & \cdots & 0 \\
\vdots & & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & -1 & 1
\end{pmatrix}
$$
であり、$D\boldsymbol{\beta}_j = (\beta_{j2}-\beta_{j1},\dots,\beta_{jK}-\beta_{j,K-1})^{\top}$ となる。

> ADMMを使うため、Generaized Lassoの形にした

---
<!-- _header: B-スプライン -->
複数の多項式を滑らかに接続して、１つの基底関数（B-Spline基底関数）を構成したものである
一般に$r$次のB-スプライン基底関数を$B_j(x;r)$とする。
$m$個の3次B-スプライン基底関数$\{ B_1(x;3),B_2(x;3),\dots,B_m(x;3) \}$を構成するために必要な結節点$t_i$を次のようにとる
$$
t_1<t_2<t_3<t_4=x_1< \dots <t_{m+1}=x_n< \cdots <t_{m+4}
$$
このように結節点を取ることによって$n$個のデータは$m-3$個の区間にによって分割される
下記のde Boor(2001)の逐次式を用いて、一般に$r$次のスプライン関数を求めることができる
$$
B_j(x;r)= \frac{x-t_j}{t_{j+r}-t_j}B_j(x;r-1)+\frac{t_{j+r+1}-x}{t_{j+r+1}-t_{j+1}}B_{j+1}(x;r-1)
$$
特に $r=0$ のときの0次B-スプライン基底関数は
$$
B_j(x;0)=
\begin{cases}
1 & (t_j \le x < t_{j+1})\\
0 & \text{それ以外}
\end{cases}
$$
で与えられる。

> 多変量解析入門―線形から非線形へ―(第９刷) 小西 （2021）

---
<!-- _header: B-スプライン -->

3次B-スプライン基底関数は $r=3$ としたとき
$$
B_j(x;3)= \frac{x-t_j}{t_{j+3}-t_j}B_j(x;2)+\frac{t_{j+4}-x}{t_{j+4}-t_{j+1}}B_{j+1}(x;2)
$$
2次B-スプライン基底関数は $r=2$ としたとき
$$
B_j(x;2)= \frac{x-t_j}{t_{j+2}-t_j}B_j(x;1)+\frac{t_{j+3}-x}{t_{j+3}-t_{j+1}}B_{j+1}(x;1)
$$
と書け，区間 $[t_j,t_{j+3})$ のみで非ゼロとなる2次多項式となる。

> ChatGPTに展開やらせたが式が巨大過ぎてしんどい。遅いけど再帰。


---
<!-- _header: ADMM Generalized Lasso (練習) -->

ADMMの更新式
$$
\begin{align}
  \beta^{(k+1)} &= \underset{\beta \in \mathbb{R}^p}{\text{argmin}} \left(f(\beta) + \frac{\rho}{2} \| A\beta -\gamma^{(k)} +\alpha^{(k)}\|_2^2 \right) \quad \alpha\text{: Scaled Dual Variable} \tag{12}\\
  \gamma^{(k+1)} &= \underset{\gamma \in \mathbb{R}^n}{\text{argmin}} \left( g(\gamma) + \frac{\rho}{2} \| A\beta^{(k+1)} -\gamma +  \alpha^{(k)} \|_2^2  \right) \tag{13} \\
  \alpha^{(k+1)} &= \alpha^{(k)} + A\beta^{(k+1)} - \gamma^{(k+1)}\tag{14}

\end{align}

$$
式13をProximity Operator->Prox（近接写像）と呼ぶ
滑らかでない凸関数＋ 現在の推定値との二乗誤差
この凸関数が微分不可能であるため解析的に解けないことが重要ポイント。
幸運なことに、Generaized Lassoのような一部の形式は簡易に解けてしまうため
効率的な更新が可能となる。

>  参考資料：[MIRU2016 チュートリアル 明日から使える凸最適化](https://www.slideshare.net/slideshow/miru2016/64563436)
> Zhu, Yunzhang. “An Augmented ADMM Algorithm With Application to the Generalized Lasso Problem.” Journal of Computational and Graphical Statistics 26, no. 1 (2017): 195–204.


---
<!-- _header: ADMM Generalized Lasso (練習) -->
式(13) は、「ソフト閾値処理」できるproxとなっている
$$
\gamma^{(k+1)} = \underset{\gamma \in \mathbb{R}^n}{\text{argmin}} \left( g(\gamma) + \frac{\rho}{2} \| A\beta^{(k+1)} -\gamma +  \alpha^{(k)} \|_2^2  \right)
$$
$g(\gamma)=\lambda|\gamma|_1$なので、和がベクトルの要素ごとに分離可能であり、各成分の最小化を考えれば良くなる
$$
\begin{align*}
\gamma^{(k+1)} &= \underset{\gamma \in \mathbb{R}^n}{\text{argmin}} \left( \lambda\sum_i|\gamma_i| + \frac{\rho}{2} \sum_i (  (A\beta^{(k+1)})_i -\gamma_i +  \alpha^{(k)}_i )^2  \right) \\
\gamma^{(k+1)}_i &= \underset{\gamma_i \in \mathbb{R}}{\text{argmin}} \left( \lambda|\gamma_i| + \frac{\rho}{2} (  (A\beta^{(k+1)})_i -\gamma_i +  \alpha^{(k)}_i )^2  \right) \\
\end{align*}
$$


> 式１３は、生存関数でも同じ応用が聞きそう
> MIRU2016の中でソフト閾値処理という言葉が出てくる。元ネタはBoyd(2010)の"4.4.3 Soft Thresholding"
> Boyd, Stephen. “Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.” Foundations and Trends® in Machine Learning 3, no. 1 (2010): 1–122. https://doi.org/10.1561/2200000016.

---
<!-- _header: ADMM Generalized Lasso (練習) -->

<div style="font-size:medium">


$$
\text{argmin}_{\gamma_i}\;\; \lambda|\gamma_i|+\frac{\rho}{2}(\gamma_i-v_i)^2 \qquad v_i = (A\beta^{(k+1)})_i  +  \alpha^{(k)}_i
$$

 $|\gamma_i|$  があるので  $\gamma_i>0$ ,  $\gamma_i<0$ ,  $\gamma_i=0$  に場合分け

ケースA:  $\gamma_i>0$ （このとき  $|\gamma_i|=\gamma_i$ ）

$$
\lambda \gamma_i+\frac{\rho}{2}(\gamma_i-v_i)^2 \qquad
\text{微分して 0:}\quad
\lambda+\rho(\gamma_i-v_i)=0 \Rightarrow \gamma_i=v_i-\frac{\lambda}{\rho}.
$$

ただしこのケースは$\gamma_i>0$が条件なので

$$
v_i-\frac{\lambda}{\rho}>0 \;\;\Leftrightarrow\;\; v_i>\frac{\lambda}{\rho}.
$$

ケースB:  $\gamma_i<0$ （このとき  $|\gamma_i|=-\gamma_i$ ）
$$
\gamma_i=v+\frac{\lambda}{\rho}. \qquad \text{ただし} \quad v_i<-\frac{\lambda}{\rho}.
$$

ケースC:  $|v_i|\le \lambda/\rho$  のとき
$\gamma_i=0$となる（劣微分を復習しておく、、、）
</div>

> a simple closed-form solution to this problem by using subdifferential calculus

---
<!-- _header: ADMM Generalized Lasso (練習) -->

まとめるとソフト閾値（soft-thresholding）

$$
\gamma_i^{(k+1)}= \begin{cases} v_i-\lambda/\rho & (v_i>\lambda/\rho)\\ 0 & (|v_i|\le \lambda/\rho)\\ v_i+\lambda/\rho & (v_i<-\lambda/\rho) \end{cases}
$$

これを一行で書くとMIRU2016にあったL1ノルムのProxになる

$$
\gamma_i^{(k+1)}=\operatorname{sign}(v_i)\max\left(|v_i|-\frac{\lambda}{\rho},0\right)
$$

</div>

---
<!-- _header: ADMM ベースラインハザード B-Spline -->
再掲
$$
\begin{aligned}
\log L(\{\beta_{jk}\},\{\gamma_m\})
&= \sum_{i=1}^n \left[
  \delta_i\left\{
    \eta_{i,k(i)} 
    + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{i,k(i)})T_i\big)
  \right\} \right. \\
&\qquad\left.
  - \sum_{k=1}^{k(i)} \int_{t_{k-1}}^{\min(T_i,t_k)}
      \exp\left\{
        \eta_{ik} + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{ik})u\big)
      \right\}
  \,du
\right]. 
\end{aligned} \tag{9}
$$

$$
\ell_{\text{lasso}}(\{\beta_{jk}\},\{\gamma_m\})
= \log L(\{\beta_{jk}\},\{\gamma_m\})
- \lambda_{\text{fuse}} \sum_{j=1}^p \left\| D\boldsymbol{\beta}_j \right\|_1 \tag{10}
$$




---
<!-- _header: ADMM ベースラインハザード B-Spline -->
**ADMM 更新式**
$\gamma$をスプライン係数につかってしまったので$z$にしている。双対変数は$u$

*   **$\beta,\gamma$更新**
    $$
    (\beta^{r+1},\gamma^{r+1}) =\arg\min_{\beta \in \mathbb{R}^{pK},\gamma \in \mathbb{R}^{M}}\Big[ -\log L(\beta,\gamma) +\frac{\rho}{2}\sum_{j=1}^p\|D\beta_j - z_j^{r}+u_j^{r}\|_2^2 \Big]  \tag{15}
    $$
*   **$z$更新（prox）**
    $$
    z_j^{r+1} =\arg\min_{z_j\in \mathbb{R}^{K} }\Big[ \lambda_{\text{fuse}}\|z_j\|_1 + \frac{\rho}{2}\|D\beta_j^{r+1}-z_j+u_j^r\|_2^2 \Big]  \tag{16}
    $$
*   **双対変数の更新**
    $$
    u_j^{r+1}=u_j^r + D\beta_j^{r+1}-z_j^{r+1} \tag{17}
    $$

式16、17はGeneraized Lassoのときと同様に取り扱える。問題は式15

> 書いてて添字jが混乱気味、、、もうプログラミングしたほうがわかりやすい気がしてきた。

---
<!-- _header: ADMM ベースラインハザード B-Spline -->

$$
   \sum_{k=1}^{k(i)} \int_{t_{k-1}}^{\min(T_i,t_k)}
      \exp\left\{
        \eta_{ik} + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{ik})u\big)
      \right\}
\,du$$
積分は区分求積法により線形結合として近似する($w$重み、$v$求積点)
$$
\int_{a_{ik}}^{b_{ik}} \exp\!\bigl(h_{ik}(u)\bigr)\,du
\;\approx\;
\sum_{\ell=1}^{L} w_{ik\ell}\,
\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)
$$

$$
h_{ik}(u)
=
\eta_{ik}
+
\sum_{m=1}^{M}
\gamma_m\,
S_m\!\left(\exp(\eta_{ik})\,u\right)
$$

スプラインなので非線形だが多項式で微分可能となる。ただし、$\gamma$と$\beta$に関して最適化がいる。
$$
    (\beta^{r+1},\gamma^{r+1}) =\arg\min_{\beta \in \mathbb{R}^{pK},\gamma \in \mathbb{R}^{M}}\Big[ -\log L(\beta,\gamma) +\frac{\rho}{2}\sum_{j=1}^p\|D\beta_j - z_j^{r}+u_j^{r}\|_2^2 \Big]  \tag{15}
$$
pangのACEのように$\gamma$だけニュートン法で更新して最適化→$\beta$はADMMで更新
