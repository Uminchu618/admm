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

<!-- _header: 再掲: 対数尤度関数 -->
$$
\begin{aligned}
\log L(\{\beta_{jk}\},\{\gamma_m\})
&= \sum_{i=1}^n \left[
  \delta_i\left\{
    \eta_{i,k(i)} 
    + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{i,k(i)})T_i\big)
  \right\} \right. \\
&\qquad\left.
  - \sum_{k=1}^{k(i)} \int_{a_{ik}}^{b_{ik}}
      \exp\left\{
        \eta_{ik} + \sum_{m=1}^M\gamma_m S_m\big(\exp(\eta_{ik})s\big)
      \right\}
  \,ds
\right]. 
\end{aligned} \tag{1}
$$

$$
\ell_{\text{lasso}}(\{\beta_{jk}\},\{\gamma_m\})
= \log L(\{\beta_{jk}\},\{\gamma_m\})
- \lambda_{\text{fuse}} \sum_{j=1}^p \left\| D\boldsymbol{\beta}_j \right\|_1 \tag{2}
$$

---

<!-- _header: Notation-->

<div style="font-size:0.70em">

**対数尤度関数**

- 観測・添字：$i=1,\dots,n$ 個体、$n$ 観測数、$\delta_i$ イベント指示（1=発生, 0=打切り）、$T_i$ 観測生存時間
- 時間分割：$t_0<t_1<\cdots<t_K$ 区間端点、$k=1,\dots,K$ 区間添字、$k(i)$ は $t_{k(i)-1}\le T_i<t_{k(i)}$ を満たす区間番号
- 共変量・回帰係数：$j=1,\dots,p$ 共変量添字（切片0は含まない。ベースラインが切片に当たるから）、$X_{ij}$ 共変量、$\beta_{jk}$ は区間$k$における係数
- 線形予測子：$\eta_{ik}=\sum_{j=0}^p \beta_{jk}X_{ij}$（個体$i$・区間$k$）
- ベースライン（B-spline）：$m=1,\dots,M$ 基底添字、$S_m(\cdot)$ B-spline基底、$\gamma_m$ スプライン係数（$\gamma\in\mathbb{R}^M$）
- 目的関数：$\log L$ 対数尤度、$\ell_{\text{lasso}}=\log L-\lambda_{\text{fuse}}\sum_{j=1}^p\|D\boldsymbol{\beta}_j\|_1$（Fused Lasso）
- 係数ベクトル：$\boldsymbol{\beta}_j=(\beta_{j1},\dots,\beta_{jK})^\top\in\mathbb{R}^{K}$
- 一次差分行列：$D\in\mathbb{R}^{(K-1)\times K}$ $D\boldsymbol{\beta}_j=(\beta_{j2}-\beta_{j1},\dots,\beta_{jK}-\beta_{j,K-1})^\top\in\mathbb{R}^{K-1}$）
- 係数全体：$\boldsymbol{\beta}=\{\beta_{jk}\}_{j=1,\dots,p;\,k=1,\dots,K}$（$\boldsymbol{\beta}$はADMMで1ブロックとして更新するため）

**ADMM**
- $r$: ADMM反復、$\rho>0$ 罰則パラメータ
- $z_j\in\mathbb{R}^{K-1}$: 制約 $D\boldsymbol{\beta}_j=z_j$ の補助変数（$\ell_1$ prox の対象）
- $u_j\in\mathbb{R}^{K-1}$: Scaled dual variable

**区分求積（積分近似）**
- $a_{ik}=t_{k-1}$、$b_{ik}=\min(T_i,t_k)$
- $Q$ 求積点数、$w_{ik\ell}$ 重み、$v_{ik\ell}$ 求積点
- $\log\tilde{L}$: 積分近似した対数尤度$\log L$
</div>

---

<!-- _header: ADMM 更新式 -->

<div style="font-size:0.85em">

*   **$\boldsymbol\beta,\gamma$ 更新**
    $$
    (\boldsymbol{\beta}^{r+1},\gamma^{r+1})
    =
    \arg\min_{\boldsymbol{\beta},\,\gamma}\Big[
      -\log L(\boldsymbol{\beta},\gamma)
      +\frac{\rho}{2}\sum_{j=1}^p\|D\boldsymbol{\beta}_j - z_j^{r}+u_j^{r}\|_2^2
    \Big]  \tag{3}
    $$
    ※ $\log L$ は $\eta_{ik}=\sum_{j=0}^p\beta_{jk}X_{ij}$ を通じて全ての $\boldsymbol{\beta}$ に依存するため、ADMMの更新は $\boldsymbol{\beta}$ を $j$全体で行う

*   **$z$ 更新（prox）**
    $$
    z_j^{r+1} =\arg\min_{z_j\in \mathbb{R}^{K-1} }\Big[
      \lambda_{\text{fuse}}\|z_j\|_1 + \frac{\rho}{2}\|D\boldsymbol{\beta}_j^{r+1}-z_j+u_j^r\|_2^2
    \Big]  \tag{4}
    $$

*   **双対変数の更新**
    $$
    u_j^{r+1}=u_j^r + D\boldsymbol{\beta}_j^{r+1}-z_j^{r+1} \tag{5}
    $$

式(4),(5)は Generalized Lasso と同様に取り扱える。問題は式(3)。

式(3)は（後述の区分求積で）$\log L$ を $\log\tilde{L}$ に置き換え、**$(\boldsymbol{\beta},\gamma)$ を同時に** Newton法で更新する。  
ADMMのステップ内でNewton法を行うため、ADMMの1STEP内で厳密最適化する必要はなく、
Newton更新は 1回（or 数回）で打ち切ってよい（inexactに解く）。

</div>

> 準ニュートン法やガウスニュートン法（Levenberg–Marquardt法）なども検討したが、次元が大きくない場合はニュートン法で十分  

---
<!-- _header: 対数尤度関数の積分を区分求積法で -->
積分は区分求積法により線形結合として近似する($w$重み、$v$求積点)

$$
h_{ik}(s)
:=
\eta_{ik}
+
\sum_{m=1}^{M}
\gamma_m\,
S_m\!\left(\exp(\eta_{ik})\,s\right)
$$
$$
\int_{a_{ik}}^{b_{ik}} \exp\!\bigl(h_{ik}(s)\bigr)\,ds
\;\approx\;
\sum_{\ell=1}^{Q} w_{ik\ell}\,
\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)
$$

$$
\begin{aligned}
\log \tilde{L}(\{\beta_{jk}\},\{\gamma_m\})
&= \sum_{i=1}^n \left[
  \delta_i\,h_{i,k(i)}(T_i)
  - \sum_{k=1}^{k(i)} \sum_{\ell=1}^{Q} w_{ik\ell} \exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)
\right]
\end{aligned} \tag{6}
$$

---

<!-- _header: B-Spline基底関数の微分 -->
B-Spline基底関数の微分は、低次のB-Spline基底関数を用いて表すことができる
$$
 S_m'(x) = \frac{p}{t_{m+p}-t_m}\,S_{m,p-1}(x) - \frac{p}{t_{m+p+1}-t_{m+1}}\,S_{m+1,p-1}(x) \tag{7}
$$

同様に二回微分もさらに低次（$p-2$ 次）のB-Spline基底関数で表せる（$p\ge2$）
$$
\begin{aligned}
S_m''(x)
=& \frac{p}{t_{m+p}-t_m}\,S_{m,p-1}'(x) - \frac{p}{t_{m+p+1}-t_{m+1}}\,S_{m+1,p-1}'(x)\\
=& \frac{p(p-1)}{(t_{m+p}-t_m)(t_{m+p-1}-t_m)}\,S_{m,p-2}(x) \\
&- \frac{p(p-1)}{t_{m+p}-t_{m+1}}\left(\frac{1}{t_{m+p}-t_m}+\frac{1}{t_{m+p+1}-t_{m+1}}\right)\,S_{m+1,p-2}(x)\\
&+ \frac{p(p-1)}{(t_{m+p+1}-t_{m+1})(t_{m+p+1}-t_{m+2})}\,S_{m+2,p-2}(x).
\end{aligned} \tag{8}
$$

---

<!-- _header: Newton法の準備 $\\beta$ に関する勾配 （$\\partial/\\partial\\beta_{jk}$） -->
記法を簡単にするため$x_{ik}(s):=\exp(\eta_{ik})\,s$とおく。$\frac{\partial x_{ik}(s)}{\partial \eta_{ik}(s)}=\exp(\eta_{ik})\,s=x_{ik}(s)$
$$
h_{ik}(s)=\eta_{ik}+\sum_{m=1}^M\gamma_m\,S_m\!\left(x_{ik}(s)\right)
$$

$\beta_{jk}$は$\eta_{ik}$にしか含まれていないため、まず $h_{ik}(s)$ を $\eta_{ik}$ で微分する：
$$
\frac{\partial h_{ik}(s)}{\partial \eta_{ik}}
=1+\frac{\partial x_{ik}(s)}{\partial \eta_{ik}(s)}\sum_{m=1}^M\gamma_m\,S_m'\!\left(x_{ik}(s)\right)
=1+x_{ik}(s)\sum_{m=1}^{M}\gamma_m\,S_m'\!\left(x_{ik}(s)\right) \tag{9}
$$

また $\eta_{ik}=\sum_{j=0}^p\beta_{jk}X_{ij}$ より $\frac{\partial\eta_{ik}}{\partial \beta_{jk}}=X_{ij}$
$$
\frac{\partial h_{ik}(s)}{\partial \beta_{jk}}=X_{ij}\left(1+x_{ik}(s)\sum_{m=1}^{M}\gamma_m\,S_m'\!\left(x_{ik}(s)\right)\right) \tag{10}
$$

---

<!-- _header: Newton法の準備 $\\beta$ に関するヘッセ行列-->

2階微分を取るため、式(9)をもう一度$\eta$で微分する
$$
\begin{align*}
\frac{\partial^2 h_{ik}(s)}{\partial \eta_{ik}^2}& =  \left(  x_{ik}(s) \right)' \sum_{m=1}^{M}\gamma_m\,S_m'\!\left(x_{ik}(s)\right) + x_{ik}(s) \left( \sum_{m=1}^{M}\gamma_m\,S_m'\!\left(x_{ik}(s)\right)\right)' \\
&=x_{ik}(s)\sum_{m=1}^M\gamma_m\,S_m'\!\left(x_{ik}(s)\right)+x_{ik}(s)^2\sum_{m=1}^M\gamma_m\,S_m''\!\left(x_{ik}(s)\right)
\end{align*}
$$
また $\eta_{ik}=\sum_{j=0}^p\beta_{jk}X_{ij}$ より $\frac{\partial\eta_{ik}}{\partial \beta_{jk}}=X_{ij}$
したがって $j,j'=0,\dots,p$ 
$$
\frac{\partial^2 h_{ik}(s)}{\partial \beta_{jk}\partial \beta_{j'k}}
=
X_{ij}X_{ij'}\,\left(
x_{ik}(s)\sum_{m=1}^M\gamma_m\,S_m'\!\left(x_{ik}(s)\right)
+x_{ik}(s)^2\sum_{m=1}^M\gamma_m\,S_m''\!\left(x_{ik}(s)\right)
\right) \tag{11}
$$

※ $h_{ik}$ は区間 $k$ の係数にしか依存しないため、$k\neq k'$ では$\partial^2 h/\partial\beta_{\cdot k}\partial\beta_{\cdot k'}=0$

---

<!-- _header: $\\log \\tilde{L}$ の $\\beta$ 勾配 とヘッセ行列 -->

区分求積で近似した対数尤度（式(6)）は、式(10)の微分を用いて
$$
\log \tilde{L}
=
\sum_{i=1}^n\Bigg[
\delta_i\,h_{i,k(i)}(T_i)
-\sum_{k=1}^{k(i)}\sum_{\ell=1}^Q w_{ik\ell}\,\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)
\Bigg]
$$
$$
\frac{\partial \log \tilde{L}}{\partial \beta_{jk}}
=
\sum_{i=1}^n\Bigg[
\delta_i\,\frac{\partial h_{i,k(i)}(T_i)}{\partial \beta_{j,k(i)}}
-
\sum_{k=1}^{k(i)}\sum_{\ell=1}^Q
w_{ik\ell}\,\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)\,\frac{\partial h_{ik}(v_{ik\ell})}{\partial \beta_{jk}}
\Bigg] \tag{12}
$$

$$
\begin{align}
\frac{\partial^2 \log \tilde{L}}{\partial \beta_{jk}\partial \beta_{j'k}}
& =
\sum_{i=1}^n\Bigg[
\delta_i\,\frac{\partial^2 h_{i,k(i)}(T_i)}{\partial \beta_{j,k(i)}\partial \beta_{j'k(i)}}
-
\sum_{k=1}^{k(i)}\sum_{\ell=1}^Q
w_{ik\ell}\,\left(
\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)\,\frac{\partial^2 h_{ik}(v_{ik\ell})}{\partial \beta_{jk}\partial \beta_{j'k}}
+ \frac{\partial\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)}{\partial \beta_{j'k}} \,\frac{\partial h_{ik}(v_{ik\ell})}{\partial \beta_{jk}}
\right)
\Bigg] \\
&= 
\sum_{i=1}^n\Bigg[
\delta_i\,\frac{\partial^2 h_{i,k(i)}(T_i)}{\partial \beta_{j,k(i)}\partial \beta_{j'k(i)}}
-
\sum_{k=1}^{k(i)}\sum_{\ell=1}^Q
w_{ik\ell}\,\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)\,\left(
\frac{\partial^2 h_{ik}(v_{ik\ell})}{\partial \beta_{jk}\partial \beta_{j'k}}
+ \frac{\partial h_{ik}(v_{ik\ell})}{\partial \beta_{j'k}} \,\frac{\partial h_{ik}(v_{ik\ell})}{\partial \beta_{jk}}
\right)
\Bigg]
\end{align}
$$
$$\tag{13}$$

---

<!-- _header: $\\log \\tilde{L}$ の $\\gamma$ 勾配 とヘッセ行列 -->
 $\gamma$  は  $h$ に線形に入っているので、シンプル。
$$
\frac{\partial h_{ik}(s)}{\partial \gamma_m}=S_m\!\left(x_{ik}(s)\right) \qquad
\frac{\partial^2 h_{ik}(s)}{\partial \gamma_m\,\partial \gamma_{m'}}=0 \qquad(\forall m,m')
$$

$$
\frac{\partial \log \tilde{L}}{\partial \gamma_m} =
\sum_{i=1}^n\left[
\delta_i\,S_m\!\left(x_{i,k(i)}(T_i)\right)
- \sum_{k=1}^{k(i)}\sum_{\ell=1}^Q w_{ik\ell}\,\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)\,S_m\!\left(x_{ik}(v_{ik\ell})\right)
\right]  \tag{14}
$$

ヘッセ行列を求めるにあたって、 $\exp(h)$  の 2 階微分は
$$
\frac{\partial^2}{\partial\gamma_m\partial\gamma_{m'}} \exp(h) =
\exp(h)\left(
\frac{\partial h}{\partial\gamma_m}\frac{\partial h}{\partial\gamma_{m'}}
+ \frac{\partial^2 h}{\partial\gamma_m\partial\gamma_{m'}}
\right)
$$
第二項は0、かつ、イベント項（$\delta$の項）も0になるので
$$
 \frac{\partial^2 \log \tilde{L}}{\partial \gamma_m\,\partial \gamma_{m'}} =
 -\sum_{i=1}^n \sum_{k=1}^{k(i)}\sum_{\ell=1}^Q
 w_{ik\ell}\,\exp\!\bigl(h_{ik}(v_{ik\ell})\bigr)\,
 S_m\!\left(x_{ik}(v_{ik\ell})\right)\,
 S_{m'}\!\left(x_{ik}(v_{ik\ell})\right) \tag{15}
$$

---
<!-- _header: ADMM更新 式(3) の第二項の勾配、ヘッセ行列 -->

<div style="font-size:0.9em">

式(3)で Newton 法が最小化する目的関数（ADMMの第二項）
$$
F(\boldsymbol{\beta},\gamma)
:=
-\log\tilde{L}(\boldsymbol{\beta},\gamma)
+\underbrace{\frac{\rho}{2}\sum_{j=1}^p\left\|D\boldsymbol{\beta}_j-z_j^r+u_j^r\right\|_2^2}_{=:~g(\boldsymbol{\beta})}
$$


各 $j$ について残差$r_j$とおく
$$
r_j^r := D\boldsymbol{\beta}_j-z_j^r+u_j^r\in\mathbb{R}^{K-1}
$$

$$
\nabla_{\boldsymbol{\beta}_j} g(\boldsymbol{\beta})
= \rho D^\top r_j^r
= \rho D^\top\left(D\boldsymbol{\beta}_j-z_j^r+u_j^r\right)
$$

$$
\nabla^2_{\boldsymbol{\beta}_j\boldsymbol{\beta}_j} g(\boldsymbol{\beta})
= \rho D^\top D
\qquad
\nabla^2_{\boldsymbol{\beta}_j\boldsymbol{\beta}_{j'}} g(\boldsymbol{\beta})=0\ (j\neq j')
$$



また $g$ は $\gamma$ を含まないので
$$
\nabla_{\gamma} g(\boldsymbol{\beta})=0,\qquad
\nabla^2_{\gamma\gamma} g(\boldsymbol{\beta})=0,\qquad
\nabla^2_{\boldsymbol{\beta}\gamma} g(\boldsymbol{\beta})=0.
$$


</div>
