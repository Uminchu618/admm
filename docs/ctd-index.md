$C^{td}$は、C-indexに相当する時間効果変動する生存時間モデルの性能評価指標である。

## Notation

- $T_i$: subject $i$ の観測時刻
- $D_i$: subject $i$ のイベント指標
  - $D_i = 1$: イベント発生
  - $D_i = 0$: 打ち切り
- $X_i(t)$: subject $i$ の時点 $t$ における共変量
- $S(t \mid X_i(t))$: subject $i$ の時点 $t$ における予測生存確率
- $\phi_{\mathrm{comp}}$: comparable pair の確率
- $\phi^{td}_{\mathrm{conc}}$: time-dependent concordant pair の確率

## Final formulals

$$
C^{td}
=
\frac{\phi^{td}_{\mathrm{conc}}}{\phi_{\mathrm{comp}}}
=
\Pr\!\left(
S(T_i \mid X_i(t))
<
S(T_i \mid X_j(t))
\;\middle|\;
T_i < T_j,\; D_i = 1
\right)
$$

where

$$
\phi_{\mathrm{comp}}
=
\Pr(T_i < T_j,\; D_i = 1)
$$

and

$$
\phi^{td}_{\mathrm{conc}}
=
\Pr\!\left(
S(T_i \mid X_i(t))
<
S(T_i \mid X_j(t)),
\;
T_i < T_j,
\;
D_i = 1
\right)
$$