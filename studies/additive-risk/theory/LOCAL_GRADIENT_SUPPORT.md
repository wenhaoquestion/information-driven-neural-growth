# n=2048 的全域合法损失 oracle 与统计局部界

此文件在总体首轮 `FIRST_PASS_FREEZE.json` 冻结后新增；不更改任何冻结总体证书。唯一统计候选为原内点 `n=2048`。参考均值与真值只用于这份独立功效认证，`Interior2048Loss` 的运行接口只有报告 `q` 和固定公开损失参数。

## 全域损失定义与接口

沿用首轮符号，`e=16 log(2048)/2048`、`A_star=2J`。取

\[
 h(q)=-F_n(4q-3/2)/(2J),\quad
 h'(q)=-4F_n'(4q-3/2)/(2J).
\]

`h` 与规范损失的 Bayes entropy 相差一个常数，所有 fitted-risk contrast、regret、报告误差均不变。规范曲率总变差恰为 2，所以 oracle 的 `variation=2` 精确合法。接口文件是 `local_loss_oracle.py`，类 `Interior2048Loss`，返回统计实现 `honest_delta.Interval` 类型，端点均为 `Fraction`。`entropy(q)` 和 `slope(q)` 对所有 `q∈[0,1]` 都合法；在针附近只回退到较宽的解析外包，不访问真实后验。

## 更强的有限尾界

令 `e_lo<=e<=e_hi` 为 30 位小数格点向外包的有理数，固定

\[
 \eta=1/200,\qquad r=e_{lo}-1/40.
\]

这是固定公开损失的数值参数。实际参数满足 `0<eta<r<e_lo<e_hi<1/16`。对 `0<=s<e` 定义

\[
 v(s)=\operatorname{arcosh}(1+(e^2-s^2)/16).
\]

由 `arcosh(1+u)=2 asinh(sqrt(u/2))` 及

`x/sqrt(1+x^2) <= asinh(x) <= x`，有

\[
 \sqrt{2u/(1+u/2)}\le\operatorname{arcosh}(1+u)\le\sqrt{2u}.
\]

令

\[
 v_0\le\sqrt{2u_0/(1+u_0/2)},\quad
 u_0=(e_{lo}^2-\eta^2)/16,
\]

\[
 v_r\ge\sqrt{(e_{hi}^2-r^2)/8},\quad k=2n(v_0-v_r)>0.
\]

脚本通过整数平方根给 `v0` 向下、`vr` 向上有理外包。核心区间 `[-eta,eta]` 长度 `2eta`，`cosh^2(nv)>=exp(2nv)/4`，因此

\[
 Z\ge(\eta/2)e^{2nv_0}.
\]

正侧中间尾 `[r,e]` 上 `p(s)<=exp(2nvr)`，而 `[e,4]` 上 `p<=1`，所以单侧尾的两个部分满足

\[
 C:=\int_r^e\kappa\le\frac{2(e_{hi}-r)}{\eta}e^{-k},\qquad
 O:=\int_e^4\kappa\le\frac8\eta e^{-2nv_0}.
\]

没有用粗糙的整段长度 8 乘内部指数高度：振荡尾 `[e,4]` 单独用 `p<=1`。因此无需进一步切线积分已足够。

每个 `exp(-x)` 上界通过正项 Taylor 部分和

\[
 e^{-x}\le\left(\sum_{j=0}^{200}x^j/j!\right)^{-1}
\]

得到；这是有方向的严格界，不需要浮点指数或未控余项。整数平方根、Taylor 和与所有四则运算都是精确有理数，随后仅做外向取整以控制分母长度。

对 `|t|>=r`，反射对称性给出

\[
 |H'(t)-\mathbf1_{t>0}|\le T:=C+O,
\]

\[
 0\le H(t)-t_+\le K:=(e_{hi}-r)C+4O.
\]

认证结果为 `T<=1.516964976805222e-6`、`K<=3.792412442013053e-8`（此处展示向上粗舍入）。在全部实数上，另有通用合法界 `0<=H(t)-t_+<=e_hi/2+delta`；它由误差最大在零点、正半边针质量为 `1/2` 和 `\int_e^4s kappa<=8/Z<=delta` 得到。CDF 则始终属于 `[0,1]`。这些给出 oracle 的全域保守回退。

## oracle 的区间运算

对于给定精确有理数 `q`，令 `z=4q-3/2`。两个 hinge 自变量随真实 `e` 落在

`[-z-e_hi,-z-e_lo]` 与 `[z-1-e_hi,z-1-e_lo]`。

若某整个自变量区间位于 `(-infinity,-r]` 或 `[r,infinity)`，对应项使用局部 `K,T`；否则使用全域界。二次项按 `e^2` 的区间直接外包。最终除以严格正的 `A_star∈[A_lo,A_hi]`，并取负号得到 `h`。斜率由

\[
 F_n'(z)=-H'(-z-e)+H'(z-1-e)+8e^2(z-1/2)
\]

直接外包，再乘 `-4/A_star`。因此 oracle 不是假定“实际值等于理想 hinge”，也不把高精度浮点 CDF 当证书。

## 对正式功效盒中所有 cell mean 的统一宽度

统计方法的正式矩形读自 `statistics/INTERIOR_2048_POWER_ENVELOPE.json` 的有理端点；每坐标均落在对应 `(1/8,3/8,5/8,7/8)` 的 `±1/160` 内。所有非空 cell mean 因而在原理想有限均值附近 `±1/160`，辅助 `z` 变动不超过 `1/40`。原均值到两针位置的最小距离为 `e`，故新均值距离至少 `e-1/40>=r`。

脚本对全部 15 个 cell 的**整个均值区间**分别验证这一条件，不仅检查端点。利用 `h'` 单调递减，每个 cell 的斜率外包为 `[slope(mean_hi).lo, slope(mean_lo).hi]`；这些完整区间已记录在 `LOCAL_ORACLE_CERTIFICATE.json`。

令 `de=e_hi-e_lo`、`de2=e_hi^2-e_lo^2`。在任一满足两个自变量皆远离针的报告点上，统一 oracle 宽度界为

\[
 \eta_h=\frac{2de+16de2+2K}{A_{lo}}
 +(2+16e_{hi}^2)(A_{lo}^{-1}-A_{hi}^{-1}),
\]

\[
 \eta_s=\frac{4(2T+16de2)}{A_{lo}}
 +4(1+16e_{hi}^2)(A_{lo}^{-1}-A_{hi}^{-1}).
\]

这里 `|z-1/2|<=2`，理想 hinge 值对 `e` 的区间宽度总计不超过 `2de`，二次项宽度不超过 `16de2`；`|F'|<=1+16e_hi^2`。分母不确定性也计入，未当作零。代码返回的精确有理界略向上取整，为

- `loss.entropy_width = 8971687927159447392533 / 500000000000000000000000000000`，约 `1.7943375854318894e-8`；
- `loss.slope_width = 358867517053286825216247 / 125000000000000000000000000000`，约 `2.8709401364262946e-6`。

这些是 oracle **本身的数值外包宽度**，不是 posterior confidence radius，也不把固定功效盒的 Delta 下界当作样本程序功效。统计程序仍须按它的独立移动中心误差推导，使用这两个宽度进行功效校正。

复算：`PYTHONDONTWRITEBYTECODE=1 python3 studies/additive-risk/theory/local_loss_oracle.py`。日志保存在 `local_loss_oracle.stdout.log`，完整常数及 cell 范围保存在 `LOCAL_ORACLE_CERTIFICATE.json`。此补充没有抽样，也没有计算或宣称最终显著性/功效结论。
