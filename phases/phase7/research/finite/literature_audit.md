# 有限信息聚类层级价格与端点条件文献核对

核对日期：2026 年 10 月 7 日。用途：判断 Phase 7 有限理论分支的新增命题与一手文献的差别。原项目与 Phase 6 均未修改。

## 结论

1. **任意有限 n 的源质量对数上界不宜作为主要新颖性。** 未找到逐字陈述该加权 KL 层级定理的一手论文，但它可由已知 KL 与 Hellinger 比较、欧氏平方距离常数层级近似直接拼接。下面给出完整的加权比较，明确这是一项数学推论，而非冒充某篇论文的原始定理。
2. **任意 n 的平方根对数上界与四点全局最佳系数，本轮仍未检出同量词结果。** 不应将未检出写成历史首创认证；也不能把四点特定族的渐近常数称为全体四点实例的极值系数。
3. **单个固定生成函数的端点有界反例仍是可继续核查的独立候选。** 本轮阅读的一手来源不包含“存在固定对称严格凸 C∞ 生成函数，端点导数有限，而四状态最优层级的超额风险比无界”这一命题。最近的现有假设是统一曲率比较或 μ-similarity；仅有一阶导数有界明显弱于它们。本文件是原研究时期的文献范围记录；最终证明见同目录 equal_mass_counterexample.tex。

## 直接相关的一手结果

| 一手来源与实际核对位置 | 原结果的对象和量词 | 对本分支的影响 |
|---|---|---|
| Chaudhuri 与 McGregor，*Finding Metric Structure in Information Theoretic Clustering*，COLT 2008，[作者全文](https://cseweb.ucsd.edu/~kamalika/pubs/cm08.pdf)，Theorem 8、Lemmas 12、16 | n 个任意离散分布、等权总 KL 到最近代表点；给出 flat O(log n) 近似。Lemma 12 将任意 t 点单簇最优 KL 代价与 Hellinger 代价夹在常数及 O(log t) 倍之间，Lemma 16 给该比较的对数下界。 | 并非同一兼容层级定理，但 KL/Hellinger 与对数损失的主要比较思想已有先例。其 γ 若出现于已有受限结果，是每个后验坐标下界，不能混同本项目源原子权重。 |
| Sason 与 Verdú，*f-Divergence Inequalities*，IEEE TIT 2016，[作者完整稿](https://arxiv.org/html/1508.00335v7#S4.SS5)，Theorem 9、Remark 13 | 任意公共可测空间的 P≪Q，已知似然比上界时，相对熵与阶数 α 的 Hellinger 散度之比有显式界。取 α=1/2，上界随最大似然比对数增长，不要求似然比具有严格正下界。 | 直接补上加权源质量比较。簇均值支配每个成员，产生 P_i/Q_C≤1/γ；这无需任何目标坐标概率下界。 |
| Plaxton，*Approximation Algorithms for Hierarchical Location Problems*，JCSS 2006，[原始出版记录](https://doi.org/10.1016/j.jcss.2005.09.004)；Lin、Nagarajan、Rajaraman、Williamson，*A General Approach for Incremental Approximation and Hierarchical Clustering*，SICOMP 2010，[原始出版记录](https://doi.org/10.1137/070698257) | 在每个 k 与 flat optimum 比较的常数层级近似及一般 nesting 框架已有来源。 | 原始全文在本次环境仍未取得：Plaxton 返回 403；Lin 返回付费摘要页，作者网站仅列 DOI。不能从摘要声称已逐项核实 Lin 的一般定理及权重细节。 |
| Arutyunova 与 Röglin，*The Price of Hierarchical Clustering*，Algorithmica 2025，[出版全文](https://link.springer.com/article/10.1007/s00453-025-01327-7)，Definitions 1–4、Theorem 1、结论段 | 定义同一树所有 k 的最坏近似比；本文主要研究 metric radius、discrete radius、diameter，并重述 Lin 的 nesting 机制。 | 当前 KL 问题属于既有层级价格范式。本文不是 KL 原始定理；其对 k-means 32 上界的陈述是追溯入口，不能替代原始证明阅读。 |
| Kartowsky 与 Tal，*Greedy-Merge Degrading has Optimal Power-Law*，IEEE TIT 2019，[作者稿](https://arxiv.org/html/1703.04923v1#S3.SS1)，Example 1、Theorems 1–2 | Example 1 的四输出二输入信道等价于四个等权后验点 0、1/3、2/3、1：贪心先合并中间点，阻止最佳两簇相邻配对。主定理控制预算 L 下最坏信道的加性信息损失，阶数为 L 的负幂。 | 四点非嵌套／贪心冲突已有具体先例。它未证明最优层级的源稀有质量乘法下界，亦未识别本项目的平方根对数或四点全局最佳系数。 |

## 为什么任意 n 的对数上界是直接推论

以下是本次核对后的数学拼接，不是引用文献已经陈述过的完整定理。

设源权重 p_i≥γ、总和为 1，后验分布为 P_i；任意簇 C 的质量为 w_C，KL 中心为 Q_C=Σ_{i∈C}(p_i/w_C)P_i。令 v_i=√P_i 逐坐标开方，令

\[
E(C)=\min_{z\in\mathbb R^{|Y|}}\sum_{i\in C}p_i\|v_i-z\|^2.
\]

取自然对数并定义

\[
\kappa(t)=\frac{t\log t-t+1}{(\sqrt t-1)^2},\qquad \kappa(1)=2.
\]

Sason–Verdú Theorem 9 的 α=1/2 特例给出：若 P/Q≤M，则 KL(P‖Q)≤κ(M)‖√P−√Q‖²；κ 单调且 κ(M)=O(1+log M)。因此，因 Q_C≥(p_i/w_C)P_i，源下界推出每个成员对中心的似然比至多 1/γ。零坐标按连续约定处理。

记 a_C=Σ(p_i/w_C)v_i 与 b_C=√Q_C。由 KL≥平方 Hellinger，及平方距离均值的最优性，得到 E(C)≤D_KL(C)。另一方面，逐坐标 Jensen 不等式给 b_C≥a_C，故

\[
\sum_{i\in C}p_i\|v_i-b_C\|^2
=2w_C(1-\langle a_C,b_C\rangle)
\le 2w_C(1-\|a_C\|^2)=2E(C).
\]

所以对所有划分同时有

\[
E(\mathcal P)\le D_{KL}(\mathcal P)
\le 2\kappa(1/\gamma)E(\mathcal P).
\]

若取平方欧氏目标的一个常数 C_E 层级近似，便得到所有 k 同时成立的

\[
D_{KL}(\mathcal H_k)
\le 2C_E\kappa(1/\gamma)D_{KL,k}^{*}
=O(1+\log(1/\gamma))D_{KL,k}^{*}.
\]

权重不会改变平方距离的均值分解与 nesting 证明：所有逐点平方距离不等式乘 p_i 后相加即可。存在性不需要把权重转为大量重复点；重复点技巧若使用不慎，可能允许拆散同一个源状态，反而改变问题。这里不声称核实了最优 C_E，也不把平方根对数上界从这个拼接中推出。

结论的量词是：对每个有限 n、每个源权重下界 γ、每个有限目标字母表，存在一个确定性完整层级，所有容量同时满足对数界。n≤1/γ 是下界质量的必然后果，而不是额外固定 n 假设。共同非负 Bayes baseline 可用于把乘法上界转成总风险上界；该操作不能把超额风险下界自动转成总风险下界。

## 端点有界反例的近邻与未覆盖范围

| 一手来源与实际核对位置 | 已有内容 | 对固定光滑生成函数反例的覆盖程度 |
|---|---|---|
| Banerjee、Merugu、Dhillon、Ghosh，*Clustering with Bregman Divergences*，JMLR 2005，[全文](https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf)，Proposition 1、Definition 2 | 任意正权有限随机变量的右 Bregman 最优代表为期望；相应 Bregman information 为到均值的期望散度。 | 直接覆盖本项目 Jensen gap 与 centroid distortion 的恒等式；不覆盖跨 k 最优层级无界性。 |
| Telgarsky 与 Dasgupta，*Agglomerative Bregman Clustering*，ICML 2012，[作者全文](https://cseweb.ucsd.edu/~dasgupta/papers/relabreg.pdf)，Proposition 3.8、Algorithm 1、Sections 2–5 | 允许非可微凸生成函数，建立 Bregman 合并中心恒等式，处理小簇退化与几何平滑，给出贪心凝聚算法。 | 必须引用的直接 Bregman 层级先例；全文未给所有层级的最坏乘法价格或有限端点导数刻画。 |
| Ackermann 与 Blömer，*Coresets and Approximate Clustering for Bregman Divergences*，SODA 2009，[作者全文](https://cs.uni-paderborn.de/fileadmin/informatik/fg/cuk/Forschung/Publikationen/CoresetsAndApproximateClusteringForBregmanDivergences.pdf)，Definition 2.2、Lemma 2.2 | μ-similarity 要求存在固定正定 A，使全部 p、q 满足 μD_A(p,q)≤D_φ(p,q)≤D_A(p,q)；由此获得近似对称与松弛三角性质，再做 flat coreset／近似。 | 统一二次比较强于 φ′ 有界。不能用该结果证明 bounded slope 足以使层级价格有界。 |
| Gneiting 与 Raftery，*Strictly Proper Scoring Rules, Prediction, and Estimation*，JASA 2007，[作者全文](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)，Theorem 2、Equation (12)、Theorem 3 | 二元严格 proper scoring rule 可由严格凸 G 生成；奖励为 G(p)+(1−p)G′(p) 与 G(p)−pG′(p)。曲率可解释为 Schervish 表示的混合测度密度。 | 从固定光滑 φ 导出有界 proper loss 属既有表示的应用，不应单列成新理论；论文未研究 cardinality hierarchy 的无界比。 |
| Bao，*Proper Losses, Moduli of Convexity, and Surrogate Regret Bounds*，COLT 2023，[全文](https://proceedings.mlr.press/v195/bao23a/bao23a.pdf)，Theorem 6 | 任意 regular proper loss 的点态 Bregman regret 用广义熵的凸性模数控制估计误差；比单一 strongly proper 条件更精细。 | 说明局部曲率／凸性模数是已有理论对象；其“hierarchy of losses”是损失分类，不是本项目的嵌套划分。 |
| Nock、Menon、Ong，*A Scaled Bregman Theorem with Applications*，2016，[作者全文](https://www.ong-home.my/papers/nock16bregman.pdf)，Theorem 1、Section 5 | 证明特定数据变换与生成函数变换下的 scaled isodistortion；应用之一是在平面与曲面间转移聚类问题。 | 可作 Bregman 变换背景，但未包含在单个闭区间光滑生成函数内拼接无限困难实例的层级下界。 |

反例应保留的准确表述为：**存在一个固定的 φ**，使得 **对每个 M 都有一个四状态全支撑实例**，其 **每一个允许的确定性层级** 在 k=2 或 k=3 至少一层具有超过 M 的 **超额风险** 近似比。只有确实证明到闭区间，才写 φ∈C∞([0,1])；仅证明内部光滑不足以得到所有导数在闭区间有界。

有界 φ′ 不排除曲率在某些尺度上任意弱。若反例同时有全域 0<m≤φ″≤M<∞，就会与平方误差比较冲突。因此闭区间 C∞ 构造中的端点或累积点必须允许曲率退化；严格凸不等于二阶导数处处有统一正下界。该逻辑是直接数学检查，不是某一已核实论文的结论。

反例不会证明“凡端点导数有界都能无界”，也不会给出全部生成函数的充要分类。它只排除“端点导数发散是无界层级价格的必要条件”。若保持四点、固定生成函数、任意最优层级及超额风险这四个量词，便不会与贪心算法失败、改变生成函数的族或 total risk 结论混淆。

## 检索证据的限度

本轮以 Bregman／KL hierarchical clustering、price of hierarchy、source minimum mass、bounded derivative／gradient、smooth proper scoring、successive refinement、nested quantization 等组合检索，并沿上述一手论文的定理与引用追踪。窄关键词查询经常返回大量无关结果，因此未命中的证明力有限。对优先权最有用的正面证据，是上表各定理对象与所选命题量词的明确差异。

尚未补齐的访问缺口是 Lin 2010 原始全文、Plaxton 2006 全文及 Großwendt 2020 原始博士论文；后者官方 Bonn 链接返回访问限制。不能据本次检索写“此前从未研究”或“首个”。相较早期文献记录，新增最有用的证据是 Sason–Verdú 的明确似然比比较、Telgarsky–Dasgupta 的 Bregman 凝聚原始论文、μ-similarity 的原始定义，以及 proper scoring／凸性模数对端点候选的准确边界。
