# Phase 7 有限理论候选结论

作者：Wenyu Huang，UCSD。核对日期：2026-10-07。

本目录只记录候选数学结果和核验；未改原项目、Phase 6 或其作者信息。

**最终主线更新：** 本报告以下 Shannon-copy 构造已冻结为先行备份。随后同一主线被加强为
**四个源质量恒为 1/4** 的完全自足反例；最终收录应优先使用
[equal_mass_counterexample.tex](equal_mass_counterexample.tex) 与
[强化结果报告](equal_mass_route_report.md)。没有覆盖本文件所述的先行证明。

## 本轮完成的可证增量

[完整证明](smooth_generator_counterexample.tex)给出一个固定的、对称的
`φ ∈ C∞([0,1])`，满足 `0 ≤ φ″ ≤ 1`、每个内部点 `φ″ > 0`、两端一阶导数有限，
但四状态二元标签全支撑实例的最优层级**超额风险**比可以任意大。
确定性比仍为 `√j/(2√log 2)(1+o(1))`，允许随机编码和随机退化后的比为
`√j/(4√log 2)(1+o(1))`。

量词是 `存在单个 φ，使每个 M 都存在一个实例，所有允许的层级均有某一层的超额比 > M`。
没有让生成元随实例变化。四个后验都处于 `(0,1)`，两项平坦最优超额风险均严格正。

构造在趋近零的两两不交小区间内精确复制越来越长的 Shannon 困难实例。
曲率幅度下降快于任何所需导数增长，使整个生成元在闭区间光滑。
每个实例的所有簇均值以及所有随机编码后验仍落在同一个复制区间，所以
**每一个**分区/随机编码的成本都等于参考 Shannon 成本乘同一个正数。
这一步给出任意层级的下界，不需要把有限算法的失败推成不可实现性。

两个必须随命题一起保留的界线：

- 严格凸不等于全局强凸。本构造 `φ″(0)=φ″(1)=0`，且端点所有更高导数平坦；
  任一固定内部闭子区间仍有正曲率下界，不与原稿的有界曲率比较矛盾。
- 这只反驳“端点斜率发散是超额价格无界的必要条件”。所导出的有界 strictly proper
  loss 在此实例序列中的**总风险比反而趋于一**。证明末尾给出该上界；不能删去这个限制。

该结果适合定位为原 `general_losses.tex` 的假设边界补充，而非另立一个通用学习理论。
它不是所有生成函数的充要分类，不提供样本复杂度或计算复杂度下界，也不声称独立理论论文的充分分量。

## 核实后不应再当作新主线的内容

| 方向 | 已核实状态 | 本轮判断 |
|---|---|---|
| 四状态最坏源质量依赖的阶 | Phase 4 `sharp_source_mass.tex` 已证 `Θ(√log(1/γ))`；原始 `main.tex` 内旧对数界不是最新成果 | 严格重合，不应再次宣称闭合 |
| 特定四点族的 sharp 系数 | 原 `main.tex` 已有确定性及随机编码系数，两者相差 2 | 严格重合 |
| 四状态全部分布的最坏 leading coefficient | 现有普适上界系数和特定族下界系数尚不匹配 | 本轮仍未解决，不把特定族极值替代全局极值 |
| 任意 n 的 `O(log(1/γ))` | 可由 KL/Hellinger 源质量比较与既有平方欧氏层级常数结果拼接 | 至多是明确推论，不能作主要原创成果 |
| 任意 n 的 `O(√log(1/γ))` 或多层更强下界 | 现有四点证明不能直接推广；尚无本轮完整证明 | 保持开放，不用增添无关状态冒充多层障碍 |
| 四点最优划分不能形成层级 | AIB、Kartowsky–Tal 已有明确先例 | 已知 qualitative pattern |
| 有界曲率与平方误差比较 | 原 `general_losses.tex` 已有；Bregman μ-similarity 文献也明确要求全域二次比较 | 已知方法，不能改名为新定理 |
| Phase 5 谱截断、二次投影与常数预处理反例 | 已读实际 `theory.tex`、`nonlinear_projection.tex` 和 `source_counterexample.tex` | 与本反例不同；不借其经典部分扩张本轮新颖性 |

## 一手文献与新颖性措辞

[独立文献审计](literature_audit.md)列出 11 项一手来源、实际定理范围及仍有的全文访问缺口。
需要保留的直接背景至少包括：

- [Banerjee 等，JMLR 2005](https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf)：Bregman centroid 与信息量恒等式。
- [Telgarsky–Dasgupta，ICML 2012](https://cseweb.ucsd.edu/~dasgupta/papers/relabreg.pdf)：Bregman 层级凝聚、合并中心公式及退化簇处理。
- [Ackermann–Blömer，SODA 2009](https://cs.uni-paderborn.de/fileadmin/informatik/fg/cuk/Forschung/Publikationen/CoresetsAndApproximateClusteringForBregmanDivergences.pdf)：全域 μ-similarity 条件；不等于一阶导数有界。
- [Gneiting–Raftery，JASA 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)：凸生成函数与严格 proper scoring 的既有表示。

本轮未检出同量词的固定闭区间光滑生成元反例。这个结论只支持“在已核对来源中未发现”，
不支持“首次”“从未有人研究”或已经认证优先权。Lin 2010、Plaxton 2006 和 Großwendt
论文全文的访问缺口仍在文献审计中明列。新的数学步骤是光滑拼接与所有编码成本的精确转移；
cutoff、仿射不变性、proper-score 表示及原四点下界本身都不是新工具。

## 有限核验

标准库脚本 [check_four_state_transfer.py](check_four_state_transfer.py)运行成功，退出码 0。
它逐个枚举四点的 7 个二簇划分、6 个三簇划分和全部 18 个兼容层级，核对参考 Shannon
族的两项平坦最优值与 `min(a/r,t/(2a))`。使用 Python Decimal，不训练、不安装依赖。

| j | 最优确定性超额价格 | 除以 `√j/(2√log 2)` |
|---:|---:|---:|
| 8 | 1.613875918473 | 0.950096840753 |
| 16 | 2.381195486893 | 0.991237641340 |
| 32 | 3.406699445020 | 1.002770517566 |
| 64 | 4.825584345884 | 1.004390624674 |
| 128 | 6.817870765649 | 1.003428671461 |
| 256 | 9.630226641500 | 1.002211199609 |

此检查只核对引入的有限成本和枚举，不是光滑拼接或无穷渐近结论的数值证明。
那些部分由 TeX 内的逐阶导数估计和精确恒等式证明。尤其 `j=256` 时原 Shannon baseline
与 `D3*` 之比仍约 0.418，不能凭此有限表把 total 与 excess 当作已数值相同；两者渐近等价
用的是理论极限，而本构造导出 bounded loss 的 total 与 excess 更完全不同。

## 剩余工作

本文件保存较早候选证明的历史状态。最终整合稿采用等质量加强版本，保留
excess-only 与 total ratio → 1 两项限制。全体四点分布的全局 sharp coefficient、任意 n
的平方根对数界以及生成函数的完整分类均不在本轮已完成结果之内。
