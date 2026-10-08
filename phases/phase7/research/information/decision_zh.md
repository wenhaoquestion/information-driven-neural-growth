**结论：信息增益—风险改善路线保留为经典工具箱与反例附录，停止把它作为 Phase 7 的核心创新。** 本轮没有找到足以独立支持新论文的非平凡信息论命题。直接可用的英文命题与完整证明见 [theory.tex](theory.tex)，独立编译入口为 [main.tex](main.tex)，书目为 [information_references.bib](information_references.bib)。未修改原项目、未训练、未安装依赖。

本笔记逐一核对了原项目 `manuscript/main.tex` 的表示／对数风险定义与层次框架、`general_losses.tex` 的凸生成元和曲率比较、Phase 3 `theory.tex` 的嵌套 Brier 投影与审计接口、Phase 5 `nonlinear_projection.tex` 的宽度无关逼近下限、Phase 6 `theory.tex` 的零输出系数出生和实际 factor Adam 边界，以及已有研究决策报告。没有将项目已承认经典的内容重新命名为创新。

可直接采用的六组命题：

| 命题 | 精确内容与量词 | 本项目用途 | 新颖性判定 |
|---|---|---|---|
| 1. 任意表示的 log 风险差 | 有限标签；任意共同概率空间上的 `Z,Z′`；`B_log(Z)−B_log(Z′)=I(Y;Z′|Z)−I(Y;Z|Z′)` | 重训旧特征不一定细化；必须扣除遗忘信息 | 链式法则的直接恒等式 |
| 2. 平方和分类 Bayes 界 | `Y−Z′−Z`；平方风险需二阶矩，有限区间 `[a,b]` 时 `Δ_sq≤(b−a)² I/2`；分类 `Δ_01≤sqrt(I/2)` | 说明 CMI 给的是 Bayes 机会的上界，非当前优化器收益 | Pinsker／条件均值经典结果；二元常数渐近 sharp |
| 3. 函数保持与表示信息 | 可互相恢复的表示信息相同；但零输出系数的新 ReLU 可使表示 CMI 正、当前风险严格不变 | 与 Phase 6 实际出生机制直接一致 | 初等显式反例，无新颖性主张 |
| 4. 四个反例 | 高 activation entropy 而 label info 为零；正 CMI 而分类 Bayes gain 为零；正 CMI 而实值平方 Bayes gain 为零；固定结构信息成本而所有 gains 任意小 | 排除常见不正确研究目标 | 经典独立性／决策充分性边界的具体实例 |
| 5. 拟合风险分解 | 精确式 `R=B+A+F`，再分 `F=O_emp+E`，`|E|≤2 sup|R−Rhat|` | 分开 Bayes 值、受限 decoder 逼近、经验优化与估计 | 加减项代数；不自动提供集中或优化保证 |
| 6. 多轮信息总预算 | 同一共同历史下确定性嵌套，或完整反向 Markov 降质链；正向 CMI 和望远镜为终点 CMI | 允许写总 Bayes 预算；禁止写 optimizer 每步下降 | 标准 chain rule/data processing |

**最实用的显式神经例子。** `X∈{−1,+1}` 等概率，`P(Y=1|X=+1)=1−δ`、`P(Y=1|X=−1)=δ`，`0<δ<1/2`。旧预测为 `1/2`。增加 `ReLU(X)` 但输出系数为零，则当前预测逐点不变，平方风险仍为 `1/4`；新隐藏表示的标签 CMI 为 `log 2−h(δ)>0`，其 Bayes 平方风险改善为 `(1/2−δ)²`。新 Bayes 均值可以被一个仿射 decoder 实现，故此例甚至不依赖 decoder 表达不足，只需出生时尚未拟合系数。不得将“函数保持”一概改写成“隐藏表示互信息不变”。

**遗忘反例。** `X=Y` 为均匀二元，表示在常量和 `X` 之间交替。每次恢复 `X` 新增正向 CMI `log 2`，每次忘记 `X` 正向 CMI 为零。经过 K 个完整循环，正向和为 `K log 2`，终点信息为零。因此任意重训网络的单步正向 CMI 不可直接相加当作一个有限收益预算；必须证明嵌套，或扣除每步反向 CMI。把所有历史表示留在累计 transcript 可恢复嵌套，但 transcript 的存储／访问不免费，其 Bayes 信息也不等于部署网络的可用表示。

**量词容易误写的地方。**

- 二元 Brier 在此为标量 `(Y−p)²`；若用两类坐标平方和，增益与界的常数整体乘二。所有信息以 nats 为单位。
- 正 CMI 不保证分类增益严格正，但本反例不主张在每一个可能的较大 CMI 水平都可做到零分类增益。
- 二元平方风险不存在统一正线性反向常数：完美揭示 `Y∼Ber(p)` 时 `Δ_sq/I=p(1−p)/h(p)→0`。这不等于否定所有非线性下界。
- 一般实值回归的 CMI 描述整个条件分布；平方 Bayes 风险只看均值。`Y=S(1+U)`、独立对称 `S` 与均匀二元 `U`、新表示 `U` 的 CMI 为 `log 2`，两条件均值都是零，平方改善为零。Phase 6 的实值二次响应不能直接套二元后验解释。
- `I(X;Z′|Z)` 是输入信息量；它不是参数个数、浮点比特、运行时间或计算量。确定性连续表示的该信息量还可能无限，故不能直接充当 Phase 6 的有限资源账本。
- `F=R(f)−inf_G R` 是实际总体拟合缺口，包含优化和估计；只有 `O_emp=Rhat(f)−inf_G Rhat` 可直接称为经验优化残差。`E` 有符号。数据依赖类的逐实现代数不保证统一集中。
- 统一条件于完整训练历史／算法随机性并使用独立新评估样本即可应用固定表示公式；逐步替换为不同 `H_t` 而省略新历史项，并非同一个证明。

**一手文献对应与已核验范围（2026-10-07）。**

| 来源 | 已打开并核读的位置、假设 | 对应关系与限制 |
|---|---|---|
| [Gneiting–Raftery 2007，作者 PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) | Theorem 1：凸概率分布类上的 regular proper score 与凸函数／subtangent；§2.2：entropy 与 associated divergence | 不得将 proper scoring 的 Jensen/Bregman 表述作为新原理；reward 与本文 loss 的符号相反 |
| [Reid–Williamson 2011，JMLR 全文](https://www.jmlr.org/papers/volume12/reid11a/reid11a.pdf) | Theorem 8：proper loss、binary discriminative task 的 Bayes-risk reduction=Bregman information；Theorem 10：binary experiment `(P,Q)` 与 prior `π`，联系 f-divergence、statistical information、Bregman information；§4.6 追溯 DeGroot 1962 | 对 coarse 每个条件单元应用即可得到细化 Jensen 缺口；此连接有直接成熟先例。未声称它保证任何 neural optimizer 收敛 |
| [Sason–Verdú 2016，v4 全文](https://arxiv.org/html/1508.00335v4) | Eq.(1) 的 Pinsker 中其 `|P−Q|` 是 L1 而非本文 TV；Eq.(5) `D≤log(1+χ²)`；Theorem 1 功能支配及最优常数条件；§VI, Remark 32 的最小质量反向界 | 本文二元 `Δ_sq≤I/2` 及 coarse posterior away-from-zero 的方便反界均为直接应用；这里没有包装新的 reverse Pinsker theorem |
| [Tao 2006，期刊全文](https://cdm.ucalgary.ca/article/download/61900/46638/176530) | Lemma 4.4：离散 `X,Y,Y′`，`Y`决定`Y′`，`X∈[−1,1]`；给 absolute-mean deviation `≤2 sqrt(I)`；原文熵为 log2。Lemma 4.3 有 finite variables 与 entropy-increment construction | 是 conditional information→conditional expectation 的直接早期先例；原文 Lemma 4.4 是 L1 形式、常数并非最优，不能误引成本文 sharp squared 形式 |
| [Polyanskiy–Wu，作者 2024-08-16 书稿](https://people.lids.mit.edu/yp/homepage/data/itbook-export.pdf) | Theorem 7.10 Pinsker；Corollary 7.11：Markov `Y→X→X′`、`Y∈[−1,1]`，平方均值差 `≤2 I/log e`，并明确常数不可改进；§2.5 chain rule/DPI | 这是恰好覆盖本文 bounded-square 命题的标准教材表述，作为已知性核对；本文还给出独立 Pinsker 证明，不依赖书稿认证新结果 |
| [Net2Net，v4 全文](https://arxiv.org/html/1511.05641v4) | §2.3 复制 hidden units 并分配 outgoing weights；前 n 个输出坐标保留；随后对称破缺扰动需另论 | 精确复制的 complete hidden vector 可互相恢复，信息不变。原论文没有因此证明任意零输出扩张的隐藏 CMI 为零；本文特意保留此区别 |

另检查了 [Banerjee–Guo–Wang 2005 的官方摘要](https://ieeexplore.ieee.org/document/1459065/) 与 [Cornell 作者技术报告页](https://ecommons.cornell.edu/entities/publication/4d8302b3-7259-4910-955b-c86d5564c556)：摘要明确 conditional expectation 对 Bregman loss 最优。本轮未核读该论文完整定理，故没有据此声称其 exhaustive characterization 的细节，也没有把它作为本附录证明的承重引用。

**研究裁决。** 这些命题能严谨约束项目叙述，但不足以支撑 Phase 7 新颖性。建议正文仅保留 zero-output birth 例子、任意表示的遗忘恒等式，以及 Bayes／受限拟合缺口；其余留作附录或审查清单。若主线转向精确历史经验查询的必要记忆量，本附录只负责阻断“CMI 自动兑现成学习收益”这条错误推理，不与主线争夺贡献定位。不能把新增证明文件数量当作新增研究成果数量。

独立数学复核已逐项核验六个核心公式和例子，没有发现实质错误，并提示了 Brier 归一化、共同训练历史、完整 Markov 方向及非嵌套情形的界限。该复核是内部交叉检查，不是外部同行评审或新颖性认证。

LaTeX 已完成 `pdflatex → bibtex → pdflatex` 引用收敛与最终复编译；[main.pdf](main.pdf) 共 5 页，最终退出码 0，日志无 warning／overfull。此处为原研究时期保存的构建状态；本公开副本不包含本地构建日志。此次整理未重跑该附录构建或项目训练代码。剩余科学限制是本附录不提供新颖性或实际优化收益。
