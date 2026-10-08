# Phase 7 来源与证据索引

历史核读日期：2026-10-07。以下保留原研究时的项目相对位置；不是公开目录的可点击路径。

以下 SHA-256 标识 2026-10-07 研究时引用的原始字节；公开副本可能因路径或私人元数据清理而不同。此次整理未重新检索文献、未重跑旧实验。早期阶段不是编译 Phase 7 主稿所需的外部文件依赖。

- `manuscript/main.tex` — 原有限定义、确定性及随机四状态层级障碍。
  SHA-256: `b4da7ccc71e6685074bd87dbdb5756c3a0c04442180f04f0f0c589455ec3bb51`。
- `manuscript/general_losses.tex` — 严格 proper loss、Jensen/Bregman、端点发散充分条件及曲率比较。
  SHA-256: `d46c9a589a482d85836510454a4e68c77d6a5350e56ac787863200236f6f83a0`。
- `phase2/manuscript/main.tex` — Phase 2 理论背景。
  SHA-256: `7c8f2da0bbae1a424c22ac4ec81dd66cc674018aa9c7ed3393d633e7098496d2`。
- `phase3/manuscript/theory.tex` — 嵌套表示的 Brier 投影与风险接口。
  SHA-256: `82e0fc6249986c9c2404eb2dc6768c05518752b12f582d7bd6a92fe4096a3aba`。
- `phase4/publication/finite/manuscript/sharp_source_mass.tex` — 最新四状态源质量平方根对数上界与同阶下界；勿以旧稿替代。
  SHA-256: `612b8796921f0828aba1f5db56847227e9c90b1f17eda6c739c33c1363a13e82`。
- `phase5/manuscript/theory.tex` — Gaussian 二次谱结构与样本更新的假设边界。
  SHA-256: `6d51887668fb09c9b1bcf55fdd9c56d548390751db25f40de3b54d5656fca2ba`。
- `phase5/manuscript/nonlinear_projection.tex` — 非线性教师的二次投影和无法由同类增宽消除的逼近下限。
  SHA-256: `3bae6b950e7ad24e1e587b272bf48d5bc1028ae52d577d7f3995df50b8b5a106`。
- `phase5/manuscript/source_counterexample.tex` — 预处理常数反例；不是本轮新结果。
  SHA-256: `2bb3066f8504fb02892b4fdc602b85c5d91901e179b65496e00578e0fc7f94aa`。
- `phase6/manuscript/theory.tex` — 第117–126行：精确经验G及历史proxy S−B的区别；factor Adam边界。
  SHA-256: `14cbffe327746c6c0e985278986b9a648239ef4292e3d97f175be4dbbaab7245`。
- `phase6/code/frontier.py` — 实际出生与历史矩近似、成本记账实现；本轮只读。
  SHA-256: `295aa086521a69ddabc5483e7fcedfb1e5a0159d8813599c4169b8bb41325ce0`。
- `phase6/analysis/confirmation/analysis_summary.json` — 既有确认分析与主检验；本轮不重跑训练。
  SHA-256: `140a229e3164debed59901046d6967acdc9ef3cdff5e7a2da3994cdfc81651a5`。
- `phase6/PHASE6_DECISION_ZH.md` — 阶段6完整实验决策与限制。
  SHA-256: `f42924a90721ef983316ec2bb83492c31731802ecc55a29968c8bc8465911008`。
- `phase6/reviews/final_statistics_audit.json` — 既有最终保存统计审计。
  SHA-256: `5582a4cc8d38492f8d7be6c28210e35252fe4576d7c39384b745503d43dd9f26`。

## 新增结果入口

- `manuscript/main.tex`：最终英文整合稿；英文PDF为 `output/pdf/phase7_working_paper.pdf`。
- `research/memory/exact_query_memory.tex`：精确查询等价、连续存储维数、局部比特界与量词例外。
- `research/finite/equal_mass_counterexample.tex`：等源质量的固定光滑损失反例；最终有限分支主结果。
- `research/finite/smooth_generator_counterexample.tex`：较早的Shannon复制版本，保留研究轨迹；最终主结果不依赖它。
- `research/sketch/query_approximation.tex`：经典FD保证到经验查询误差的直接转移。
- `research/information/theory.tex`：经典信息论桥梁、遗忘项及反例。
- `research/finite/literature_audit.md`：有限层级的一手文献与访问缺口。
- `research/information/decision_zh.md`：信息论原始定理量词核对。
- `TECHNICAL_REVIEW.md`：历史内部技术审查的整理说明；不是外部同行评审。
- `research/memory/check_results.json`：标准库精确分数小维代数核验。
- `research/finite/check_four_state_transfer.py`：较早Shannon参照的18层级枚举。
- `PHASE7_DECISION_ZH.md`：中文研究路线、验收和停止标准。

## 关键一手链接

- Wagstaff et al. (2019), Theorems 3.2 and 4.1: https://proceedings.mlr.press/v97/wagstaff19a.html
- Ghashami et al. (2015), Theorem 1.1, arXiv v2: https://arxiv.org/pdf/1501.01711
- Frequent Directions 作者代码（只读核对，未运行）: https://github.com/edoliberty/frequent-directions
- Brouwer维数不变性，作者公开证明说明: https://terrytao.wordpress.com/2011/06/13/brouwers-fixed-point-and-invariance-of-domain-theorems-and-hilberts-fifth-problem/
- Telgarsky–Dasgupta (2012): https://cseweb.ucsd.edu/~dasgupta/papers/relabreg.pdf
- Banerjee et al. (2005): https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf
- Reid–Williamson (2011): https://www.jmlr.org/papers/volume12/reid11a/reid11a.pdf
- Gneiting–Raftery (2007): https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Net2Net: https://arxiv.org/abs/1511.05641

完整书目及其他文献的已核验位置在分支审计和 `manuscript/references.bib`。未检出同量词先例不构成首创证明。
