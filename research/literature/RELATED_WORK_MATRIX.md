# 最近邻矩阵：自适应概念测量、可靠停止与真实成本

核验截至：2026-09-21。范围：公开一手论文/正式论文集；不是穷尽性综述。已读范围逐条记录于 `VERIFIED_SOURCES.json`，未逐行检查所有附录证明，也未复现文献结果。新近 arXiv 工作只按预印本处理，不因时间新而称为公认最强方法。

## 结论先行

“依照已观测概念选择下一个概念，获得自动回答，适时停止”属于主动特征获取（AFA）与有噪序贯实验设计的交集，不足以单独构成方法创新。“再加可靠停止/重新校准”也已有直接近邻。特别是 2026 年 BCEA、RouteCert、EDFA 覆盖了原 proposal 中部分拟议延伸，需要实质性收缩原创性表述。

现阶段可争取的增量不是创造一个新组合名字，而是回答：**同样的单概念质量，在不同语义误差依赖和真实共享计算成本下，什么情况下自适应概念测量值得做，什么情况下应一次批量完成？** 新结论须来自受控干预、真实响应器和公平强基线，不能由下面的文献差异直接推导“创新成立”。

## 关键最近邻

表中“差异”只说明任务边界，不代表该差异已足以发表。S01–S21 对应 JSON 条目。

| ID / 一手来源 / 状态 | 已有核心能力及与本项目重合处 | 可区分边界、实验责任 |
|---|---|---|
| S01 [BCEA: Look Again Before You Abstain](https://arxiv.org/abs/2606.16667)，2026 预印本，核验 v4 | 在视觉语言判断中获取额外视觉证据，比较回答、拒答、再次观察；校准部署后的获取流程。 | 直接阻止把 ask/verify/abstain 或 acquisition 后重校准当主创新。其任务不是显式 CBM 概念价值学习；应迁移其校准设计，不能宣称复现原任务。v4 的 exact 有限网格修正与 practical 扫描须分开。 |
| S02 [RouteCert: Conditional Validity for Adaptive Modality Acquisition](https://arxiv.org/abs/2608.15520)，2026 预印本，v2 | 认证策略产生的终止模式；处理使用校准结果选择完整策略的情形；包含小联盟获取和校准样本不足。 | 最强可靠停止重合。文中获取成本按源相加；许多实验认证对全信息参考模型的一致性，另有真实标签实验。我们需使用真实标签风险、实测非加性计算成本，不能把这两个差别包装成新校准定理。 |
| S03 [Explanations-Driven Active Feature Acquisition for Algorithmic Recourse](https://arxiv.org/abs/2609.12179)，2026-09-10 预印本 | 以解释价值/成本获取特征；联合 recourse 和部分信息决策，并给出可靠性校准。 | Recourse 有效性不是 CBM 语义正确性，也不自动等于真实标签错误率。直接削弱“可解释获取+有保证停止”的宽泛主张。 |
| S04 [BRiG-AFA: Bellman Risk-to-Go Learning for Non-Myopic Active Feature Acquisition](https://arxiv.org/abs/2608.02305)，2026 预印本 | 用监督 Bellman 回归学习候选条件的剩余风险；输入观测值、mask、候选和预算；无需专家轨迹。 | 原 proposal 的 risk-to-go 控制头与其高度重合。给它同样的有噪概念响应、任务头和批动作扩展。原文也未声称已完成全面 SOTA 验证，不宜只战胜它就下结论。 |
| S05 [Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition](https://proceedings.mlr.press/v235/valancius24a.html)，ICML 2024 | 非贪心 AFA，直接考虑联合获取及成本，避免只看单步收益。 | 是实质性强基线，不是只在 related work 提一句。不能把“组合收益、没有 RL、没有专家查询顺序”说成新。输入必须与我们的自动概念测量完全相同。 |
| S06 [Stochastic Encodings for Active Feature Acquisition](https://proceedings.mlr.press/v267/norcliffe25a.html)，ICML 2025 | 随机潜变量编码支持 AFA；避免纯贪心与直接 RL 的部分问题；附录也评估加性特征噪声。 | 需要纳入正式基线候选。噪声实验不等于真实 VLM 语义错误，但不能写“既有 AFA 从未考虑噪声”。 |
| S07 [Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding](https://papers.nips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html)，NeurIPS 2018 | 集合状态、联合分类/获取、依赖已观测值的序贯选择。 | mask-aware/DeepSets 状态、STOP 与预算化获取本身均不是新概念。可作为历史 RL 路线参照，资源有限时优先 S05–S06。 |
| S08 [Near-optimal Bayesian Active Learning with Correlated and Noisy Tests](https://proceedings.mlr.press/v54/chen17b.html)，AISTATS 2017 | ECED 明确考虑给定目标后仍相关的有噪测试，研究依赖下的信息获取与保证。 | “错误相关性影响下一步”“非贪心互补性”不是新科学发现。我们的候选增量须是自动语义响应器中可量化、可迁移的具体规律，而非重新展示 XOR。 |
| S09 [Near-Optimal Bayesian Active Learning with Noisy Observations](https://papers.nips.cc/paper_files/paper/2010/hash/1e6e0a04d20f50967c64dac2d639a577-Abstract.html)，NeurIPS 2010 | EC² 将带噪查询与等价类识别联系起来，已有昂贵测试、不同成本及依赖建模。 | 历史优先权边界。不能以“模型回答有噪”把问题与全部先前序贯测试工作切断；其已知概率模型假设与学习真实语义响应的差异应准确说明。 |
| S10 [Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning](https://arxiv.org/abs/2402.15610)，ACL Findings 2024，ReCoVERR | LLM 基于已收集证据提出后续视觉问题，筛选可靠/相关回答，证据充足才回答，否则拒答。 | 直接自动语义查询近邻。原方法从一个原始 VLM 答案开始并验证它，不是严格概念-only 分类器；我们应比较受限证据策略，且不能把原文经验风险表现改写为普适保证。 |
| S11 [Interactive Concept Bottleneck Models](https://ojs.aaai.org/index.php/AAAI/article/view/25736)，AAAI 2023，CooP | 用不确定性、影响和成本选择要请人修正的概念。 | 主要是已预测概念的专家干预，不是没有专家的自动测量。但“选哪个概念更值钱”不是本项目独有。其取得正确概念值的 oracle 实验只能作为上界。 |
| S12 [Learning to Receive Help: Intervention-Aware Concept Embedding Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract.html)，NeurIPS 2023，IntCEM | 模拟干预训练与学习求助/概念干预策略。 | 不需要真人标查询顺序也可以训练策略早已成立。软概念嵌入可能带额外信息；不能让 IntCEM 获取真概念而我们获取有噪响应后直接比较成本。 |
| S13 [Addressing Leakage in Concept Bottleneck Models](https://papers.neurips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html)，NeurIPS 2022 | 包含自回归概念预测与硬概念信息通道讨论。 | 固定顺序预测的条件依赖不同于选下一个测量；“按顺序预测概念”以及 hard bottleneck 均不宜作首要原创性。我们的主实验不采用未受语义约束的 side channel。 |
| S14 [Matryoshka Concept Bottleneck Models](https://arxiv.org/abs/2605.20612)，2026 预印本，v3 | 全局相关性/冗余排序的嵌套概念前缀，多个概念预算与专家干预。 | 必须比较静态前缀；它不是依赖每个实例已返回概念值的动态下一问。其对数成本等理论表述有假设，不能移植成我们实测计算保证。 |
| S15 [Selective Concept Bottleneck Models Without Predefined Concepts](https://lmb.informatik.uni-freiburg.de/Publications/2025/SAB25/paper-ucbm.pdf)，TMLR 2025 | 输入相关的稀疏概念选择，减少参与决策的概念。 | 稀疏使用不等于免去所有候选的前端计算。本项目应报告“测量了多少”与“最终用了多少”的区别，不能将这种差异称为泛化性能优势。 |
| S16 [Zero-Shot Active Feature Acquisition via LLM-Elicitation](https://arxiv.org/abs/2606.18933)，2026 预印本 | LLM 提供离线结构/统计知识支持零样本 AFA；在线查询实际特征值。 | “LLM + 零样本选下一特征”不是新。它的获取值假设与有噪自动语义测量不同；需要清楚区分结构先验生成器和响应器。 |
| S17 [Learning-To-Measure: In-Context Active Feature Acquisition](https://arxiv.org/abs/2510.12624)，2025 预印本，核验修订版 | 预训练跨任务获取策略，以 in-context 方式适应新任务。 | 无本任务专家轨迹不意味着无其他训练数据；不能把任务适应与严格零样本混用。可作跨数据集泛化参照，不是最小主实验必跑项。 |
| S18 [Towards Cost Sensitive Decision Making](https://proceedings.mlr.press/v258/li25h.html)，AISTATS 2025 | 主动获取 POMDP、模型化未观测信息、不同获取过程与层次策略。 | acquisition+decision 联合建模与批/序获取已有基础；本项目更窄，固定实例、语义瓶颈、测量计算。不能把 POMDP 形式化本身当贡献。 |
| S19 [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052)，预印本 2021；Annals of Applied Statistics 2025 | 学习后对候选规则进行风险检验；多重检验框架支持有效选择。 | 应优先使用现成风险控制技术。冻结有限策略族、union bound/Bonferroni 不是新理论。正式出版元数据另经[作者机构页面](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk)核验。 |
| S20 [CAP: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://jmlr.org/papers/v26/24-0452.html)，JMLR 2025 | 在线样本选择后的预测校准、特定条件下的 selection-conditional 保证与 FCR。 | 它不是在单个样本内获取概念；但“选择后校准”是一条成熟研究线。不得把其 coverage/FCR 保证直接等同于我们分类错误率。 |
| S21 [Interpretable Reward Modeling with Active Concept Bottlenecks](https://arxiv.org/abs/2507.04695)，2025 预印本 | 用信息增益选择概念标注，服务可解释偏好/奖励学习。 | 是训练期主动标注，不是部署时自动测量。列出是防止术语误匹配；本项目不新增专家标注，不能按其原设定获取人工反馈。此项仅摘要级核验，不作具体实现依据。 |

## 三个最近来源的精读警戒线

### BCEA：需要读最新 v4，而非只看早期摘要

[v4 正文](https://arxiv.org/html/2606.16667v4)明确区分具有有限候选校正的 exact 方案和可能轻度反保守的 practical 阈值扫描。因此实验必须匹配校准程序和多重比较预算。固定、与认证集无关的完整 acquisition 流程可以整体纳入校准；不能笼统写“自适应本身破坏 exchangeability”。真正的问题是部署对象改变了却沿用旧阈值，或认证数据反过来改变路由而没有处理选择。

### RouteCert：比“获取之后做温度缩放”近得多

[v2 第 3–5 节](https://arxiv.org/html/2608.15520v2)处理完整策略及其产生的终止模式，并非只校准一个静态分类头。模式条件保证、完整策略族认证、获取小联盟、认证样本碎片化均已有专门分析。本项目若只换成 concept 字段，差异太薄。需区分：语义准确性、真实标签风险、全模型一致性是三个不同目标。

### EDFA：确认编号与目标，防止误引用

[2609.12179 正文](https://arxiv.org/html/2609.12179v1)是算法 recourse 导向的获取，不是已知的“Jev-CBM”。其部分信息 recourse 的有效性与我们概念预测风险不是同一随机变量。可借其实验约束与校准思想，但不能照搬保证后声称诊断安全；官方初次提交日期为 2026-09-10。

## 最小公平对照集合

1. 非学习参照：全概念单次批量、静态排序/前缀、随机、低深度自适应树。
2. 强 AFA：ACO；SEFA；BRiG-AFA 或经明确定义的同信息监督 risk-to-go 实现。每种方法使用相同训练样本、相同冻结响应、相同任务头预算；不能让竞争方法只选单概念而我们选便宜批量。
3. 可靠性：固定完整策略的终局校准；若主张模式条件可靠性，再按 RouteCert 类方案处理模式/策略选择。只拿 pooled ECE 不能证明 STOP 可靠。
4. 语义查询：ReCoVERR 式证据收集作为相关机制参照；若不实现它原本开放问答任务，标记为 adapted，而非声称原论文复现。
5. 控制：真概念 oracle、实际自动响应、共享编码器一次全头（预计自适应不省时）三条线同时保留。算法学习看训练标签合法；部署和测试时不能读取未查询概念或测试标签。

## 检索边界

未确认的缩写不补造文献：本轮未将 CATE 解释为一个已核验的自适应概念方法。未把 ECTS（常见早期时间序列分类语境）自动视为可自由选择下一概念。预印本存在不等于结果可靠，文中数值未独立复现；所有科学主张以本项目未来实验为准。
