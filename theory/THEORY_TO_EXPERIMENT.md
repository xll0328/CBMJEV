# 从理论到可执行机制实验

版本：2026-09-21。所有数值曲线为解析 toy 预言或待测项，不能混作 CUB/CEBaB/Derm7pt 实验结果。正式研究仍维持两个主科学问题、五个核心实验块；下面是块内子测试，不扩成六条独立主贡献。

## 1. 最小映射

| 理论 | 落在哪个实验块 | 必须记录 | 可证伪信号 | 论文措辞 |
|---|---|---|---|---|
| T1 信息重放 | B4 信息路径 | 原输入替换、未查询字段投毒、latency 日志替换前后的动作/预测 | 同 H、同 seed、未查询前输出变化 | 架构审计通过，不等于语义/因果解释已证 |
| T2 误差敏感性 | B2 机制 | 每候选 action 后终局 loss、估计风险、估计成本、decision margin | 平均校准改善但决策无变化；边界动作误差大 | 风险估计更有决策价值，需实测而非仅 ECE |
| T3 成本条件 | B3 真成本 | setup/encode、概念计算、controller、head、通信、总延迟 residual | 查询下降但总延迟上升；简化模型 residual 大 | 在指定成本结构下有效/无效，不说通用提速 |
| T4 噪声互补 | B2/B5 机制与失败 | Bayes toy 风险、learned head 差距、joint error、λκ、single/pair/all action | 模型只会独立边际推断；忽略 pair；真数据无对应互补 | 互补 × 误差依赖 × 成本是机制；XOR 不证明自适应独特优势 |
| T5 独立认证 | B5 可靠性 | M、n、δ、α、每系统风险上界、是否空认证 | 医学小样本认证全空；shift/重复家族破坏前提 | 同分布总体有限族护栏，非临床安全 |
| T6 选后校准 | B5 可靠性 | pooled、selected-query、stop-time risk/Brier；每切片 n | pooled 好、selected 差；或无显著选择效应 | 保留/放弃 calibration 主叙事，服从结果 |
| T7 条件排序差 | B2/B5 真实机制 | 固定初问后的条件第二行动风险、同拟合数据的静态对照、回答置乱、留出集配对误差 | `G` 近零或估计误差淹没交叉；K2 CE 有益但准确率/K16 无益 | 只说该预算和损失下有局部条件信号；不等同 JEV 必要性 |

## 2. 完全不需要 GPU 的预飞测试

运行：

```bash
python theory/sanity_check.py
```

脚本输出 JSON，只是有限枚举和反例自检，不是训练、不使用真实数据、没有新的标注。它将检查：

1. 独立噪声网格上的空/单/双概念 Bayes risk 与公式相符。
2. 噪声临界式、并列边界和不可省成本示例。
3. 同边际三类 error coupling 的不同 parity 风险。
4. 合法 mask-only 预测与固定证据 replay。
5. 小型有限候选的 regret inequality。
6. 有限策略认证余量和空认证输出。
7. 总体校准而选后失准的四行以内反例。

数值枚举只能发现实现/代数错误；证明在附录，不因脚本通过而把一般命题当成形式验证。

## 3. 预注册的解析相图

### 3.1 Toy A：独立自动测量误差 × pair 成本

- 生成 `C1,C2∼iid Bern(.5)`，`Y=C1 xor C2`；固定每例 noisy Z，不允许同动作反复重抽直到答对。
- `ρ∈{0,.05,…,.5}`；`v=λκ_pair∈{0,.025,…,.5}`。这是两个轴，成本须标 normalized risk penalty，而非医疗费用。
- 主解析边界 `v=.5(1−2ρ)^2`；区域标成 `BUY PAIR / STOP / TIE`。
- 控制：single-step greedy+STOP、static-pair、two-step lookahead、pair-aware greedy、强制查询两个、oracle Bayes reference。
- 预期：精确 single-step+STOP 在严格正单项罚项下全部停止；static-pair 与 pair-aware 在其可行动作相同时风险相同。后者是必要负对照。
- 学习版任务：训练 mask-aware head 和 r/V，显示 learned boundary 与 analytic boundary 的偏离，不能假定 learned head 自动 Bayes optimal。

### 3.2 Toy B：固定边际准确率，改变错误耦合

- 固定 `ρ=.25`，取独立、同向、互斥三种 joint noise；每项 accuracy 都是 .75。
- 明确解析 pair error 依次 .375、0、.5。该例说明 error dependence 与 task function 交互，不是“correlation 总有害”。
- 扩展连续参数 `u=P(E1=E2=1)∈[0,.25]`，其余概率为 `P10=P01=.25−u,P00=.5+u`；`d=.5−2u`，该区间 pair error `d`，收益 `2u`。
- 真实数据映射：在独立审计集统计两个概念错误的共现与 task-loss 增益，只对有足够支持的 pair 报区间。不可把事后发现的最佳 pair 当预注册“普遍现象”。

### 3.3 Toy C：批量 setup × 实例查询数

- 固定 `K` 与每概念边际成本 `d`，扫描 `a/d` 和 `(q,τ)`；画 `d(K−q)−a(τ−1)−oτ`。
- `>0` 是模型内成本节约，`≤0` 是不省；标明无 empirical latency。
- 服务器 pilot 测得实际 `a,d,o` 后，将工作点叠在这张图上；若残差大，不强行拟合为直线。
- 成本不变而答案因 batch 改变时，需要分别测任务风险；纯成本图不提供 accuracy 等价保证。

## 4. 真实数据怎么承接，而不把 toy 强塞给真实世界

### CUB

在训练/验证集估计有限候选概念组的协同收益 `Δ(A)−Σ_{j∈A}Δ({j})`，只让候选生成用训练知识。测试集用于冻结候选的最终审计。可分 visibility、概念频率和 responder error strata，但不得用测试标签反向挑有利 pair 或子群。instance attributes 不是类别多数模板。

### CEBaB

四概念允许枚举 16 个子集，适合绘制 exact empirical mask-risk lattice；这仍不是 population Bayes risk。按照 original_id 家族隔离，验证食物/服务等组合是否对任务有实例相关价值。all-at-once 很可能很强，若全量批量始终最好应如实报告。

### Derm7pt

七组允许枚举 128 个子集，但完整状态包括取值。概念互补是否存在必须用诊断任务验证，不能仅复原七点计分公式。小样本不适合大量 pair/subgroup 认证；独立认证集 n 少于其所需余量时报告 no certificate，不再分裂到几十个 terminal mask 后声称可靠。

## 5. 论文图的可直接使用规格

| 草图 | 横/纵轴与图层 | 标题/图注必须出现 | 不能画什么 |
|---|---|---|---|
| Analytic panel A | x=ρ，y=normalized pair cost v，解析边界 | `Analytic noisy-XOR fixture; not empirical model performance` | 不用“ours beats baseline”图例 |
| Analytic panel B | 三种 joint error；y=pair Bayes error | `Identical marginal concept accuracy, different task risk` | 不声称该耦合比例来自真实病例 |
| Cost panel | x=setup/per-item ratio；多条固定(q,τ)边界或净节约 | `Specified cost model; operating points to be measured` | 不能把模拟单位标 ms |
| Real data frontier | 留空 accuracy–latency axes，预列 baselines | `RESULTS PENDING` | 不生成假曲线、误差条或排行 |
| Reliability panel | pooled/selected/stop 校准曲线留空，附 n 字段 | `DATA PENDING` | 不预设 selected 必然更差 |

## 6. 理论与写作的停止规则

- 如果只是 T4 toy 成功、真实数据没有可利用互补，理论只能保留为限制示例，不能支撑核心性能贡献。
- 如果真实成本落在 T3 的不利区间，转向“何时不该按需获取”的边界分析或更换真实后端；不得通过虚构单项成本改善结果。
- 如果 T6 的 selected calibration gap 小且不影响决策，校准不升级为主贡献；T5 仍可作为审计工具。
- 如果 λ 网格认证全空，先报告样本量/目标不支持认证。不能偷换为低经验错误率或使用测试集帮助通过。
- 如果强 AFA、静态 pair 和本方法已完全解释所有现象，就不能靠六条附录命题包装成原创算法论文。可以做严谨实证/系统研究，或者停止该方向。

可边实验边完成的部分：定义、方法接口、数据隔离、T1–T7 附录、成本协议、图架、related work。必须等待真实结果的部分：核心 contribution 中的效果动词、abstract 数字、性能/机制结论、临床或泛化措辞。
