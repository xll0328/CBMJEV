# CBMJev 实验合同：先闭环，再论证

日期：2026-09-22，2026-09-23 更新 sprint overlay。实验矩阵见 [experiment_matrix.json](../configs/experiment_matrix.json)；状态与可运行性必须分开记录。默认训练seeds为17、18、19，数据split固定seed17，不随训练seed重抽划分。

2026-09-23 当前已有开发证据和活动作业，不再是“无真实数据结果”状态：CUB seed60 concept-budget frontier、CEBaB/ISIC boundary evidence、CUB seed61/62 live-profile cost accounting 已导入为 development evidence；CUB seed60/61/62 value-objective 三折和 seed63 robustness 分支正在服务器运行。权威状态看 [48h sprint status](../results/main/SPRINT_STATUS_20260923.md) 和 [delivery gates](../results/main/DELIVERY_GATES_20260923.md)。这些结果仍不等于最终 CVPR 主张；dynamic value-policy aggregate、强 baseline、locked test 与 citation/template audit 仍是门槛。

## 1. 主张与反主张

| 主张 | 最低可信证据 | 必须排除 | 核心块 |
|---|---|---|---|
| C1 测量误差结构和共享成本决定动态Mask适用区间 | 合成可控机制 + 真实响应 + held-out成本配置；至少两种现实测量制度检验边界 | 只重复XOR、仅人为sleep制造优势、平均质量未匹配 | B3、B5 |
| C2 动态Mask产生可归因的决策收益 | 同R/f/动作/信息，胜强静态；算法声称另需胜matched强AFA；真实live成本支持 | 更强R/head、raw-x selector、未计费预计算、给本方法更多batch能力 | B2、B3、B4 |

**反主张A：** 所谓改进只是all-predictions后的sparse Mask。**反主张B：** NanoJev只是更大的backbone而非必要接口。**反主张C：** 经验lookahead或自写DP被当成已复现published AFA。设计直接检验这三点。

## 2. 三个baseline族与统一条件

1. **静态/简单获取：** all-at-once、训练集拟合的static prefix、lazy shallow tree、random作为sanity。全量再top-k单列解释稀疏性，不隐去全量计算。
2. **自适应获取：** signed loss-drop greedy、risk controller、小K empirical lookahead；完整paper补 ACO 与 SEFA/BRiG中至少一个合适的强近邻。`empirical_dp` 仅是本项目训练经验分布上的小池参照，不叫ACO、SEFA或BRiG复现，不读取测试y选择最优路径。
3. **测量器/现代组件：** cheap multitask responder、NanoJev-style typed responder、可选视觉query-conditioned backend。raw x→y只是任务能力参考，不是信息匹配的获取policy。

同一次policy机制比较共享R、f、split、hard值、合法组集、batch候选、声明成本、最大预算与冻结规则。若某已发表方法原生不含batch/STOP，报告原版和适配版的区别；不能只向本方法提供这些动作后把全部收益归给新策略。

## 3. 五个核心块

### B1：数据、语义与信息边界

- **目的/主张：** C1/C2的前置有效性；判断问题是否可识别，不制造性能贡献。
- **数据：** 首发CEBaB；通过后CUB，Derm访问/图像adapter通过后追加。
- **设置：** 严格角色划分R-fit/head-fit/policy-fit；R用公开c，f/控制器用各自角色y；固定概念schema；完整training-ancestor追踪。
- **比较：** raw x→y能力参考、全量gold c→y诊断上限、全量自动z→y；gold缺失时不能完整case筛选偷偷简化任务，说明各比较的有效样本集合。
- **指标：** group/split重叠、missing/status分布、typed合法率、每概念macro-F1、全量任务accuracy/macro-F1、3项核心依赖不变性及metadata替换检查。
- **种子/预算：** 单seed17先跑，身份/解析测试无须浪费三遍；语义与任务基线最终3seeds。
- **成功条件：** schema和依赖测试全通过；自动概念显著有任务信号；所有有效样本和排除项可追溯。
- **失败解释：** gold c不足说明概念库/任务不匹配；gold够而z差说明R瓶颈；违规输入说明工程无效。此时不扩大controller。
- **位置：** 主文设置 + Table1能力参照；完整审计附录。**MUST-RUN**。

### B2：动态组Mask是否优于静态

- **目的/主张：** C2，拆开静态稀疏、实例适应与新增算法收益。
- **数据：** CEBaB四组先闭环；CUB提供大池扩展；Derm七组为条件医学补充。
- **比较：** all、static、tree、random sanity、risk、value、small-K empirical lookahead；paper阶段追加published strong AFA。
- **设置：** f随机采样**组级**mask，覆盖0..K；r预测一次动作后立即停止错误，V回归signed CE-drop；动作含STOP/single/frozen pair/allremaining；候选生成不看未来响应。完整系统冻结前只使用validation选择操作点。
- **指标：** task accuracy/macro-F1–mean acquired groups；另报原子属性数、requests、STOP分布、每预算点配对区间。mask entropy/path diversity仅描述行为，不作效果替代。
- **种子/预算：** seed17挑固定小配置；通过后17/18/19。相同R缓存的3个policy seeds仅支持条件于R的方差；正式涉及R稳定性时补R独立训练seeds。
- **成功条件：** 动态方法相对强static/tree有实质量级的稳定改善；若声称新算法，需matched强AFA差异。阈值由pilot后、最终test前一次冻结。
- **失败解释：** 静态已近小池参照，则该任务适应空间小；r不如V则保留V；只胜random不算研究成功。
- **位置：** Table1、[Fig2左图](../figures/fig2_dual_cost_template.svg)。**MUST-RUN**。

### B3：少查概念是否真的更快

- **目的/主张：** C1/C2，隔离非加性batch成本。
- **数据/样本：** 当前profile CLI取冻结顺序的最多100个validation cases，另报独立group数量；首个validation case每配置3次warmup、正式3次重复，不参与拟合。paper-level统计若要求100个独立groups须单独冻结组级选样，不能把`--limit 100`当作100个独立组。正式live使用预冻结样本列表与全部候选操作点，不按测试错误挑例。
- **比较：** all-at-once、single、batch2、allremaining；相同backend和精度；shared encoder cheap-head作为必要负对照。
- **设置：** R批不变性先测；不成立则cache以完整A/prompt为键；每方法同硬件/并发/缓存/输入长度制度。controller使用冻结声明成本，actual计时独立记录。
- **指标：** p50/p95端到端latency、固定并发throughput、GPU-seconds、calls/tokens、preprocess/R/policy/head/communication分解。replay时间不进入live列。
- **种子/预算：** live样本和重复计时顺序固定；warm至少两轮，cold独立。无须把所有开发λ全部live跑，先冻结至多3个代表操作点，但all-batch永不省略。
- **成功条件：** 真实成本的节约可复现，并解释何种共享开销/候选数条件下失效；只有query节约则只写query节约。
- **失败解释：** 全量共享头更便宜是合法负结果；不能人为按312次encoder调用重写其成本。8卡吞吐对单卡latency不公平。
- **位置：** [Fig2右图](../figures/fig2_dual_cost_template.svg)、cost table、Fig3经验面板。**MUST-RUN for efficiency claim**；CEBaB cheap首发可先声明无效率结论。

### B4：NanoJev必要性与简约性

- **目的/主张：** C2；现代组件有可验证作用，且不需要多余复杂度。
- **数据：** 先CEBaB，文本原生；视觉不是现成Nano文本模型的直接迁移。
- **比较：** cheap文本多任务R vs NanoJev-style R；固定R时risk vs value；MLP vs小容量更复杂controller仅当基本证据要求，首发不默认加attention。CE/Brier可作有预算的训练loss对照，不把RLCD-inspired名称当贡献。
- **公平性：** 分开“同R比较policy”和“不同R完整系统”。训练监督、参数量、backbone、输入截断、候选路径数、真实成本全部报告。若backbone不同，只能作为系统比较，不能将差异归于typed head。
- **指标：** 语义F1、类型合法率、task–actual-cost、训练与推理开销；类型合法不等于语义正确。
- **种子/预算：** cheap与Nano各seed17验证后才开另2seeds；删除无效果复杂模块，不全扫backbone×loss×head×policy。
- **成功条件：** Nano-style在固定条件下有清晰用途（质量、成本或转移），或得出无需它的明确简化结论。项目可继续叫工作名，但论文不把无证据组件写作主贡献。
- **失败解释：** cheap够用→首发保持cheap；risk额外复杂无收益→V主版；4概念下LLM无必要→不强行用热词。
- **位置：** Table2，必要时附录的matched-capacity对照。**MUST-RUN before claiming Jev/NanoJev necessity**。

### B5：机制、冻结风险与失败轨迹

- **目的/主张：** C1以及C2的可信边界；认证本身不是第三条创新。
- **设计：** 在训练/validation拟合混淆/成本诊断；比较匹配边际质量但不同误差耦合的合法合成通道，再检验真实R上的可预测关系。XOR toy只作解析检查，static-pair和adaptive-pair在该toy同样好。
- **信息审计：** poison未查询响应、固定H替换raw x/ID/timing/cache状态、集合重排、首问输入独立、gold缺失不进H。
- **认证：** 先冻结R/f/r/V、所有后处理、prompt、成本、预算、有限候选策略族和选择规则；独立group校准units上评完整终局风险。iid条件/估计对象不满足时，不称Hoeffding证书；一般exchangeability不够。STOP不是ABSTAIN。
- **指标：** 固定边际下的task-cost差异、held-out预测方向、STOP错误率和样本数、certified/not-certified及上界、已付费失败率、全样本failure taxonomy。
- **案例：** 两个不同分叉和一个错误早停按预设规则选；已有gold纠错属于研究干预，不是医生交互或临床因果结果。
- **成功条件：** 机制在真实条件中有对应并对保留配置有预测力；证书和经验风险含义准确，即使无候选通过也完整报告。
- **失败解释：** toy成立现实不成立→收窄为toy说明；校准图改善但决策不变→仅风险估计分析；跨模式样本不足→取消子群保证。
- **位置：** [Fig3](../figures/fig3_analytic_boundary.svg)、计划Fig4、理论/审计附录。**MUST-RUN for C1 and reliability statements**。

## 4. 初始网格不是全组合

首个真实可复现版本只有CEBaB cheap R，固定数据划分和一个f结构：hidden128、两个线性层（一个隐藏层）、dropout0（实现支持情况以配置验证为准）。开发比较risk/value两目标，声明单位成本的λ候选暂定 `{0,0.025,0.1}`，组预算`{1,2,4}`；最终前沿完整记录，不只报最好λ。当前参考配置单点为`cost_weight=0.03`，小网格是下一步开发计划，不声称已写入默认训练配置。

调参计数按**实际独立拟合artifact**计算，不按评估操作点重复训练：

- seed17：1个cheap R、1个f、risk/value各1个controller；9个λ×budget点是对冻结权重的评估，不是9次模型拟合。
- 同一R/f上的all/static/tree/empirical lookahead尽可能共享输入缓存，仍有自己的训练统计和artifact ID。
- pilot通过后seed18/19重复整条声明的随机链；若只能复用R，记录为policy-seed-only并收窄稳定性结论。
- Nano首轮只seed17、相同f/controller配置；先证明能训练、能输出合法响应、完整测量成本再增加seeds。
- λ在risk/BCE目标中单位是风险概率/声明cost；value用CE-drop时数值尺度不同。上述网格不可自动复用为公平超参，value应在validation上采用相同**搜索次数**的预声明尺度网格，例如按训练CE-drop尺度归一化后使用同数值。正式配置必须记录normalization。
- CEBaB只有4组，allremaining与pair可能重合；动作去重后记录实际候选数，不重复计采样量。

任何额外backbone/loss/prompt/数据扩展先增加显式job，说明它检验哪条主张，不能隐式扩大笛卡尔积。configs文件中的gated条目默认不启动生产作业。

## 5. 顺序、门槛和成本

| 阶段 | 产物 | 运行范围 | Go / No-go | 预算方式 |
|---|---|---|---|---|
| M0 本地工程 | 新包tests、synthetic smoke、schema/failure checks | CPU，不下载真实数据 | 任一关键边界失败即停 | 实测秒/分钟，只是工程 |
| M1 服务器数据+profile | CEBaB receipt、100group质量/成本与显存 | seed17、1卡或CPU cheap | 无label代理、split有效、R语义有信号 | profile后计算真实预算 |
| M2 单seed闭环 | cheap R→cache→f/r/V→validation→summary | CEBaB、≤3λ×3budget | all/static齐全；tree未实现明确pending，无未解释泄漏 | 1条依赖链，不8卡空转 |
| M3 首发复现 | seeds17/18/19、复现receipt、冻结最终评估 | CEBaB | 另一干净环境按runbook复跑；结论范围准确 | 独立seed并行，预留独占计时卡 |
| M4 科学判别 | Nano必要性、strongAFA、live、机制 | CEBaB先，再CUB | C1/C2有可信非平凡信号，否则收缩 | gate后批准额外artifact |
| M5 paper补全 | 视觉/医学、3seeds、CI、失败/证明对应 | CUB必选、Derm条件 | claim-evidence audit支持每句比较 | 成本来自profile，不预填总GPUh |

总GPU小时=`Σ(gpu_count × measured_job_seconds)/3600`；wall time受最长依赖链、I/O与卡可用率限制。8卡不等于每条串行依赖加速8倍。工程余量建议20%，超过预算则减少可选模块，不减少必要公平基线。

## 6. 论文表与机器结果要求

主表先空着：`dataset, split_hash, R_revision, f_id, policy_id, seed, budget, lambda, task_metric, group_cost, actual_latency, response_source, cost_source, CI, n_groups`。无结果为null/NOT_RUN，不写0。

每张图都能回溯原run ID；种子原值保留；CEBaB按family、Derm按case/patient、CUB按已定义独立图像/duplicate group统计。派生mask/轨迹不能充当独立样本。探索性validation图可共享用于开发，但不得改名为test结果。

首发summary必须直接显示：哪些是fixture、哪些是真实数据；哪些replay、哪些live；哪些published method verified、哪些自写reference。认证报告另列候选族大小、freeze hash、有效independent groups、失败/未认证原因；不将没有证书写成零风险。
