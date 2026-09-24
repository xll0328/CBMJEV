# Project CBMJev：从动态概念 Mask 到可复现的测量研究

日期：2026-09-22。工作项目名：`CBMJev`；它不是已经确立的新方法名、官方 Jev 合作项目或论文结果。数据只使用既有公开标注，新增专家概念/轨迹标注为 0。可用资源为 8×A100 或 8×4090，实际显存、互联与时段由服务器 preflight 确认。

本文件管理本轮工程边界。更细的方法定义见 [METHOD_SPEC](../research/METHOD_SPEC.md)，优先权与近邻见 [novelty decision](../research/literature/NOVELTY_DECISION.md)，旧研究历史保留，不重写。

## 1. 项目一句话与最终问题

> 构建一个只根据已获得语义证据生成动态概念 Mask 的 CBM：它决定下一步查询哪些概念组，以及何时停止；同时检验这种顺序决策在有误差的自动测量和真实批处理成本下，究竟何时值得使用。

研究单位不是“会变化的二进制向量”本身，而是**该 Mask 由哪些信息决定、它对应的概念是否真的按需计算、它改善了什么可测量的目标**。

Mask 可以成为用户理解和开源接口的中心，但不能代替 AFA/CBM 的已有方法定位。不得宣称首次 sequential CBM、首次无需专家轨迹、首次 noisy acquisition 或首次停止后校准。早期发布可以提供可复用的定义和协议；它不保证引用量、接收或相对近邻的原创性。

## 2. 四种 Mask 不是同一研究设置

| 名称 | 产生 Mask 时能看到什么 | 是否真正按需测量 | 本项目角色 |
|---|---|---|---|
| 固定 Mask / 全局前缀 | 训练统计 | 可以 | 必须比较的强静态基线 |
| 输入条件 Mask | x 或其 latent | 可能 | 额外输入能力的对照，不符合严格 concept-only 主线 |
| 全量预测后 top-k Mask | 所有预测概念 | 否，已支付全量感知 | 稀疏解释参照；不能据此宣称推理节省 |
| 已观察证据条件的序贯 Mask | 当前已查询的值、状态、组身份，以及声明预算 | 可以 | 主研究对象 |

同样画出 `m⊙z`，以上四种实现的信息与成本完全不同。介绍项目时，应优先展示依赖边界图而不是一张没有上下游的 Mask 热图。[已有图册](../figures/gallery.html)

### 2.1 精确状态

设查询组为 `g=1,…,K`，存储的原子概念为 `j=1,…,D`，冻结映射为 `J_g`。`m_t∈{0,1}^K` 表示哪些查询组已经获取，购买动作 `A_t` 后 `m_{t+1}=m_t∨1_{A_t}`。组内原子属性同时揭示，不能从中挑最有利的一部分。

已观察历史 `H_t` 包含已购买组展开后的概念 ID、离散值与合法 runtime 状态。任务头读取 H 或等价的 `(masked values, expanded mask, statuses)`；控制器读取 H、冻结候选描述与声明预算。所有未查询原子值必须在编码前屏蔽，不能先做全量数据依赖归一化再 mask。

- `m_g=0`：尚未查询；不等于否、不存在、不适用或标注缺失。
- 查询后模型表示无法判断：`m_g=1`，runtime 状态为 `UNCERTAIN`，费用仍已产生。
- CEBaB 的 categorical `unknown` 是合法语义类别，与缺标不同。
- `MISSING_ANNOTATION`、`NO_MAJORITY`、原始可见性/标注确定性只进监督/审计侧，不作为控制器的免费特征。
- 初始 `m_0=0` 时，相同预算下的首个动作应相同，或由与样本独立的随机数决定。若先读图像 latent 决定第一问，必须标为另一制度。
- Mask 随合法已得证据而携带标签相关信息是正常现象；mask-only 能预测 y 不是单独的泄漏证据。

### 2.2 不能暗中加入的输入

policy/head 禁止读取 x、视觉/文本 latent、原始 sample/group ID、未查询缓存、gold c/y、实际 token 数、逐样本时延、缓存命中、测试批内其它样本。模型输入白名单与外部日志字段分开。

优化使用 train/validation profile 后冻结的 `κ_declared(H,A)`；实际成本由 evaluator 记录，不能通过预算扣减偷偷进入 policy。硬 wall-clock deadline 会额外产生输入相关停止信号，是可选部署扩展，不混入严格主实验。

## 3. NanoJev 到底在哪个位置必要

三个模块：`R(x,A) → typed observations`，`f(H) → task distribution`，`r(H,A)` 或 `V(H,A) → action score`。只有 R 读原输入。NanoJev 首先是 R 的开放设计参考，而非整个系统都必须使用大语言模型。

| 假设 | 怎样公平测试 | 失败后的项目决定 |
|---|---|---|
| typed question 接口对语义回答有帮助 | 同 public c、同数据、同训练预算比较 cheap multitask responder 与 NanoJev-style responder；分开语义准确率和类型合法率 | 保留更简单 R，不以品牌作为贡献 |
| 非生成式候选打分比生成文字更合适 | 控制 backbone、候选集合、长度及真实成本；实现匹配后才比较 | 若优势不能与模型大小/额外监督分离，降为实现选项 |
| 文字概念定义帮助 controller 跨概念迁移 | 在真正 held-out ontology/task 上比较 ID-MLP 与文本描述编码，统一合法输入与动作 | 没有迁移评估则不声称语义可迁移，首发不实现该扩展 |

首发使用 cheap 文本 responder 把 CEBaB 全链路跑通；它是工程基线，不是“证明 NanoJev 已经有用”。当前实现候选是哈希词ID + EmbeddingBag + 多概念头，不需预下载语言模型。源码中的 `CheapSession` 第一次查询会计算全部概念头并缓存，只向环境揭示所请求值；日志必须记 `SHARED_CHEAP_ALL_HEADS` 和实际 `atoms_computed`，不能称感知计算随Mask变稀疏。可选 NanoJev adapter 必须固定来源、模型/tokenizer revision、完整问题损失与数据转换规则；本地独立实现写清 `NanoJev-style`，不冒充原仓库精确复现。NanoJev 文本 backbone 不直接处理 CUB/Derm7pt 图像；视觉路线是另一个已验收的 responder。

不要把各动作的成功事件 softmax 成和为 1 的概率：多个动作都可能成功或失败。risk head 预测“执行该动作后立即分类是否错误”；value head 回归 signed loss-drop，允许额外概念使有限容量分类头变差。两者都是必须比较的透明起点，不自动组成新算法。

## 4. 只保留两条研究主张

**C1：测量过程的收益区间。** 在指定真实后端与公开任务上，任务相关测量误差结构和共享/批成本解释动态 Mask 的获益或失效区间；仅平均概念准确率或 mask sparsity 不足以判断收益。需要真实测量与可预测的 held-out 边界，不能只有 XOR toy。

**C2：可归因的决策收益。** 同 R、f、动作集、预算定义与信息约束下，基于真实自动响应的动态组获取能否超过强静态和强 AFA，并在实际端到端成本上保持收益。仅超过固定顺序支持实例适应性，不足以证明新控制器优于已有 AFA。

两条均为待检验假设。若 C1 成立但 C2 不胜 AFA，转为测量机制/评估论文；若两者均弱，保留开源协议与负结果，不继续用更大模型包装。理论中的 no-bypass、argmin 稳定性、成本记账、XOR 与有限策略风险界均按真实归属使用，不能重新命名为原创定理。[理论附录](../theory/THEORY_APPENDIX.md)

## 5. 三层交付：不要把代码首发等同完整论文

| 层次 | 可交付内容 | 通过条件 | 不能声称 |
|---|---|---|---|
| L0：工程可运行 | 统一包、fixture、CLI、schema、组查询环境、head/controller、replay、日志、tests | 干净环境 smoke + 单元测试 + lineage/信息边界检查 | 真实数据有效或节省计算 |
| L1：CEBaB 可复现首发 | 官方数据适配、严格角色拆分、cheap R训练、缓存、f/r/V、matched简单基线、validation闭环、显式冻结测试流程 | 服务器真实数据3 seeds、完整receipt与运行配置、独立重跑、成本/模式准确标记 | 大概念池、视觉/临床泛化、NanoJev必要性、强AFA领先 |
| L2：paper-ready研究 | CUB视觉、Derm条件验证、真实Nano/按需R、strong AFA、live测量、机制与统计、理论/实验对应 | 五块证据完成、claim audit通过、真实live比较、全部结果可追溯 | 超出数据与假设的通用最优/临床安全 |

L1 可以先公开，但应清楚标注 research prototype、真实验证范围和剩余基线。是否公开、创建仓库、发布release属于另一步明确动作，本文件不自动执行外部发布。

## 6. 工程状态：已存在与待验收

状态区分“代码存在”“本机fixture路径运行”“真实数据/服务器验收”。本轮已实际运行CEBaB合成fixture的R→cache→f/r→validation replay/live→CSV链；它不支持任何真实任务收益主张。完整回归结果以项目验收记录为准，不在本表复制易漂移的测试数量。

| 能力 | 当前可核实状态 | 新统一包的验收证据 |
|---|---|---|
| 研究、数据与理论协议 | `DESIGN_AVAILABLE`：此前文档/证明/审查已存在 | 本轮协议与接口一致性审查 |
| 协议脚手架 | `LEGACY_AVAILABLE`：独立 scaffold；不等于新包已通过 | 新包 equivalent tests |
| 简单学习 pilot | `LEGACY_AVAILABLE`：atomic单问/STOP离线学习 | 组展开、batch、r/V、live/cert新测试 |
| 统一 `cbmjev` CLI/包 | `IMPLEMENTED_LOCAL_SMOKE_VERIFIED` | 14命令help；本机CPU fixture依赖链；服务器环境仍pending |
| CEBaB/CUB/Derm适配 | `IMPLEMENTED_REAL_DATA_PENDING` | 三个adapter及fixture tests存在；真实官方原包验收pending |
| cheap text R | `IMPLEMENTED_LOCAL_SMOKE_VERIFIED` | 实际自动语义训练/cache/live；真实CEBaB质量pending |
| 可选视觉/Nano | `IMPLEMENTED_REAL_BACKEND_PENDING` | 可选adapter/训练接口存在；本轮未跑真实backbone/GPU |
| 组级 f/r/V与经验lookahead | `IMPLEMENTED_REAL_DATA_PENDING` | risk/value训练与checkpoint tests；不是ACO复现；真实比较pending |
| replay/live/summary | `IMPLEMENTED_LOCAL_SMOKE_VERIFIED` | CPU fixture双模式完整轨迹与逐run CSV；无真实速度结论 |
| 冻结cert / nested cross-fit | `IMPLEMENTED / PLAN_ONLY` | cert有失败关闭与哈希检查；cross-fit仅DAG，未执行nested fits |
| seed调度 / 全matrix执行 | `EXPLICIT_HELPERS / DISABLED` | 单seed和显式seed列表；JSON永久planning-only，不自动扩展 |
| 真实GPU训练与结果 | `NOT_RUN` | 服务器run目录与指标；本机不下载/训练真实数据 |
| ACO/SEFA/BRiG论文级基线 | `PENDING` | 原方法/适配声明、实现核查、统一条件运行 |

实现是否完成以 [RELEASE_CHECKLIST](RELEASE_CHECKLIST.md) 与 [本轮验证报告](VERIFICATION_20260922.md) 的证据为准，不以接口文件存在为准。运行参数见 [SERVER_RUNBOOK](SERVER_RUNBOOK.md)，学习实现限制见 [LEARNING](LEARNING.md)。strong AFA复现、nested cross-fit dispatcher和并发吞吐benchmark均未实现。

## 7. 默认工程边界

首次闭环：CEBaB，四个categorical组，固定 `train_exclusive`，5类总体评价，cheap text R；train内 `R-fit/head-fit/policy-fit` 严格隔离，不做 final refit。validation用于开发与temperature等可训练后处理；calibration只认证完整冻结候选族；test需显式开关，默认不评估。

每个artifact保存数据/schema/split/model/config哈希、fit-group集合摘要和所有监督训练祖先。生成policy target的样本不得出现在R或f的任何监督祖先，包括early stopping。完整 nested cross-fit 是后续增强，不能仅把文件命名OOF就绕过祖先检查。

从 CEBaB 扩到 CUB/Derm前，必须验证图像只在R内部使用、组动作展开与计费一致、病例/重复图分组、缺失标注语义正确。CUB的312原子属性不自动等于312次查询；组数从官方词表推导。Derm临床/皮肤镜图属于同case，不能跨split。

## 8. 风险登记与停损

| 风险 | 触发信号 | 应对 / stop条件 |
|---|---|---|
| Mask只是事后筛选 | 获取前已有全量R forward | 计入全量成本，重命名sparse explanation；不能进入lazy提速表 |
| 信息旁路 | 同H替换raw x/ID/timing后动作变 | 阻断该字段、补回归；未修复前不训真实规模 |
| 训练祖先泄漏 | policy-target group在任何supervised parent fit中 | 重建角色/OOF；旧结果不可用于主表 |
| 缺标变免费信息 | gold missing/visible控制合法动作或runtime status | gold/runtime双sidecar，拒绝自动填补标签 |
| 单例与批量回答不一致 | 相同问题不同batch改变z | 将完整batch入cache key，重新建立对照 |
| 真实cost无空间 | All-batch与最小query同价或更快 | 保留负对照；不人为拆慢共享encoder |
| Nano只是装饰 | cheapR相当且更便宜 | 首发保持cheap；没有必要性证据不升级方法名 |
| 强基线缺失 | 只有random/fixed或自己的empiricalDP | 标pilot；完整paper前补published AFA |
| 小数据认证无力 | 独立group少、没有候选通过上界 | 报not certified；不放松α或复用认证集调参 |
| 8卡I/O或显存失配 | 4090跨卡通信/OOM、共享盘瓶颈 | 单卡replica起步、profile后调度，不承诺线性加速 |
| 结论被单个漂亮案例驱动 | 只展示成功轨迹，无总体统计 | 固定抽样规则、失败轨迹和总失败率并报 |

## 9. 可以有的后续升级，但首发不承诺

概念描述编码、attention over acquired evidence、held-out concept pool transfer、同概念多测量模式与纠错后重规划都具有具体研究问题。进入条件是当前实验暴露了ID-only控制器、单源测量或冻结路径的明确瓶颈；最多启用一条。控制器即便使用语言模型也只看H，不得重新给它原图或完整评论。

详细运行顺序见 [EXPERIMENTS](EXPERIMENTS.md)，服务器步骤见 [SERVER_RUNBOOK](SERVER_RUNBOOK.md)。历史 [论文草稿](../paper/STORY_AND_DRAFT.md) 和 [条件结论](../paper/RESULT_BRANCHES.md) 可以继续复用，但摘要/贡献的比较措辞必须等真实证据。
