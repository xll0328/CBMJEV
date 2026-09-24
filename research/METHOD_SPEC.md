> 2026-09-22 v4 工程更新：本方法合同继续有效；文中 scaffold/pilot 描述是历史实现边界。最新统一包的组查询、批动作、r/V 和 live 路径见 [代码地图](../docs/CODE_MAP.md)、[学习模块](../docs/LEARNING.md) 与 [验收记录](../docs/VERIFICATION_20260922.md)。当前训练执行互斥角色划分，完整 nested cross-fit 仍是计划，不能称已执行。

# 方法与训练规格：可实现的参考版本

状态：设计已具体化；真实模型训练尚未执行。本文优先于旧 proposal 中含糊的预算/风险定义。基本命题证明见 [理论附录](../theory/THEORY_APPENDIX.md)。

## 1. 对象、目标与可见性

统一两个粒度：**查询组** g∈[K] 是付费measurement item；**原子属性** j∈[D] 是观测的存储坐标。固定字典 Q={q₁,…,q_K} 和映射 J_g⊆[D]，默认各组不重叠且覆盖所用属性。动作 A⊆[K] 购买查询组，展开为 J(A)=∪_{g∈A}J_g。组响应 Z_g 可以是有限类别、binary、multilabel，或多个原子属性的有限tuple；不能把one-hot坐标算独立检查。

CUB的D为所用逐图binary属性数（主字典预期312），K为按词表解析的语义组数；CEBaB与Derm7pt每组就是一个categorical概念，K分别4和7，D按这一存储约定也是4和7，不是one-hot维度。预算主横轴为获取组数，同时记录原子属性数与调用数。样本为(x,c,y)，c可缺标；gold缺标mask只参与语义训练/评估。当前scaffold/pilot最小运行单位是一个atomic concept ID：只有J_g为singleton时与正式组协议直接一致；CUB正式运行必须先实现group→attributes expansion和按组扣账，不可直接用312代替K。

主回答器 Rη(x,A) 返回集合 Z_A；history-independent 表示不把先前预测作为提示词。假如 batch 会改变答案，必须把 A 和batch prompt当作测量定义的一部分，而不是假设同一个 z_j 处处可复用。

H 是按原子concept ID排序的已观察集合，并可由冻结映射恢复已获取组集合。每项包含id、合法离散值、runtime状态；组动作日志另存g及其展开成员。UNQUERIED表示不在H，不能填0代表“无”。UNCERTAIN为模型拒绝判断，NOT_APPLICABLE只能来自允许此含义的schema；MISSING_ANNOTATION绝不能从gold侧进入H。

主版本控制器信息集 I=σ(H,b_declared,Q,训练冻结统计,独立随机种子)。不能含 x、sample ID、未查询响应、gold c/y、真实tokens、实际时延、缓存状态或同批别的测试样本。任务头只看H；同H更换原输入或未观测响应必须输出不变。

部署研究目标：J(π)=E[ℓ(f(H_T),Y)+λ C_actual(π,X)]；ℓ以0–1风险和任务指标报告。优化时用训练/验证profile冻结的 κ(H,A) 近似费用，不向policy暴露样本特定实际计时。外部日志单独记录真实端到端成本。

预算b是声明费用单位，不声称逐病例严格约束真实wall clock。外部hard deadline属于另一部署制度：它会产生额外停止信号，必须单独评估，不能混入严格concept-only主实验。λ的量纲必须记录，例如error probability per normalized millisecond。

## 2. R：概念回答器

### 2.1 三个需要并列的后端

R-cheap：共享视觉/文本encoder + 全概念分类头。它是现实中很强的低成本负对照，不能人为重复encoder假装查询昂贵。

R-typed：概念问题 + 原始输入 → schema内分布/值；文本可采用NanoJev-style Qwen小模型，视觉采用已核实模态的开放模型。主实验先固定一个后端，避免R×policy×数据的全排列。

R-official：官方Jev作为文本外部效度补充；不需要付费API才能完成核心研究。API模型版本、prompt、计费和外传权限明确后才运行。

R的训练loss按概念kind采用masked CE/BCE；可比较Brier，但proper reward/RLCD风格不设为核心。按概念/问题平均以免多类别概念支配loss。语义训练后冻结，不接受y-only梯度把概念重编码为标签。

### 2.2 hard、soft 与 unknown

主通路传argmax类别或预定义unknown；概念概率只用于离线质量审计。若启用confidence进入策略，单列实验，说明它是更宽的信息瓶颈，不能和hard主版混写。

unknown阈值只在训练内验证选择；没有可见性标签时不能伪造真实unknown监督。强行把低置信度映射unknown是一个**决策规则**，不是数据集事实。对不确定输出与not-applicable分别报告比例。

## 3. f：部分证据任务头

默认固定池mask-aware MLP：每个原子属性values one-hot、queried mask、合法runtime状态编码；组mask同步展开到J_g，不允许在一次组动作中只揭示最有利的部分属性。缺失值占位不携带原数据缺标。128或256隐层、两层、dropout0或0.1，候选不超过4配置。

训练采样先均匀取子集大小 m∈{0,…,K}，再随机取m组。随后可加入固定比例（默认50%，待验证）的训练策略rollout masks。所有策略比较共享同一f和mask mixture；不能只为本方法单独训适应性更好的头。

L_f=E CE(f(H_{i,S}),y_i)。先用CE训练，评估报告accuracy/macroF1及校准；证明若要求bounded loss使用0–1或显式clipped CE并说明clip。不要把无界CE直接放入[0,1]浓缩界。

对CEBaB用group划分；对Derm7pt按case；CUB沿官方train/test，训练内切分。随机mask行不是独立样本量。

## 4. 候选生成：看不到未来答案

默认候选池：STOP、所有未查询single group、固定训练统计选出的至多8个group pair、全量remaining groups。K小的任务可枚举全部pair；不枚举CUB所有子集。下文j/k用于pair启发式时均指查询组的局部索引，而非把原子属性单独计费。

pair来源可选：训练集上概念对的平均任务交互，或由当前H估计的单项价值top-L组合。二者都只用训练信息/当前H，不读取候选真实回答。保留固定pair清单以便重现。候选构造和打分用时全部计入policy overhead。

训练pair若用interaction `g(j,k)=Δ(j,k)−Δ(j)−Δ(k)`，应直接从相同样本/相同H/同f估计；不能把相关性高当互补性高。正interaction有统计误差，选对后应在独立状态样本复核。该启发式无最优性保证，强AFA也获得同一候选池。

同一concept不同measurement mode是后续扩展，不在默认动作中；不可免费重复查询直到得到喜欢的答案。

## 5. 两种价值目标，不混淆语义

### 单步误差概率（推荐起点）

对样本外f、训练状态H和候选A，先在离线训练环境实际取得Z_A，再计算

`e_i(H,A)=1[argmax f(H∪Z_A) ≠ y_i]`。

STOP用当前H计算。同一个样本可以让多个动作都成功或都失败，因此每个(H,A)是独立事件预测问题：`rφ(H,A)=sigmoid(gφ(H,A))`，以BCE或Brier拟合e，**不能softmax成动作成功率总和1**。

选择 `A*=argmin_{feasible A including STOP} rφ(H,A)+λκ(H,A)`，其中κ(STOP)=0或明确的最终头成本；平局固定优先STOP/低成本/字典序。该r只预测“执行一次后停止”的风险，循环重规划是receding-horizon heuristic，不是已求最优长期策略。

### Loss-drop（必要简约对照）

`vφ(H,A)≈E[CE(f(H),Y)−CE(f(H∪Z_A),Y)|H,A]`。目标允许负数。选择max(v−λκ)，若非正则STOP。若此简单模型与r一样好，删去r为核心方法的包装。

## 6. 有限前瞻（仅互补失败明确时启用）

理想Bellman对象：`V*(H,b)=min{ r_stop(H), min_A [λκ(H,A)+E V*(H∪Z_A,b−κ) | H,A] }`。这里b限制剩余声明预算，terminal风险不重复相加。

用训练split的真实自动响应构造转移，不把概念真值替代噪声响应。深度d=2为默认上限；拟合Q_d时目标由冻结上一层V_{d−1}生成，seed、训练轮次和候选数与强近邻匹配。若使用0–1风险+成本，输出不再限[0,1]，不能还称单个Bernoulli概率用BCE；应分解未来终局风险与未来成本，或用实数回归Q。

离线全表用于生成训练状态合法；部署controller仍只能看已取得H。test环境离线回放可用于大规模query-risk实验，但必须标记replay，不能拿缓存命中时间作live加速。

## 7. 拟合分割与部署一致性

最低成本pilot：训练数据内R-fit/head-fit/policy-fit分离；若R完全冻结无本数据监督，可将R-fit预算并入head/policy但要记录。validation选结构/λ，calibration只做冻结策略评估，test留作最终比较。

完整监督R路线按 [数据协议](../execution/data/DATA_PROTOCOL.md) 做严格nested cross-fitting。关键不是文件名带OOF，而是给样本i生成policy target的所有有监督祖先都没看过i或同group。

部署重新全量拟合R/f会改变响应和风险目标分布。两种合法处理：

1. 保持pilot/交叉拟合部署协议原样，用相应独立holdout估计；实现简单但少数据。
2. 预定义最终重拟合流程，在独立validation/calibration上重新生成完整rollout并核验。不能直接沿用旧OOF校准证书。

### 7.1 最终重拟合的漂移闸门（实验前约定）

最终R/f与生成risk目标的fold模型不同，这是性能风险，不是OOF自动消除的问题。在**validation**上让outer-fold R/f和final R/f都对相同病例产生预测；这些模型均未用validation监督拟合。记录：(i) 每概念argmax disagreement与公开c上的macro质量；(ii) 同一固定H集合上各f的task error差；(iii) 原r预测最终f错误事件的Brier/NLL；(iv) 原冻结policy在两种部署组合下的终局error、声明成本与STOP率。这里叫held-out fold-vs-final comparison，不把两套不同样本分布的OOF统计硬相减。

工程报警门槛暂定：平均逐概念agreement<95%，或最终组合的task error比fold参考高2pp以上，或r的Brier恶化>0.02，任一触发就进入修正分支。它们是预注册工程阈值，不是理论成立条件或统计显著性阈值；正式test前可据pilot修订一次并版本冻结，不能读test后修改。

修正顺序：检查模型/预处理/类别映射；在validation重新评估完整候选族并做有限后处理；重新冻结完整系统后才进入独立calibration认证。若仍不稳定，主结果退回严格R-fit/head-fit/policy-fit分区且不做final refit的可部署版本，将nested-refit作为未通过的扩展。不能在同一已用认证集上不断修改直到通过，也不能把旧fold证书赋给final模型。不得用f_final已训练过的样本重新生成“样本外”policy targets。

跨seed区分R seed、f seed、policy seed。共享一个R缓存的三个policy seed只代表条件于该回答器的变化；主claim若涉及R稳定性，至少在关键比较重训R。

## 8. 停止风险与统计保证的边界

r(H,STOP)校准好，不自动保证自适应选择后的终局风险；需要独立完整rollout。默认只做有限候选完整策略的总体终局风险上界；如果没有策略通过上界，报告未认证，不自动放宽α。

Calib的独立单位必须是样本group，而非重复轨迹。定义group-average risk与row-average risk时明确权重；两者不是一个estimand。路径/病种/少数群体风险单列样本数和区间，不能从总体保证推导。

STOP表示提交当前预测，ABSTAIN表示不提交，两者不同。主研究先不包含拒答；需要拒答时另定义coverage与conditional risk，禁止用全部拒答获得“零风险”而不报告覆盖率。

## 9. 计时与信息路径

日志至少保存 `run_id, dataset_revision, input_hash, group_id, split, responder_revision, schema_hash, policy_hash, seed, action_ids, hard_values, statuses, declared_cost, actual_calls, tokens_in/out, response_ms, policy_ms, head_ms, total_wall_ms, cache_mode, retry_count`。group_id等只进外部日志，不能嵌入模型prompt。

GPU计时需同步或使用正确events；并行kernel各段耗时之和不等于wall time。固定同一硬件、batch、并发、输入分辨率、精度与warmup，报告单例p50/p95及吞吐，不能将8卡并行adaptive与单卡all-at-once比较。

推理按请求live调用；离线cache回放只记理论购买量。对batch-invariant R可缓存单项，batch-dependent R必须包含完整A/prompt/cachecontext。缓存不在不同split或模型revision间误复用。

## 10. 十项最低验收

1. 同H交换raw x后policy/head不变。
2. 未查询响应poison后当前action不变。
3. 同H交换sample ID/实际latency/tokens/cache hit后action不变。
4. f对H排列不变；随机种子固定时策略可重放。
5. empty history不因样本改变首个动作（除独立随机化）。
6. gold缺标不会变成runtime unknown/absence。
7. 查询不超声明预算；STOP不产生未声明responder调用。
8. live调用日志可证明未提前算全量候选；共享cheaphead例外须全计费用且单独标明。
9. train/validation/calibration/test group无交叉；OOF模型祖先可追溯。
10. 单步risk、长期Q、最终risk保证三者的定义不混用。

这些检查是实现可信性的必要条件，不足以单独证明概念语义真实性或现实因果可解释性。
