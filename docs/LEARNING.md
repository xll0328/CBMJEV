# CBMJev：任务头、获取控制器与训练型基线

状态：本模块是可运行的 PyTorch 学习实现，不是协议占位。CPU 测试通过；CUDA 接口已实现，但本机没有 CUDA，因此 GPU 执行尚未验证。合成测试只验证软件行为，不支持论文性能或真实成本主张。

本轮实现文件：`cbmjev/learning.py`、`cbmjev/baselines.py`、`tests_cbmjev/test_learning.py`。不修改旧 `pilot/`；旧 pilot 的逐原子 single-query 实现不是本模块的正式组级协议。

## 1. 核心接口

```python
from cbmjev.learning import fit_models, save_models, load_models

head, controller, report = fit_models(rows, schema, {
    "objective": "risk",       # or "value"
    "seed": 17,
    "device": "cpu",           # or "cuda:0"
    "hidden": 128,
    "head_epochs": 20,
    "policy_epochs": 20,
    "batch_size": 64,
    "masks_per_sample": 4,
    "max_pair_actions": 8,
})

observed = schema.empty_state()
probabilities = head.probabilities(observed)
predicted_class = head.predict(observed)

from cbmjev.contracts import candidate_actions
actions = candidate_actions(observed, schema, pairs=controller.pairs)
scores = controller.predict(observed, actions)

save_models("outputs/run17/models.pt", head, controller, report)
head, controller, report = load_models("outputs/run17/models.pt", schema, device="cpu")
```

这是 Python 接口示例，不会自动执行服务器训练。checkpoint 输出已存在时拒绝覆盖；CPU/CUDA checkpoint 都保存 CPU tensor state_dict，加载使用显式 `weights_only=True`、严格参数名和 `schema.hash` 校验。不会 pickle 整个模型对象，也不接受不同语义字典/数据集的同维度 checkpoint。

`head.predict/probabilities` 只接固定宽度的 partial observed tuple；`controller.predict` 只接这个 tuple 和待评估的组动作列表。不能把完整 row、raw x、sample/group ID、真实标签、未查询答案或实际延迟传入这些接口。

## 2. 两个严格不重叠的训练角色

输入 `rows` 是已验证缓存：

```json
{"sample_id":"case-1","group_id":"family-1","split":"head_fit","z":[1,0,2,1],"y":0}
```

此处只是格式示例，4 个值不代表某个真实数据集的已确定字典。

- `head_fit`：仅训练任务头 f。
- `policy_fit`：f 冻结后，用真实自动响应构造 action 后的目标，训练 r/V。
- `validation/calibration/test/responder_fit`：不用于本模块的模型拟合；本模块筛选训练 role 后才读其 z/y。

完整工程的数据入口仍必须审计全部 split/group/provenance。`fit_models` 另检查 head_fit 与 policy_fit 的 sample/group 无交叉；同一训练 family 可以有多行但不能跨角色。它不凭 JSON 行证明上游 R 真正冻结/样本外，也不能区分伪装成自动预测的 gold c；这些要由生成器、split manifest、模型 ancestry 和 responder revision 证明。

**不做 final refit。** 本模块生成风险目标所用的 f，就是最终部署的同一个冻结 f；不会先 held-out 造标签，随后把全量数据重拟合的另一个 f 静默替换进来。Nested cross-fitting 和 refit drift gate 仍属于后续工程，不在此实现中。

## 3. mask-aware f：查询组与原子属性分开

Schema 用 D 个原子属性存储语义、用 K 个互不重叠的组付费。比如一个 CUB 组包含多个属性，购买该组必须同时揭示所有原子属性。`mask_answers` 根据组 mask 整组展开；输入部分可见组直接报错。

对每个原子属性 j，语义编码范围为 `0..C_j-1`；`C_j` 表示已查询但 UNCERTAIN，`C_j+1` 表示已查询且 NOT_APPLICABLE；未查询为 `-1`。gold missingness 不是其中某个 runtime 值。

输入为每个可见属性的 one-hot（包含两个 runtime 状态）加 queried mask。未知值在 one-hot 前被置安全占位并乘 mask，不把 `-1` 编成额外语义类别。编码坐标固定，因此不是对 acquisition 顺序编码。

每次训练均匀采样组数 m∈{0,…,K}，再均匀采 m 个组。采样只依赖固定 RNG，不读隐藏 z/y。两层 MLP 用 CE 拟合 y；默认 hidden=128、dropout=0。没有混入某方法独占的策略 rollout masks，因此所有共享 f 的比较得到同一训练分布。

`masks_per_sample` 控制 policy target 的历史采样，不是每个 head batch 自动复制同样数量。head 每个 epoch 每条训练记录取一个组 mask；policy 每条记录生成多条状态，强制覆盖 empty（以及多 mask 时的 full）端点，其余均匀采样。

### 可选任务类别加权

统一 JSON 配置可以显式加入：

```json
{"learning": {"class_weighting": "inverse_frequency"}}
```

默认 `"none"` 保持原有训练损失及数值路径；不会自动根据数据集名称开启。不增加命令行分支，也不更改现有 ISIC 配置或正在运行的服务器链路。Python `fit_models` 的平坦配置使用同名键。

启用后只统计**原始 `head_fit` 行**的任务标签：设 C 为任务类别数，n_c 为该类行数、N 为总行数，固定 `w_c = N / (C * n_c)`。它不是按 family 等权的频率估计，不统计 mask/action 展开后的重复监督，不从 `policy_fit/validation/calibration/test/responder_fit` 借标签计算权重。任何类在 `head_fit` 中缺失就报错；不平滑、不借 held-out 类频率补齐。关闭加权时保留历史的缺类行为，但不能据此宣称缺类任务已可验收。

f 最小化 `mean_i[w_(y_i) * CE(f(H_i), y_i)]`；**不按当前 minibatch 的权重和重新归一**。因此单样本 batch 的权重仍生效，且与逐样本 loss-drop 的单位一致。该权重在 head-fit 行分布上的均值为 1，但不保证在其他 split、类别分布或 action 扩展分布上均值为 1。

value 的监督同步变为 `w_y * (CE_before - CE_after)`，始终沿用 f 的同一套冻结权重，不按 policy-fit 重新估计；STOP 仍精确为 0。risk 的事件及 BCE **不加类别权重**，仍预测实际分类错误 0/1；加权 f 本身可能改变这个错误事件。任务评估指标、static 基线的全局未加权 CE 排序和 lookahead 的未加权错误风险也不自动更改；当前不能把 static 与 weighted value 称为同一加权目标下的完全配对对照。加权不等同于概率校准，也不保证 balanced accuracy/F1 或少数类召回改善；这些须在 validation 分别检验。

`training.json`/训练 report 记录 `class_weighting`、按类别索引排列的 `head_class_counts`、`head_class_weights`、来源和 head/value/risk/static 损失定义。checkpoint 额外保存冻结的 `head_class_weights`，加载不用任何数据行重算；`weights_only=True` 仍有效。旧 checkpoint 缺加权配置/元数据时按 `none` 加载；声明加权却缺少、维度错误或非有限/非正权重时拒绝。真正部署更新应创建新 run，不改写历史模型/receipt；新增语义代码会改变 code hash，不得同步进活动实验目录。

## 4. r 与 V 的事件定义

离线 target producer 可以访问 policy_fit 的完整自动响应 z 和 y；模型 forward 不能。对于 H 和 A，先把 A 对应整组的真实自动响应加入 H，得到 H′。

| objective | target | 训练 loss | 推理输出 |
|---|---|---|---|
| `risk` | `1[argmax f(H′) != y]` | binary cross-entropy with logits | 每个动作独立 sigmoid 概率 |
| `value` | `CE(f(H),y) - CE(f(H′),y)`；启用类别加权时乘固定 `w_y` | MSE | 无界、可负的 signed loss-drop |

动作是 K 维 multi-hot 加一个 STOP flag。包括 STOP、所有合法 single group、冻结的 pairs、all remaining。**不对动作 softmax**：不同动作后的错误事件不是互斥类别，所有动作都可出错或都可成功。

STOP 的 risk 使用当前 f(H) 的真实错误；STOP 的 loss-drop 恒为 0，推理接口强制返回精确 0。风险值不是经过独立认证的错误上界；未经校准不得声称“风险受控”。

runtime 决策由外层实现：risk 比较 `r(H,A)+λκ(H,A)`；value 比较 `V(H,A)-λκ(H,A)`，无正净收益则 STOP。risk 的 λ 单位是错误概率/声明费用；value 的 λ 单位是 CE/声明费用（加权时为加权 CE/费用），不应把相同数值误称为同一业务风险权重。启用类别加权后不能直接继承原 value 的 λ 最优性，需预先约定 validation 选择规则。

两者都是一步“获取后立即分类”目标。反复调用形成 receding-horizon heuristic，不等于求解多步最优策略。可并列比较小 K 非贪心参考基线，不能把 r 自称长期 Q。

## 5. Pair 池与大数据内存

显式 `config["pairs"]` 优先，必须是已排序且不重复的组 index 对。未指定时，以固定 seed 从所有组对中选至多 `max_pair_actions=8`，完全不读未来响应或标签。这只是冻结的、响应无关的候选池，不包装为学到了概念互补性。

部署时必须使用 `controller.pairs`，使对照策略获得相同候选。若换成训练统计产生的 pair pool，应预先冻结并给全部方法共享，另记候选分布改变。

`policy_examples` 为 generator，按 minibatch 生成 H/A/H′/y，再调用冻结 f 构造目标。不建立旧 pilot 那种 N×masks×actions 的全量 dense tensor。输入缓存本身仍占 O(ND)；额外密集训练状态主要受 `batch_size` 限制。CUB 上 K 组、D 原子不能混算。

`actions_per_state=64` 是默认训练候选上限；0 表示不截断。需要截断时保留 STOP/all-remaining，再随机采其他动作，选择过程不检查候选真实回答。输出 report 记录每 epoch 产生的训练对数；这些对共享病例，不能当独立统计样本量。

## 6. Nano 风格 outcome 训练的共用目标

```python
from cbmjev.learning import iter_risk_training_examples

for item in iter_risk_training_examples(rows, head, schema, learning_config):
    # item contains only observed, action, error, split='policy_fit'
    # A separate adapter can convert this to a Nano RiskTrainingExample.
    pass
```

此公共 iterator 与 MLP risk 完全共用实际错误定义和组动作生成。每个输出不包含 y、sample ID、raw x 或未查询响应表；`error` 是独立 BCE supervision 而非模型输入。`example_epoch` 可显式改变固定种子的状态采样流。它不会先按该样本真实哪个动作最好进行 argmin 再造“医生轨迹”。

## 7. 两个训练型基线

### 全局静态贪心前缀

```python
from cbmjev.baselines import fit_static_order
order, report = fit_static_order(rows, head, schema, {"batch_size": 64})
```

只用 policy_fit，每一步对当前全局 prefix 的各候选组计算真实平均 signed CE gain，选一个统一的下一组。先对训练样本平均，再选全局动作，不是在每个测试病例上看未知真实回答的 oracle。最终 budget/停止前缀应在 validation 选择；不是 Matryoshka 论文复现。

成本复杂度约随 N×K² 增长，但按 batch 计算不一次铺开 dense 全表。`static_max_rows>0` 可用固定 seed 训练子采样控制耗时，report 明报实际使用数量；默认 0 使用全部 policy_fit。

### 小 K 经验模型非贪心前瞻

```python
from cbmjev.baselines import EmpiricalLookaheadPolicy
from cbmjev.contracts import DeclaredCost

policy = EmpiricalLookaheadPolicy.fit(
    rows, head, schema, depth=2, max_groups=7,
    include_pairs=True, include_all=True, pairs=controller.pairs,
)
action = policy.choose(
    observed, actions, remaining_budget=4.0,
    declared_cost=DeclaredCost(setup=1.0, call=0.2, per_group=1.0),
    cost_weight=0.03,
    remaining_groups=2,
)
```

基线从 policy_fit 的响应/标签表估计条件分布，不查当前测试样本的未知值。按当前 H 匹配训练行，以这些训练行诱导的响应分支估计转移；终止风险是冻结 f 在相同 H 上的训练条件经验错误。对未见 H 显式回退到全局 policy_fit 分布并保留已观察值，这种回退可能很差，要报告失败例。

每条模拟边重新调用 `declared_cost(simulated_H,A)`，因此 setup/shared-prefix 成本只按相应模拟状态支付；不会在第二步重复把初始 setup 加一遍。只在叶节点计算 terminal error，中间只累计新增费用。

费用预算和组数上限是两个独立约束。`remaining_budget=math.inf` 表示没有费用上限，允许使用，但 NaN、负无穷和负数会被拒绝；`remaining_groups` 必须由 runtime 传入尚可购买的组数（默认全部未获取组）。每条模拟边都同时扣除实际声明费用和 `len(action)` 个组，非 unit cost 下也不能靠超组数上限的未来动作制造虚假收益。内部不可行动作使用 `+inf` 排序哨兵；runtime 的 lookahead 日志 `scores={}`，不会把无穷预算或该哨兵直接写入 JSON。若新增分数日志，必须将不可行项标记为 null/状态，而不是非标准 JSON Infinity。

`depth=1/2` 是有限深度；`depth=None` 在有限训练经验 MDP、指定 budget、冻结候选集下做 exhaustive DP。**Exact 仅指这个经验模型，不是总体 Bayes 最优，更不是 ACO/BRiG 复现。** 默认 K>7 直接拒绝，超过 `max_states=20000` 也明确报错，不悄悄退成另一条弱基线。估计默认 row-weighted，不把同 family 多条轨迹当独立患者。

这提供一个可执行非贪心对照，但正式论文若要求强 AFA 近邻，仍应按实验计划补可信的 ACO/BRiG 复现或解释替代依据。

## 8. 验证与尚未验证项

2026-09-22 运行：

```bash
python3 -m unittest discover -s tests_cbmjev -p test_learning.py -v
```

类别加权增量验证：37 tests，36 passed，1 skipped（本机 CUDA 不可用）。覆盖：

- group mask 整组揭示，隐藏值 poison 不影响当前编码和候选采样；runtime 状态互异。
- validation/test 的 y/z 即使被 poison，训练权重逐位不变；policy_fit y 不能改写 f。
- BCE 目标等于真实 post-action f 错误，MSE 目标等于 signed CE drop，STOP value 精确 0。
- 独立 sigmoid 而非 action-softmax；非法/非有限输出拒绝。
- checkpoint 往返相同、weights_only 加载、schema 错配和覆盖拒绝。
- 两步经验 DP 在合成 XOR 中能看见互补收益；setup 成本按模拟历史只收一次。
- 大 K / 搜索超限拒绝；未见 H 的回退和预算/STOP 合法。
- 正无穷费用预算合法、独立剩余组数约束贯穿每条模拟转移；JSON 日志无 Infinity。
- inverse-frequency 仅用 head_fit 原始行；4:1 合成计数得到 `[0.625, 2.5]`，单样本 batch 的加权损失及梯度不抵消。
- 加权 value 前后目标一致、risk 仍为 0/1；其余 split 的标签/响应 poison 不改变训练结果，policy-fit 标签不能改变权重或 f。
- 默认/显式 none 逐位相同；本机修改前后同一合成 fixture 的 risk/value f 与控制器参数哈希及损失序列也逐项相同。这是本机回归证据，不是跨版本/跨设备的数值保证。
- 加权 checkpoint 往返、坏权重拒绝及旧 none checkpoint 兼容；加权缺类 fail closed，关闭加权保留历史行为。

本次增量同时运行 `python3 -m unittest discover -s tests_cbmjev -q`：245 tests，241 passed，4 skipped；`py_compile` 通过。所有验证均在本地完成，未同步语义代码到服务器，也未启动真实 ISIC 加权实验。

这些是工程回归测试，不是数据实验结果。未验证：GPU 训练正确性/速度、真实 CUB/CEBaB/Derm7pt 语义质量、actual-cost 改善、超参数最优性、risk 校准、总体风险保证、nested OOF、第三方强 AFA 复现。8 卡资源应优先用于独立 R/折/种子任务；这里的小 MLP 不要求或自动启用 8 卡 DDP。
