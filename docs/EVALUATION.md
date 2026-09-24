# CBMJev：统计、认证与拟合谱系

这些模块计算明确的统计量、检查声明的来源链，不会自动证明 iid、临床安全、无预训练污染或真实模型优越性。只使用 Python 3.9 标准库；没有梯度、模型调用、付费 API 或数据下载。

## 1. Trace 输入契约

每个完整终局预测一行：

```json
{
  "sample_id": "sample-001",
  "group_id": "original-family-001",
  "split": "test",
  "method": "static-prefix",
  "y": 1,
  "prediction": 1,
  "queried_groups": ["food", "service"],
  "queried_atoms": ["food_positive", "service_positive"],
  "calls": 1,
  "declared_cost": 2.0,
  "mode": "offline_replay",
  "steps": []
}
```

- `queried_groups/queried_atoms` 可为非负整数计数，也可为无重复 ID 列表；明确区分组和原子属性。`calls` 为非负整数。声明成本使用同一预冻结单位，不自动等于秒或 GPU-hours。
- `y/prediction` 为非负整数类别，不接受拒答、概率、NaN 或 bool 伪装标签。每个 policy 内 `sample_id` 必须唯一；反复 rollout 不能偷偷扩大样本分母。
- `mode="live"` 必须带有限非负 `total_wall_ms`。`offline_replay` 即使带此字段也只记为 ignored，不输出部署 latency。
- 单次汇总只接受一种 method、split、mode；不能把离线回放和实测合并成一条延迟曲线。
- 所有嵌套字段（含 steps）必须可序列化为无 NaN/Inf 的 JSON。模块不验证 responder 是否真的执行，live 标记仍需运行日志和硬件证据。

## 2. 指标与不确定性

```python
from cbmjev.evaluation import summarize_traces, paired_group_bootstrap

summary = summarize_traces(traces, num_classes=5)
comparison = paired_group_bootstrap(
    adaptive_traces, static_traces, seed=17, n_resamples=10000, confidence=0.95
)
```

`summarize_traces` 返回 sample-level accuracy/error、固定 `num_classes` 的 macro F1、各类 F1、confusion matrix、sample/group 数量，以及查询组/原子属性/calls/declared_cost 的总量与均值。未出现且未被预测的类 F1 取 0；不会从预测中删掉难类或自动缩小类别集合。

另列 `group_mean_risk`：先在每个 group 内平均 sample 0–1 loss，再等权平均 group。这与逐 row error 是不同 estimand。CEBaB edits 多的家族不会在此指标中获得更大权重。group IDs 有效、互斥不代表它们随机独立。

live 才返回 `deployment_latency.p50/p95`，采用有序统计量线性插值。这里没有自动认证硬件、warmup、同步、服务负载和并发一致性；这些是 live profiling 协议的义务。不能将 p95 kernel duration、离线查缓存时间或声明成本替换为部署 p95。

### 成对 group bootstrap

两个方法必须具有完全相同 sample_id 集合，且每个 sample 的 group_id、目标 y、split 和 mode 完全一致；遗漏困难样本或不同目标映射直接报错。先计算每个 group 的平均 error/cost 差，之后使用相同重采样 group 同时计算两种差。返回差值方向为 A−B，因此负值意味着 A 更低 error/更低 cost。

这是 percentile bootstrap；10,000 次仅是计算设置，不是有限样本覆盖保证。少于 20 groups 有显式 warning；一个 group 的区间可退化，但不能解释成确定性强证据。这里比较的是 declared_cost，不是自动的真实延迟。

`summarize_seed_metrics([{"seed":17,"accuracy":...}, ...], metric="accuracy")` 独立计算训练 seed 均值与 sample SD；一个 seed 时 SD 为 `None`，不是 0。bootstrap 输出明确 `training_seed_uncertainty_included=false`，不能把测试行数当成训练 seed 数量。

### Pareto

`pareto_frontier(points, error_key="error", cost_key="declared_cost")` 保留 lower-error/lower-cost 的非支配点，至少一维严格更优才构成支配；完全相同的点都保留。输入 error 必须为 [0,1]，成本非负有限。它只筛点，不执行 validation 选点，不支持测试后调 λ；也不自动证明前沿差异显著。

## 3. 认证必须先冻结完整候选家族

```python
from cbmjev.evaluation import freeze_policy_manifest, certify_policies

manifest = freeze_policy_manifest(
    [
        {"policy_id": "lambda0", "system_hash": complete_system_sha256_0},
        {"policy_id": "lambda1", "system_hash": complete_system_sha256_1},
    ],
    frozen_at="2026-09-22T08:00:00+08:00",
)
# 先由调用方持久化 manifest 与 manifest["manifest_hash"]，再接触 certification 数据。
# system_hash 必须涵盖 R、head、policy、prompt、阈值、预算、失败处理、版本和随机部署机制。
result = certify_policies(
    calibration_traces,
    manifest,
    expected_manifest_hash=previously_retained_manifest_hash,
    alpha=0.10,
    delta=0.05,
    assumptions={
        "independent_groups": True,
        "deployment_distribution_match": True,
        "family_frozen_before_calibration": True,
        "same_rollout_mechanism": True,
    },
)
```

认证行还必须带 `system_hash` 和 `policy_id`（后者未填时仅回退 method）；`split` 严格为 `"calibration"`，不是 validation/test。每个预冻结 policy 都必须交齐相同病例与 group，不能认证时删掉“不想报告”的 policy 来缩小 M。hash 篡改、额外未注册候选、不同 case/group/target 集合或非有限数据均拒绝。

hash 只能证明给定内容的一致性，不能证明时序。如果用户在看完 calibration 后新造一个更小家族，同时替换保存的 expected hash，程序无法从字符串识别该行为。因此必须保留独立于认证工作的原始冻结产物/版本记录；四个 assumption flags 是外部责任的 attestations，不是程序推断。group-disjoint 绝不自动设置 independent_groups=true。

### 具体计算

对冻结的 M 个完整系统，每个 group 是一个假设独立的单位，其损失为该 group 内终局 0–1 loss 均值，仍属于 [0,1]。令 n 为 group 数而非 rows 数，计算：

`radius = sqrt(log(M/delta)/(2*n))`

`U_m = min(1, empirical_group_mean_risk_m + radius)`。

这是 THEORY T5 的 Hoeffding + union bound 实例。完整轨迹内无需步间独立；各策略在同一批 group 上的结果也可相关。仍要求 group 间 iid、与部署同分布、相同 rollout 机制和预冻结策略家族。

输出状态：

- `NO_CERTIFICATE`：没有任何数值上满足 `U_m≤alpha` 的候选，不能默认 all-concepts fallback 安全。
- `ASSUMPTIONS_UNVERIFIED`：数值上有通过候选，但至少一项前提未明确 attest；`certified_policy_ids=[]`、`selected_policy_id=None`。
- `CONDITIONAL_CERTIFICATE`：界通过且前提都已 attest；在认证候选中选 calibration group-mean declared_cost 最低者。这个选择不获得总体成本最优保证。

`numerically_eligible_policy_ids` 始终与真正授予的 `certified_policy_ids` 分开。比如 M=20、δ=.05、n=203 时 radius≈.12148，零经验错误也不能认证 α=.10；这是合法的样本不足结果。

保证对象是同分布总体的等权 group 平均终局损失，不是每例错误概率、病种/肤色/terminal mask 条件风险、回答子集 selective risk、分布外风险或临床安全。默认系统每次 STOP 都必须分类；不可把 abstention 当正确样本。

## 4. 拟合祖先 DAG：直接和间接监督都排除

```python
from cbmjev.provenance import make_fit_record, validate_target_exclusion

r_record = make_fit_record(
    "R_outer0", supervised_group_ids=actual_r_training_groups, parent_ids=[],
    metadata={"checkpoint_sha256": responder_checkpoint_hash},
)
f_record = make_fit_record(
    "head_outer0", supervised_group_ids=actual_head_training_groups,
    parent_ids=["R_outer0"], metadata={"checkpoint_sha256": head_checkpoint_hash},
)
audit = validate_target_exclusion(
    [r_record, f_record], artifact_ids=["head_outer0"],
    target_group_ids=actual_target_groups,
)
```

record 保存排序后的实际 group ID 列表、数量、列表 hash、全部 parent IDs、metadata 及整个 record hash。不能只存一个无法核对成员的 hash。`validate_provenance_dag` 验证内容、重复 ID、悬空引用和环；`validate_target_exclusion` 从 prediction roots 沿全部祖先搜索监督组，根本身和任意祖先有目标 group 都拒绝。

`fit_kind` 可取 supervised、derived（无直接监督）、frozen_external（目标任务监督为空）、target_construction（用 y 制作策略监督）。`make_fit_record` 记录调用者声明的已完成拟合，不负责训练，也不自动验证 checkpoint 真存在。未知的外部预训练成员不能填成“已证明不污染”；结果恒注明 `external_pretraining_membership_verified=false`。

生成 policy target 时 y 当然用于计算终局 error/loss；此时排除检查必须针对生成预测的 R/head 祖先，不能把 target artifact 本身“使用目标标签”的正当操作混同 OOF predictor 泄漏。最终 policy 使用这些 target labels 训练后，测试目标仍必须排除所有监督祖先。

## 5. Nested 3×3 只是可审查 job plan

`plan_nested_crossfit(group_ids, outer_folds=3, inner_folds=3, seed=17)` 按排序 group IDs 和显式种子产生稳定、group-disjoint 的计划：

1. 每 outer fold 内训练三个 inner responders，产生 outer-training groups 的 inner OOF 响应，供该 outer head 拟合。
2. outer responder 与 outer head 的所有监督祖先都不含该 outer-held-out group，用它们产生 policy prediction，再读取 held-out y 构造监督 target。
3. 完成计划中的 final responder、基于 outer OOF 的 final head 和基于 OOF targets 的 final policy。

本实现 3×3 计划为 13 个 responder fit jobs、4 个 head fit jobs、1 个 policy fit job、3 个 target-construction jobs。这个复用计划与“未复用时最多 16 次 responder 拟合”的上限不是同一配置，必须按实际 job graph 计账。

所有项明确 `status="PLANNED"`；输出 `has_trained_models=false`、`has_validated_actual_oof_predictions=false`。计划不能冒充 nested CV 已经训练或 OOF 缓存已经有效；真正执行后必须以实际监督 ID/checkpoint 生成 fitted provenance 再审计。

## 6. 最小检查

```bash
python3 -m unittest discover -s tests_cbmjev -p test_evaluation.py -v
```

33 项测试通过（2026-09-22），覆盖固定类别 macro F1、sample/group estimand 区别、live/replay 延迟隔离、成对集/目标严格匹配、seed SD 区分、Pareto ties、M/n 认证公式、极小 δ 数值稳定性、聚合溢出拒绝、空认证、manifest/系统 hash 拒绝、直接/祖先标签泄漏与 nested-plan 假完成拒绝。所有测试数据均为 synthetic fixtures，不能进入论文性能表。
