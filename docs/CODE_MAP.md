# CBMJev 开发者地图

更新：2026-09-22。本文描述当前统一 `cbmjev/` 的实际接口与产物流；旧 `scaffold/` 和 `pilot/` 保留为历史，不是新包的依赖。研究设计见 [PROJECT_BLUEPRINT](PROJECT_BLUEPRINT.md)，服务器命令见 [SERVER_RUNBOOK](SERVER_RUNBOOK.md)，本轮通过/跳过项以 [VERIFICATION_20260922](VERIFICATION_20260922.md) 为准。代码存在、fixture 测试通过、真实数据验收是三种不同状态。

## 1. 从哪里读代码

先读 `contracts → runtime → learning → pipeline`；修改特定后端再读 `responders/nanojev`。CLI 负责参数分派，研究逻辑不应复制到命令处理层。

| 模块 | 主要对象/函数 | 职责与边界 |
|---|---|---|
| [contracts.py](../cbmjev/contracts.py) | `Schema`、`ModelInput`、`DeclaredCost`、`candidate_actions`、`validate_rows` | 语义类别、组→原子映射、合法状态/动作、内容专用输入；group 不可部分揭示 |
| [data.py](../cbmjev/data.py) | `prepare_dataset` | 读取本地 CEBaB/CUB/Derm7pt 原格式，映射标签、检查 joins/重复、建立角色；不下载、不修改源数据 |
| [responders.py](../cbmjev/responders.py) | `ConceptTrainingExample`、`HashingTextResponder`、`SharedVisionResponder`、`fit_responder` | 只在 `responder_fit` 用公开概念标签训练；模型只接内容，不接病例路径/身份 |
| [nanojev.py](../cbmjev/nanojev.py) | `NanoCandidateScorer`、`NanoSemanticResponder`、`NanoRiskController` | 本地 frozen-backbone 候选打分；semantic CE 与独立 outcome BCE 分开，不冒充官方 Jev 实现 |
| [learning.py](../cbmjev/learning.py) | `fit_models`、`MaskedHead`、`ActionController`、`iter_risk_training_examples` | 组级随机 mask 的 f；样本外于 f 的一步 r/V 目标；分批生成而非铺开完整训练对 |
| [baselines.py](../cbmjev/baselines.py) | `fit_static_order`、`EmpiricalLookaheadPolicy` | policy_fit 全局贪心前缀；小 K 经验条件分布 DP，非 ACO/BRiG 复现 |
| [runtime.py](../cbmjev/runtime.py) | `load_payload`、`ReplayEnvironment`、`LiveEnvironment`、`choose_action`、`run_episode` | 两种执行环境、合法候选过滤、声明预算与真实计时分离；终局返回前不附 y |
| [pipeline.py](../cbmjev/pipeline.py) | `train_responder`、`cache_responses`、`train_models`、`evaluate_models`、`freeze_family` | 串接文件、谱系、哈希、模型与终局统计；统一 fail-closed 检查所在处 |
| [audit.py](../cbmjev/audit.py) | `audit_responses` | 公开 gold c 仅在审计侧对照自动响应，计算语义质量/错误耦合；不改模型 |
| [profiling.py](../cbmjev/profiling.py) | `profile_inputs`、`profile_backend` | validation-only 的 all/single/batch live 测量、hard-answer 一致性、重复稳定性 |
| [evaluation.py](../cbmjev/evaluation.py) | `summarize_traces`、`paired_group_bootstrap`、`freeze_policy_manifest`、`certify_policies` | 区分 row/group 风险、训练 seed 与病例不确定性、有限冻结系统的条件认证 |
| [provenance.py](../cbmjev/provenance.py) | `make_fit_record`、`validate_target_exclusion`、`plan_nested_crossfit` | 检查实际声明的监督祖先；nested 只生成计划，不执行训练 |
| [config.py](../cbmjev/config.py) / [io.py](../cbmjev/io.py) | `resolve_config`、`learning_config`、`fresh_dir`、JSON/哈希工具 | 严格配置键、非覆盖输出、无 NaN/Inf 的文件、环境记录 |
| [cli.py](../cbmjev/cli.py) / [__main__.py](../cbmjev/__main__.py) | `parser`、`dispatch`、`main` | `python3 -m cbmjev` 的统一入口，不自动使用远程 API |

详细契约分见 [DATA_ADAPTERS](DATA_ADAPTERS.md)、[RESPONDERS](RESPONDERS.md)、[LEARNING](LEARNING.md)、[EVALUATION](EVALUATION.md)。

## 2. 一条真实产物流

```text
本地公开数据 + source_revision
  └─ prepare → prepared/{samples.jsonl,schema.json,membership.jsonl,audit.json,exclusions.jsonl}
       ├─ train-responder → responder/{responder.pt,receipt.json,training.json,...}
       │    ├─ profile → profile/{timings.jsonl,warmup.jsonl,summary.json,...}
       │    └─ cache → cache/{responses.jsonl,schema.json,manifest.json,exclusions.jsonl}
       │         ├─ audit-responses → 语义质量/错误耦合审计
       │         └─ train → models/{models.pt,receipt.json,config.json,static_order.json,...}
       │              ├─ train-nano-controller → {nano_head.pt,training.json,...}（可选）
       │              ├─ evaluate → {traces.jsonl,metrics.json,settings.json,environment.json}
       │              └─ freeze → frozen/{manifest.json,settings.json}
       │                   └─ evaluate --split calibration --certification-run
       │                        └─ certify → 认证 JSON（可以合法地不获证）
       └─ plan-crossfit-prepared → 冻结prepared fold上的PLANNED job DAG
            └─ verify-crossfit-prepared → 确定性重建、哈希与规范字节核验

一个或多个 evaluate 输出 ── summarize → {metrics.csv,table.json}
```

`evaluate` 省略 `--responder` 是离线回放；提供该参数并提供 `--prepared` 才进入 live。默认 split 是 validation；test 需 `--evaluate-test`。Calibration 同时要求已冻结家族与 `--certification-run`，不能边调 λ 边当认证样本使用。

准备数据和构建离线缓存会读取 test 记录；“未评估 test”指没有用其标签拟合、选模型或报告测试指标，不是从未打开测试文件。`audit.json` 仍会明确列出未检查的许可证、源数据真实性、近重复/患者独立性等事项。

所有默认目录写入均防覆盖；失败可能保留部分输出，换新目录重跑，不把半成品补写成完整 run。单 seed 与显式多 seed 辅助脚本为 [run_cebab.sh](../scripts/run_cebab.sh)、[launch_seeds.sh](../scripts/launch_seeds.sh)；后者只调度用户列出的 seed/GPU，不消费整个规划矩阵。并行训练下的 live 时延不能直接当隔离部署 benchmark。

## 3. 六种角色与三条模型输入线

| 角色 | 允许用途 |
|---|---|
| `responder_fit` | 用已存在的公开 c 训练 R；缺概念标注只屏蔽相应监督项 |
| `head_fit` | 冻结 R 后训练 f(H) |
| `policy_fit` | 冻结 f，构造实际 post-action error / signed CE drop；训练 r/V、静态排序或经验 DP |
| `validation` | 开发、参数/策略选择、backend profile；不是最终认证集 |
| `calibration` | 完整冻结候选族的终局 rollout 与有明确前提的认证 |
| `test` | 冻结方案后的最终评估，显式开启 |

当前训练协议是 `DISJOINT_ROLES_NO_FINAL_REFIT`。R、f、policy 的受监督 group 必须按角色隔离；`validate_target_exclusion` 沿直接和间接祖先检查。它核验已声明的 provenance，不能自动证明外部预训练数据未污染。正式cross-fit入口是`plan-crossfit-prepared`，它直接消费prepared数据中冻结的outer `fold_id`，在各outer-training集合内确定性生成inner folds；随后必须运行`verify-crossfit-prepared`核验prepared输入、计划哈希和规范字节。两者仍只形成可审查DAG，尚未包含实际job executor、折训练/OOF合并和final-refit漂移验收。旧`plan-crossfit`重新随机分配样本，仅为legacy toy/scaffold入口，不得用于正式实验。

模型面对的接口只有：

```python
responder.respond(payload: ModelInput, atom_ids: tuple[int, ...]) -> tuple[int, ...]
head.probabilities(observed) -> tuple[float, ...]  # head.predict(observed) -> int
controller.predict(observed, actions) -> tuple[float, ...]
```

`ModelInput` 仅含 text 或 image bytes；图像路径在 `load_payload` 内解析、检查逃逸并转为去 EXIF 的 RGB PNG。组动作由环境展开成 atom IDs；R 返回的顺序必须与请求 atoms 对齐。f/controller 的 observed 是部分硬状态，不是完整 row。`-1` 是未查询，语义类别后的两个编码分别是 runtime UNCERTAIN/NOT_APPLICABLE；gold 缺失标注不得变成免费 runtime 信息。

Controller 的候选包括 STOP、single、冻结 pair pool、all-remaining。risk 为独立错误事件概率；value 为可负的 CE-drop，不能动作 softmax。一个 `models.pt` 只含其配置指定的 risk 或 value 控制器；分别训练的匹配 bundle 才能并列比较。经验 lookahead 的费用预算与剩余组数分别递减，即便费用无上限也不允许越过组数 cap。

## 4. 维护时最容易破坏的五道闸门

### A. `all` 必须真正一次问全

`run_episode(method="all")` 要求 fresh empty history、`include_all=True`、`max_groups=K`，且 `max_cost` 能支付首个全组动作；否则报错，不降级成“预算内最大 batch”却仍叫 all。生成 paired-against-all 比较前再次检查其恰好一次调用且 queried_groups 完整。预算受限的其它方法应与这个明确的 full-budget reference 分开解释。

### B. 模型组件身份与 bundle 文件身份不能混用

`models_sha256` 保护整个 checkpoint 文件；`component_digest` 分别哈希 f 与 controller 的命名 state_dict tensor、dtype/shape、schema，controller 还含 objective/pairs。谱系使用 `f:<head digest>` 与 `policy:<controller digest>`，而不是把混合监督的整个 bundle hash 同时当二者身份。这样只改变 policy 训练不会错误地改变 f 的监督身份。加载同时检查 bundle、组件及 `static_order.json` 的 hash。

### C. 提示/预处理代码属于 R 的身份

`responder_code_fingerprint` 保守覆盖 `contracts.py/responders.py/nanojev.py/runtime.py`。R receipt 保存 `semantic_code_hash`，加载 backend 时要求与当前代码一致；改 prompt、解码或预处理后不得沿用旧 cache receipt。另有 `code_fingerprint` 覆盖包内 Python 源文件；冻结系统 hash 还绑定模型/cache/static-order/schema、配置、method、live/replay mode、Nano 权重及失败处理。

这意味着代码变更可能主动使旧冻结家族失效：使用记录的代码版本，或重新生成、验证、冻结产物；不能删除检查“兼容”旧结果。哈希证明内容一致性，不证明冻结的真实时间顺序或统计前提。

### D. Flat cache 必须经得起 batch 语义检查

`cache_responses` 首先要求 backend 声明 batch-independent，再用至多 8 个确定性选取的 validation 病例运行 all、正向 singles、反向 singles、pairs 检查。发现答案不同就拒绝生成 flat cache。`profile` 另外测不同 batch 及重复 all 的稳定性；所有结论只限实测病例/配置，不是总体不变性证明。

当前统一 pipeline 没有 action-keyed batch-dependent 缓存实现。不能把逐项表假装成 batch 回答；有这类后端必须新增动作级缓存协议并重新统一训练/部署比较。

### E. Live 必须与冻结输入及实际购买的缓存回答一致

`LiveEnvironment` 首次查询才加载内容并验证 input digest；每次购买后，将返回的 atoms 与训练 cache 对应已购买位置比较。不一致就失败，不偷偷换回缓存值、不忽略困难病例。未查询缓存仍只在环境/审计侧，不能交给 controller。

Live 的预处理、R、policy、head、协调开销计入单例外层 wall time；模型加载不在该范围，cold session 也不等于 cold model。cheap shared responder 首次查询实际算全部 heads 并缓存，`backend_stats` 如实暴露这一成本制度，不能以“只揭示两项”声称“只计算两项”。Replay 只输出回放计时，不能生成部署 latency 结论。

## 5. 扩展接口：改一处时连带检查什么

- **新增数据集**：在 `data.py` 增加转换器，输出相同 prepared schema/membership；补标签值域、缺失、group/病例、重复和 role 交叉 tests。再显式接 CLI；不要绕过 prepared 审计直接喂模型。
- **新增 responder**：实现上述 `respond`，可选 `start_session(payload).respond(atom_ids)` 复用样本内计算，`last_stats` 仅走外部日志。补 semantic-only 训练/receipt、权重与本地依赖身份、batch consistency/profile、路径身份不变性、live acquired-answer equality 测试。
- **新增 controller**：暴露 `objective` 和 `predict(H,actions)`，只读已得概念。Nano 风格可以复用 `iter_risk_training_examples` 的实际 post-action f 错误监督；不能重新定义为自报 confidence 或每样本 oracle 最优动作。
- **改变 costs/actions**：保持 `DeclaredCost(H,A)` 仅依语义 history 与冻结参数；环境还要执行组数与声明费用双预算。改 pair pool、measurement mode 或 STOP 语义后，重新训练/验证并冻结完整系统。
- **新增结果指标**：从 traces 计算并保留 estimand、split/mode、独立 group 数；不要在 plotting 层手填结果。样本 bootstrap 与 seed SD 分开；认证必须保留全部预冻结候选。

## 6. 最小开发验证入口

在项目根运行：

```bash
python3 -m cbmjev --help
python3 -m unittest discover -s tests_cbmjev -v
python3 -m cbmjev smoke --out /absolute/path/to/new-smoke-directory --seed 17
```

最后一条会创建本地合成数据并真实训练小模型，不下载真实数据；输出目录必须是新的。它默认不评估 test，不能把合成指标放入论文主表。

| 修改区域 | 优先回归文件 |
|---|---|
| Schema、组预算、信息隔离、strict all、live equality | [test_contracts_runtime.py](../tests_cbmjev/test_contracts_runtime.py) |
| 数据 adapter、缺失/确定性、病例/重复 | [test_data.py](../tests_cbmjev/test_data.py) |
| R、Nano prompts/候选/权重、本地 backbone 身份 | [test_responders.py](../tests_cbmjev/test_responders.py) |
| f/r/V、held-out poisoning、经验 DP 双预算 | [test_learning.py](../tests_cbmjev/test_learning.py) |
| 端到端依赖、component hashes、flat-cache guard、冻结/测试开关 | [test_pipeline.py](../tests_cbmjev/test_pipeline.py) |
| 语义质量/错误耦合审计 | [test_audit.py](../tests_cbmjev/test_audit.py) |
| live batch、重复稳定性、流式 profile | [test_profiling.py](../tests_cbmjev/test_profiling.py) |
| group 统计、认证、祖先排除、legacy toy crossfit | [test_evaluation.py](../tests_cbmjev/test_evaluation.py) |
| prepared-fold crossfit计划、held-out隔离、确定性重建与防篡改 | [test_crossfit.py](../tests_cbmjev/test_crossfit.py) |

本页不固定复制易漂移的总测试数量。CUDA、真实 HF backbone 和真实数据是否运行，必须查验收报告中的独立状态，不由 mock/CPU tests 推断。

## 7. 当前仍待补齐

真实官方数据包的服务器验收与三种子实验；真实 CUDA/HF/视觉后端测量；论文级 ACO/BRiG 等强 AFA 复现；batch-dependent action-keyed cache；nested cross-fit 执行器与 refit gate；并发服务吞吐 benchmark。现有有限家族认证有 fail-closed 实现，但是否获得证书取决于数据与外部前提，不是“命令执行成功”即可宣布安全。

执行顺序与扩展闸门继续以 [EXPERIMENTS](EXPERIMENTS.md)、[RELEASE_CHECKLIST](RELEASE_CHECKLIST.md) 为准；规划矩阵不是自动批准的无限实验队列。
