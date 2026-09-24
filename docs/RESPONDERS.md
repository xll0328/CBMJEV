# 响应器与 NanoJev-style 候选评分接口

本模块实现真实可训练的语义回答器与候选风险头，不是只有 mock；但没有执行真实数据论文实验。廉价文本头的合成拟合、可选视觉前向及 fake-backbone Nano 接口测试仅是工程验收。没有下载权重、没有调用付费/外部推理 API、没有安装依赖。

## 来源与实现边界

参考项目：[TianyuCodings/NanoJev，锁定 commit 76fdfc9ecdca45a9bcef17991a07d3041a87685a](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9ecdca45a9bcef17991a07d3041a87685a)。2026-09-22 尝试 GitHub REST commits/main、commits?per_page=1、git/ref/heads/main 均被匿名 rate limit 拒绝；随后以只读 `git ls-remote ... refs/heads/main` 核验 SHA，并读取该 SHA 对应原始 README。这里**不声称 GitHub API 核验成功**。

锁定版本的[官方仓库 README](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/README.en.md)描述了候选路径评分、Choice 的 set attention/softmax 和 Boolean sigmoid。本项目仅采用“本地 backbone + 候选 scalar head”的简约适配思路，**没有复现其 set-attention、全部游戏训练、服务协议、kernel 或完整检查点格式**，也不是 Typesafe 官方 Jev 或官方 RLCD。其候选 Choice 分布与我们的独立错误事件风险不是同一目标。

## 内容边界

统一复用 `cbmjev.contracts.ModelInput`：

```python
ModelInput(text="原始评论正文")
ModelInput(images=(image_bytes,))
```

两者二选一。公开概念描述/可选语义值由 `Schema` 持有，不从病例元数据读取。`sanitize_model_input` 拒绝额外字段、路径、sample_id、split、y、gold，返回同一统一契约类型；调用方必须先显式投影允许的文本或读取后的图片 bytes。元数据与原始文件读取归外部编排器，不传到模型。该检查不能自动判断一段原始文本中是否本身有目标泄漏，数据准备仍须审计。

## Cheap：共享文本/视觉编码器与全部概念头

`HashingTextResponder(schema, vocab_size=2048, embedding_dim=64, threshold=None)` 使用稳定 SHA256 token hashing、EmbeddingBag(mean) 和每个概念的线性分类头。它不是预训练语言模型；从公开概念标注学语义，不接任务标签梯度。每次前向共享一次编码，随后计算全概念头。

```python
from cbmjev.responders import (
    ConceptTrainingExample, fit_text_responder, save_responder, load_responder,
)

examples = [ConceptTrainingExample(payload, concepts=(1, None, 0), split="responder_fit")]
model, report = fit_text_responder(
    examples, schema, epochs=30, batch_size=32, learning_rate=.003,
    vocab_size=2048, embedding_dim=64, seed=7, device="cpu",
)
save_responder(model, "new_dir/responder.pt")
model = load_responder("new_dir/responder.pt", schema, device="cpu")
```

上例概念数必须与实际 schema 对齐。`None` 仅表示缺失 gold，不变成运行时 unknown；CE 只对有标注概念计算，每批先逐概念平均，再在有监督概念间平均。若某概念完全无监督，报告其计数为 0，不能把该头视为已学会。训练接口拒绝非 responder_fit；没有 y 字段。`examples` 可为 lazy Sequence，标签校验/计数只遍历一次，训练按 batch 取样；避免按概念数反复加载全部图片。

### 新增：`hf_text` 预训练文本概念响应器

这是可训练的 encoder-only Transformer + 每概念分类头，不是 Nano 候选评分器，也不是预训练任务分类器。`HFTextResponder(schema, encoder, tokenizer, max_length=512, freeze_backbone=False)` 支持注入模型做无外部依赖的契约测试；生产入口为 `load_hf_text_responder(schema, model_name_or_path, revision=None, allow_download=False, ...)`。实际 HF 模型训练仍需目标环境验证，fake 测试不构成预训练模型性能证据。

模型来源没有默认值：`--kind hf_text` 必须同时提供 `--backbone`。默认只读取本地目录或已缓存的明确 Hub ID；只有显式 `--allow-download` 才允许请求该模型的 Hub 权重。所有加载均关闭 `trust_remote_code`，使用 eager attention；不安装依赖、不回退至 toy。HF 的 `from_pretrained` 支持本地/Hub 来源、revision 和离线参数，参见[官方加载文档](https://huggingface.co/docs/transformers/main_classes/model)。本实现先解析 encoder 的实际 commit，再用同一 commit 加载 tokenizer，避免两次解析可移动 tag；无可验证 commit 时要求显式完整 SHA。原数据 `source_revision` 与模型 `initialization.requested_revision/resolved_revision` 分开记录。

以下路径是占位示例，须换成已有目录：

```bash
env CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python3 -m cbmjev train-responder \
  --kind hf_text --prepared /srv/data/cebab_prepared \
  --backbone /srv/models/text_encoder --out /srv/runs/cebab_hf_seed17 \
  --device cuda:0 --seed 17 --epochs 3 --batch-size 16 --learning-rate 3e-5
```

显式 Hub 初始化还需 `--backbone ORG/ENCODER --revision FULL_COMMIT_SHA --allow-download`；这里的名称和 SHA 都是占位符，不内置任何远端模型。学习率未指定时，`hf_text` 使用 `3e-5`，其他原有 kind 仍为 `0.003`。冻结特征实验加 `--freeze-backbone`；默认联合更新 encoder 和概念头。冻结时参数不求梯度且 encoder 保持 eval，避免外层 `train()` 重新启用 dropout。共同训练器使用 Adam、逐概念缺标 mask CE、seed 与严格确定性设置，报告实际参数、设备和环境；不使用任务 `y`、head/policy/validation/calibration/test 的概念监督。

预处理固定右 padding，以 attention mask 加权平均最后一层 hidden states；padding 不进入平均，special tokens 仍是编码序列的一部分。`--max-length` 默认 512，包含 special tokens；超过长度会在 encoder 前拒绝，**不静默截断**。因此首次正式运行前应审计文本 token 长度，并显式选择与 encoder 容量兼容的长度。无 pad token、decoder/encoder-decoder、全 padding mask 或不支持的输入字段均明确报错。

仍采用共享一次编码、全部概念头的会话模型：首次非空请求 `atoms_computed=D`，标记 `SHARED_HF_TEXT_ALL_HEADS`；后续请求只读本会话内部缓存。这允许比较语义质量，但**不支持“少问概念就省下对应 Transformer 前向”的结论**。策略/任务头只看到实际取得的回答，不能访问模型内部完整头结果。

`save_responder` 保留原有 `cbmjev-semantic-responder-v1` 容器，`hf_text` 保存完整 encoder + 所有概念头 state_dict，而非仅保存 heads。同时生成 `responder_hf_assets/`，由 encoder config 与 `tokenizer.save_pretrained()` 的完整本地资产构成；每个文件的 SHA256 清单写入 `responder.pt`，因此原有 R checkpoint 身份也绑定 tokenizer/config。重载先验证清单，然后只用本地 `AutoConfig`、`AutoTokenizer` 和 `AutoModel.from_config()` 重建并严格加载全部权重；不访问原始模型目录、Hub 或原始 revision。可整体移动 artifact，不能只移动 `.pt`；资产删改或符号链接、缺失权重都会拒绝。

receipt 的 `initialization` 记录模型来源、请求/解析 revision、初始化 encoder 权重指纹、冻结开关及随机概念头来源；训练后的完整权重身份仍由 checkpoint SHA256 给出。预训练数据是否与评估集重叠保持 `UNKNOWN_NOT_CERTIFIED`。配置和分词器内容可复核不等于已验证预训练数据独立性。

新增无 transformers 测试覆盖：masked pooling 的单例/批处理一致性、全量/冻结训练梯度、超长拒绝、明确来源和 tokenizer revision 绑定、完整 checkpoint 离线移动重载、资产修改拒绝，以及 pipeline 中 poison 全部任务 `y` 和 held-out 输入/概念后响应器权重不变。新路径没有下载真实权重，亦未运行真实 HF CUDA 实验。

已安装 Transformers 的目标环境还应先运行下面的真实库准入测试：它临时生成微型随机 BERT 和词表，执行概念训练、完整保存、移走原来源、离线重载与输出一致性检查，**不下载模型，也不是预训练性能实验**。未安装 Transformers 时明确 skip。

```bash
python3 -m unittest tests_cbmjev.test_responders.ResponderTests.test_optional_real_transformers_tiny_local_roundtrip_without_download -v
```

会话接口（主程序应使用）：

```python
session = model.start_session(payload)
values = session.respond((0, 2))  # 原子索引，不是组索引；顺序与请求一致
stats = session.last_stats
more = session.respond((1,))
```

首次非空请求共享编码、计算**全部**概念头，返回只请求的值。`last_stats` 明确记录 `encoder_forwards=1`、`atoms_computed=D`、图片视图数，后续仅取已付费结果。外部必须计入首次全量前向的真实耗时，并标记 `SHARED_CHEAP_ALL_HEADS`；不能把后续缓存命中当作便宜独立测量、更不能宣称本后端因少问几个概念就省编码时间。每个病例新建会话，禁止跨病例复用。直接 `model.respond(payload, atom_ids)` 每次创建新会话，适合单次调用，序贯执行用它会重复编码。

`threshold` 若启用，把低最大概率映射为 `UNCERTAIN=C`；这是验证集冻结的决策规则，不是 gold 缺标推断。默认只返回语义 argmax；不自动生成 NOT_APPLICABLE。空原子请求不触发前向。

### 可选视觉

`SharedVisionResponder(schema, image_size=224, backbone_checkpoint=None, threshold=None)` 延迟导入 torchvision；ResNet18 永远 `weights=None`，不会联网下载。可以提供本地原始 ResNet18 `state_dict` 初始化，随后去掉 fc 并添加语义头；无 checkpoint 就明确是随机初始化，不冒充预训练视觉模型。

输入是一个或多个图片 bytes；RGB、固定 resize、ImageNet normalization；同一病例多视图在一次 encoder batch 前向后均值池化，再接全部头。所有视图计算均须计费。`fit_responder(model, examples, epochs=..., batch_size=..., learning_rate=..., seed=..., device=...)` 可训练视觉模型；随机初始化应由调用方在构建模型前固定 torch seed。32×32 图片进行训练时不要给 ResNet BatchNorm 单样本末批；正式默认 224。本轮只验证本地多图前向，未进行真实视觉拟合或 GPU 验收。

## NanoJev-style：可选本地 HF backbone

`load_local_nano(local_directory, device="cpu", max_length=512, freeze_backbone=True)` 延迟导入 transformers，`local_files_only=True`、`trust_remote_code=False`，路径必须已存在。没有 transformers 或本地权重时明确报错，**不 silent fallback 到 toy 模型或自动下载**。默认加载 AutoModel，不调用生成。

`NanoCandidateScorer(backbone, tokenizer, hidden_size=None, max_length=512, freeze_backbone=True)` 也支持依赖注入，供单测使用。每个完整候选路径产生一个 scalar logit；批内不同长度支持左/右 padding：pooling 使用最后一个有效位置，而不是 `sum(mask)-1`，position_ids 由 attention mask 累积生成。超长 prompt 报错，不悄悄截断历史或候选。当前每条候选路径独立编码，**无共享前缀优化**；batch forward 不代表前缀只算一次。

### 语义值训练与按请求测量

`fit_nano_semantics(scorer, examples, schema, epochs=5, learning_rate=.001, seed=0)` 接收同一 `ConceptTrainingExample`。对于每个已标注概念，所有语义值构成互斥候选集合，使用 CE；不同概念的 CE 平均，不以任务标签学习概念。

`NanoSemanticResponder(scorer, schema, threshold=None)` 提供相同 `start_session/respond` 接口，但仅为请求原子构造值候选并前向；不提前算未查询概念。`last_stats` 除原子数还记 `candidate_paths`，不把一个批次当成一条 token 路径。该适配仅文本，不暗示 Qwen 文本 backbone 能读取图片。

### 概念历史上的独立动作风险

```python
from cbmjev.nanojev import (
    RiskTrainingExample, fit_nano_risk, NanoRiskController,
    load_local_nano, save_nano_head, load_nano_head,
)

scorer = load_local_nano("/absolute/local/Qwen-directory", freeze_backbone=True)
events = [RiskTrainingExample(observed=(-1, -1), action=(0,), error=1., split="policy_fit")]
scorer, report = fit_nano_risk(
    scorer, events, schema, epochs=5, batch_size=16, learning_rate=.001, seed=7,
)
save_nano_head(scorer, "new_dir/nano_risk.pt", schema, task="risk")
scorer = load_nano_head("new_dir/nano_risk.pt", schema, expected_task="risk")
controller = NanoRiskController(scorer, schema)
probabilities = controller.predict(observed, actions)  # alias .score; objective='risk'
```

`action` 是排序后的查询组索引 tuple，空 tuple 为 STOP；schema 展开为原子。`observed` 是完整 D 维硬状态、未查询位置为 -1。外部根据冻结任务头实际执行一次后的错误构造 `error`；同一病例多个动作可以同时失败，所以每个事件独立 BCE/sigmoid，**不做动作 softmax**。`risk_prompts` 只包含已观测语义值、候选概念描述和错误事件定义，没有原输入、样本 ID、完整隐藏 z、gold、真实成本或计时。候选集合和成本决策由外部构造，控制器只打分。训练 prompt 按 batch 动态构造，不全量保留展开文本。

风险训练只接收 policy_fit；head 初始随机（除非明确加载已训练 head），不是预训练置信度。默认 backbone 冻结、head 训练；训练/导出记录 `freeze_backbone` 与 `calibrated=False`。这些分数不带 STOP 可靠性保证，更不是策略长期最优性证明。

### 保存与可复现边界

普通响应器保存全 state_dict、schema hash、配置，重载检查 schema。Nano head-only checkpoint 保存 head、任务类型、schema hash、本地 backbone 路径、长度配置和上游参考 SHA；它不是完整 backbone 归档。同时流式计算并保存该标准 HF 目录根层 `.json/.safetensors/.bin/.model/.txt` 文件的内容 SHA256 清单，重载先验证，内容变化则明确拒绝。不递归扫描训练输出目录，不将 head.pt 自身纳入清单；只接受标准根层 HF 布局，移动模型路径的 override 尚未实现。非冻结 backbone 时拒绝 head-only export，防止丢失已经更新的 backbone 权重。fake 注入模型标记 `INJECTED_TEST_NO_LOCAL`，只能显式注入同一 fake backbone 重载，不能冒充可部署 LM 检查点。冻结 backbone 在 head.train() 时仍保持 eval，避免 dropout 改变固定特征。

## 本轮验收

2026-09-22 本机最终专项回归 **19/19 通过**（CPU，6.8 秒）；torchvision 多图前向实际执行，未跳过。系统 Python 3.9 与 conda Python 的编译检查通过。文本合成数据上的拟合仅用于验证梯度和训练闭环，不作真实性能证据。

```bash
python -m unittest discover -s tests_cbmjev -p test_responders.py -v
python -m py_compile cbmjev/responders.py cbmjev/nanojev.py tests_cbmjev/test_responders.py
```

覆盖文本实际拟合、缺标 mask、split 拒绝、单次共享编码记账、模型 roundtrip、左右 padding、超长拒绝、独立 BCE 可同时高风险、冻结 backbone 不更新、语义 CE、risk/semantic head 不混载、可选 torchvision bytes 前向。HF 的真实本地 Qwen 加载、训练及 server **未验收**：本机没有 transformers/权重，不安装补齐；fake 测试不能替代它。正式数据上的语义质量、嵌套样本外来源和实时成本仍待真实实验。

## Validation-only live profiling

`cbmjev.profiling.profile_backend(prepared, responder_dir, out, raw_root=None, device="cpu", limit=100, repeats=3, batch_sizes=(1,2,4), warmup=3)` 读取已有 prepared 与 responder artifact，仅选 validation；校验 role manifest 一致且病例组未用于响应器监督。无 validation 时报错，不借用 test 或 fitting splits。它不依赖 prediction cache。

每个病例/配置/重复均创建新 `LiveEnvironment` 和会话。`batch_sizes` 指每次调用购买的**查询组数**，不是多病例 GPU minibatch；按固定组顺序分块，另测 all-at-once。模型在 profile 前已加载，不计模型加载时间；每个配置先用首个 validation 病例进行 warmup，warmup 单独记账，不进入分位数。测量配置顺序确定性轮换，减少固定首个配置偏置；不声称消除温度、OS 文件缓存或系统负载影响。

计时在 CUDA 前后同步，外部 wall 包含 lazy 原文件读取/解码、去 EXIF、会话建立、模型预处理/前向和协调开销；不包括任务头/控制器。`cold_session=true` **不等于 cold_model 或冷文件缓存**。Cheap 首次全量计算的 `shared_encoder_first_call_ms` 实际是整次响应（encoder+所有 heads 等），不是隔离测得的 encoder-only 时间，字段另附明确 scope。

输出目录禁止覆盖；逐条流式写 `timings.jsonl`，另写 `warmup.jsonl`、`summary.json`、`environment.json`。每条包含外部 sample/group ID、组与原子展开、调用数、各阶段毫秒、schema/model hash、模型实际接收文本/去 EXIF 后 PNG bytes 的内容摘要；不是路径 hash 或原文件容器 byte hash。摘要计算在被测 wall 之后完成，单独记录其开销，不重复解码图片。模型只收到内容契约，不接收这些审计元数据。

同病例同重复比较分块与整批硬回答；重复整批也检查稳定性。任何不一致令 `batch_independent=false`；仅测 all 没有分批比较时为 null。即使 true，也只是所测病例/配置/重复的经验结果，`flat_cache_certified` 始终 false，不修改 responder receipt、预测缓存或训练配置。汇总按配置记录 p50/p95/mean，但重复 episode 不是独立临床样本；不把串行 ms 数字外推成多卡吞吐。当前不拟合 declared-cost，避免将少量 profile 强行解释为稳健通用费用单位。

单测注入入口是 `profile_inputs(schema, input_rows, responder, out, ...)`；每条输入必须含 `sample_id/group_id/split="validation"/input`，图片实际内容延迟加载，原始计时逐条输出，不收集全数据集图片/完整日志到内存。正式数据入口仍通过已有 prepared 验证与训练 receipt，不能拿注入 fake 后端测试替代真实模型测量。
