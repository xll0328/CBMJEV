# 本地数据适配器

`cbmjev.data.prepare_dataset` 只读取用户已获得的本地公开数据；不下载数据、不生成替代数据、不访问模型。成功解析不是许可验收，也不等于真实实验通过。返回 `status=PREPARED_NOT_ACCEPTED`，尚未核实的访问许可、来源真实性、pHash近重复和训练祖先隔离明确为 `NOT_CHECKED`。

```python
from pathlib import Path
from cbmjev.data import prepare_dataset

paths = prepare_dataset(
    "cebab", Path("/data/raw/cebab"), Path("/data/prepared/cebab-v1"),
    seed=17, source_revision="用户实际下载的版本或commit")
print(paths["samples"], paths["schema"], paths["membership"], paths["audit"])
```

输出目录必须不存在或完全为空；非空目录绝不覆盖。所有输入先验证，之后才写文件；写文件用独占创建。不要把失败后留下的部分目录当成成功结果，选择新的输出目录重跑。相对图片路径以返回的 `source_root`（亦在audit中）为根，不以prepared目录为根；原图片不复制。来源路径、原始元数据和哈希仅供orchestrator使用，严禁放入模型prompt/特征。

返回dict键：`samples`, `schema`, `membership`, `audit`, `exclusions`（均为Path）；`source_root`（Path）；`report`（audit字典）。文件分别为 `samples.jsonl`, `schema.json`, `membership.jsonl`, `audit.json`, `exclusions.jsonl`。前者是原 `acam-data-v1`（完整、固定顺序gold概念）；schema是 `cbmjev-schema-v1`（明确num_classes、每个概念语义values与组的0-based atoms）；membership记录 `sample_id,group_id,split,outer_split`。这里membership的split为六个运行角色，samples的split仍为外层四split；原官方split永远保留在 `provenance.source_split`。

通用选项：`fold_count=3`，`decode_images=True`。Pillow存在且启用时核查图像解码并计算RGB像素哈希；未安装/显式关闭时只做字节存在性和SHA256，audit的像素检查为 `NOT_CHECKED`，不能据此声称图像验收完成。未知选项报错。只依赖Python 3.9+标准库，Pillow可选。

## CEBaB

目录内提供 `train_exclusive.json`、`dev.json`或`validation.json`二选一、`test.json`；任一可用同名JSONL替代，但不能同一split同时放JSON与JSONL。支持records列表、列名→列表、pandas列名→行索引→值、ID→record，以及单个JSON的split→table格式。忽略inclusive/observational，不拼入训练；不直接读Parquet/HF磁盘格式，用户可用HF本地snapshot导出上述格式后固定来源revision。

必需身份/输入字段：`id,original_id,is_original,description`；is_original必须为真正JSON布尔值。字符串ID保留前导零，整数ID转字符串；缺失家族ID报错，不借助标签或文本猜编辑家族。`original_id='0'`及`'000000'`是合法家族ID，不当missing！作者README原文示例正是`000000`，而HF revision的ID格式可能不同。当前作者分析代码直接按original_id分组。对于空、`None`、`null`、`-1`等未经本适配器核实的特殊编码，拒绝导入，需依据该版本的原始文档显式修正版本映射，不能把所有原文误合并，也不能盲目截断id。[作者格式说明](https://github.com/CEBaBing/CEBaB)、[作者分组实现](https://github.com/CEBaBing/CEBaB/blob/gh-pages/code/eval_pipeline/utils/data_utils.py)。

唯一模型输入为description；原文字面拼写保留。概念顺序food,noise,ambiance,service，Negative=0/Positive=1/unknown=2。空值/缺key→MISSING_ANNOTATION；no majority→NO_MAJORITY，二者value=null，不变成absence或unknown。目标review_majority为1..5→0..4；缺失与no majority同样保留、单独统计，不进入有监督task损失。非空未知标签严格报错。标注分布支持字典、JSON字典字符串、Python字典字面量字符串（仅safe literal解析），核查非负有限计数与已知key后归一；不是模型置信度。

## CUB-200-2011

2026-09-22 官方实际包的顶层 `attributes.txt` 与 `CUB_200_2011/` 同级，尽管原包 README 将它写在 attributes 子目录。可使用 [官方包校验与准备工具](../tools/prepare_cub_official.py) 自动传入正确字典路径；该工具先验证官方 MD5/字节数，再安全解压，不下载数据。

source指向含 `images.txt,image_class_labels.txt,train_test_split.txt,classes.txt,images/` 的解压目录，以及 `attributes/{attributes.txt,certainties.txt,image_attribute_labels.txt}`。可显式传 `attributes_file=Path(...)`、`certainties_file=Path(...)` 指向同版本官方字典，写入来源哈希；绝不根据分类文件夹猜类别或根据图片行号拼label。[官方入口](https://www.vision.caltech.edu/datasets/cub_200_2011/)。

图片/类别/split按image ID严格一对一join，重复/陌生ID、未知class、坏split报错。字典ID必须从1连续编号，属性排序按原ID；不硬编码200/312以便验证微型fixture，真实包数量在audit中另核对。类ID减一；`cub_attr_001`等为概念ID，binary values=[absent,present]；group为属性描述`::`前缀，组数由字典计算，组内不是one-hot。

正常annotation严格解析image,attribute,is_present,certainty,time五列。certainty整数含义仅从文件读取：not visible→NOT_VISIBLE、guessing→UNCERTAIN_ANNOTATION，均value=null；probably/definitely→OBSERVED原is_present。缺少annotation行显式missing。certainty其它词报错，原annotation三元组留在audit_metadata；不把gold缺失性作为运行时状态。默认不裁剪bbox，不使用112个class-majority属性。图片路径拒绝绝对路径、URL、`..`和逃出source的symlink。

唯一原生格式例外：已核验官方annotation文件SHA256 `5ebb9782d589f41a9c046bc7c5b1365e01839308e7cf88cbe83c0fb6d5362d98` 中606行有六列。仅该完整hash、images 2275/9364×attributes 10..312的完整606-key集合及已知suffix形式可通过；保留前四语义字段，duration=null，原行/后缀/行号写 `annotation_anomalies.jsonl`，状态 `AMBIGUOUS_TRAILING_FIELDS`，不推断后缀含义。其它非五列格式仍拒绝。该sidecar只在出现异常时生成，返回dict额外提供 `annotation_anomalies` 路径；无异常数据输出不变。来源依据和完整处置记录见 [数据访问记录](DATA_ACCESS_STATUS_20260922.md)。

## Derm7pt

2026-09-22 实测官方 ZIP 需要 HTTP Basic 认证；用户须亲自完成机构信息登记与许可确认，图像和 metadata 都在受控包中，不存在已核实的匿名完整 metadata 入口。实际操作与访问边界见 [数据访问记录](DATA_ACCESS_STATUS_20260922.md)。

source应有 `images/`、`meta/{meta.csv,train_indexes.csv,valid_indexes.csv,test_indexes.csv}`。索引CSV必须有作者原header `indexes`，值是原meta文件**0-based行位置**（不是case_num）；先join后才排序导出。跨split索引重复/越界/未覆盖全体行均报错。[作者最小示例](https://github.com/jeremykawahara/derm7pt/blob/master/minimal_example.py)、[作者类别映射与iloc实现](https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py)。

模型只取derm图；clinic列仍必需，非空clinic文件需存在，仅留audit sidecar并参与重复检查。默认以原source_row为case身份；case_num只审计，不假装patient ID。仅有已确认可信患者列时传 `patient_column='实际列名'`；完整非空患者ID决定group，但patient真实性仍需人工审计。

严格使用作者Derm7PtDatasetGroupInfrequent的五类diagnosis与七组概念合并。概念顺序pigment_network、blue_whitish_veil、vascular_structures、pigmentation、streaks、dots_and_globules、regression_structures，类别数3/2/3/3/3/3/2，共19个概率坐标、7个概念/查询组。`linear irregular`→vascular类别2；`within regression`→vascular类别1；`diffuse/localized regular`→pigmentation1，irregular→2；blue/white areas/combinations→regression present1。原始unknown非空字符串报错；空字段missing绝不变absent。诊断映射完整记录于audit.adapter.diagnosis_mapping，不允许从字符串包含melanoma等模糊匹配。

## 划分、去重与未完成验收

原生family/case/image group先通过文本精确规范化（NFC、LF换行、首尾空格仅用于去重，原输入不改）、图像字节与可用时RGB像素哈希合并组件，组件ID取原生ID最小字符串。临床/皮肤镜都参与图像重复组件。跨官方split组件按test>dev>train保留高优先级样本，低优先级样本写exclusions，不搬进test。同split重复仍保留、group统计；pHash近重复**未实现/未核实**，不能把精确去重等同全面泄漏审计。

稳定排序键为 `SHA256('acam-split-v1|seed|purpose|dataset|group_id')`，同哈希按ID，比例按group数floor，不按派生edit行数。CUB官方train按类别（混合/缺失单独stratum）每层前floor(n/10)为calibration、接着同数validation；purpose=cub-holdout。CEBaB dev按全部group一分为二，前floor(n/2)为calibration；Derm valid按诊断stratum一分为二；均purpose=dev-half。测试保留；任何小层为空照实报告。

train内purpose=pilot-role，按stratum前floor(.6n) responder_fit，接着floor(.2n) head_fit，余policy_fit。CEBaB按原文目标分层，原文缺失为missing-original，多目标组件为mixed；CUB/Derm按目标类别分层。所有group成员同角色，不因sample ID排序而拆家族。outer-fold则另按purpose=outer-fold排序rank%fold_count，只在train赋fold；这份角色manifest本身**不证明**训练代码没有泄漏、也不实现嵌套cross-fit。角色不足的微型/稀有层报告为空，不改比例凑齐。

validation可开发/温度拟合；calibration只作完整模型和有限候选族冻结后的终局认证，不能拟合任何R/f/V/r、改prompt或看到认证标签后新增阈值仍沿用原M；test只评估。数据适配器是离线可信预处理，会读取gold进行标准分层与缺失统计，不把gold放进运行时request。

audit分开source原始数量、canonical数量、目标有效数量、目标status、逐概念status、group数量、六角色数量及空strata；保存词表/split/来源文件哈希。早期验收仅使用tiny fixtures；2026-09-22开始在xtech服务器接入真实数据，最新完成项见 [部署记录](SERVER_DEPLOYMENT_20260922.md)，真实数据不留Mac。仍须核对来源版本、许可、预期规模、pHash/患者独立性、模型训练祖先与真实输入白名单，完成 `execution/data/DATA_PROTOCOL.md` 的验收清单。

```bash
python -m unittest tests_cbmjev.test_data -v
```
