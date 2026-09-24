# CBMJev 首发与论文就绪检查表

日期：2026-09-22。**本文件是验收清单，不是完成声明。** 未附证据的框保持未勾选。[本轮验证报告](VERIFICATION_20260922.md)统一记录实际检查；不能从旧pilot测试结果推断新统一包或真实数据已通过。

状态词：`DESIGN_AVAILABLE` / `IMPLEMENTED_NOT_VERIFIED` / `FIXTURE_VERIFIED_REAL_PENDING` / `REAL_VERIFIED` / `NOT_RUN` / `BLOCKED`。禁止只写done而不说明验证范围。

## A. L0统一工程包：本地就应通过

- [ ] `python -m cbmjev --help` 与所有实际存在的子命令可用；文档不使用未实现参数。
- [ ] 清洁临时环境能导入核心包；可选视觉/Nano依赖缺失有明确提示，不破坏CPU smoke。
- [ ] fixture smoke生成新输出目录、环境/seed/config/hash/模式标签；已有目录不会被静默覆盖。
- [ ] 核心 tests实际运行并记录命令、Python/PyTorch版本、通过/跳过/失败数；不能只编译不执行。
- [ ] CEBaB/CUB/Derm fixture adapters有有效/非法值、缺标、重复身份、split冲突测试。
- [ ] query groups和原子attributes分离；一次组查询完整展开且只收一次组费用；single/pair/all去重。
- [ ] 缺标/不确定/不适用/负值/未查询语义分离；gold侧status不能直接进入runtime observation。
- [ ] 数据和训练祖先审计能拒绝policy-target group出现在任一监督R/f祖先中的配置。
- [ ] mask-aware head在编码前屏蔽未知值；同H换raw x/ID/token/timing/cache状态不改policy/head。
- [ ] 初始动作输入独立；固定seed重放一致；f对同证据排列不变。
- [ ] r=动作后错误概率，独立sigmoid；V=signed loss-drop；没有动作softmax概率误用。
- [ ] empirical lookahead/DP标清训练经验分布、小K限制和深度；不是published AFA复现。
- [ ] replay与live的响应/成本来源在机器输出中明确区分，replay墙钟不进部署加速列。
- [ ] budget用冻结declared costs更新；actual计时外部记账，不形成控制器旁路。
- [ ] cache action/prompt/model/schema/split/precision键完整；batch-dependent响应不能借singleton cache冒充。
- [ ] 默认test关闭；显式开关、freeze manifest和输出日志能辨认最终评估。
- [ ] certification只读取预冻结完整候选系统，输入loss有界/units合法；无通过候选时fail closed。
- [ ] summary拒绝缺metric/NaN/模式混淆，不把NOT_RUN填0，不自动从不同R结果拼成matched policy表。
- [ ] matrix/grid默认planning-only，不下载、不启动GPU；执行需明确配置与已通过gate。
- [ ] 单seed/显式seed列表helper通过help、dry-run无写入、已有输出/缺数据/重复seed或GPU拒绝；CUDA需明确单job设备，不扩展整matrix。
- [ ] 启动Python前配置CUBLAS deterministic workspace，保留用户已有设置；真实CUDA路径另验，CPU通过不替代。
- [ ] audit-responses只以gold作监督审计，报告实际有效label denominator和共同可评估样本；不把gold错误/缺标写入runtime状态。

证据栏：`verification_report = docs/VERIFICATION_20260922.md`；`verified_commit_or_tree_hash = SEE_REPORT`；`test_log = SEE_REPORT`。未逐项验收的框仍不自动勾选。

## B. L1 CEBaB真实首发：必须在服务器完成

- [ ] 官方数据revision、原包哈希、许可、实际counts与排除receipt齐全；非官方镜像差异说明。
- [ ] 固定train_exclusive，不拼inclusive/observational；original/edit family不跨角色/split。
- [ ] R只读评论description；评分、edit_goal/type、family ID不进入模型输入。
- [ ] R-fit/head-fit/policy-fit祖先互斥；不以final refit替换已认证系统。
- [ ] cheap R真实训练、checkpoint可信可加载；预测来自模型而不是gold concepts。
- [ ] full c/full z/raw x能力诊断完成，语义合法率与准确率分开。
- [ ] 全链R→cache→f/r/V→validation→summary完成，所有output可由配置重建。
- [ ] all/static/动态risk/value至少同场；small-K empirical lookahead可作额外参照；random仅sanity。若独立lazy tree尚未实现，明确标pending，不宣称已通过完整强静态对照。
- [ ] 三seeds的随机链清楚：R/f/policy分别哪些重训，哪些共享；不把三次replay称三次训练。
- [ ] 独立环境依据runbook复跑至少一条real run，验证指标在声明容忍范围内。
- [ ] 初步成本profile包含all-at-once；若首发只有replay，醒目标明无实际加速证据。
- [ ] 最终test与认证状态诚实；未认证可以发布，但不能写保证可靠。
- [ ] 测试集未反复调参；探索性结果与冻结结果分开。
- [ ] 病例/评论原文、权重、cache是否可再分发逐项明确；默认不打包原数据或凭证。
- [ ] README/project card说明“research prototype、CEBaB verified、vision/strongAFA pending”。

证据栏：`real_dataset_receipt = PENDING`；`server_profile = PENDING`；`three_seed_runs = PENDING`；`independent_reproduction = PENDING`。

## C. NanoJev命名与贡献边界

- [ ] 明确区分官方Jev、NanoJev独立仓库、本项目adapter；不暗示官方权重/架构/合作。
- [ ] 实际使用Nano时固定commit和model/tokenizer revision，并记录修改与许可证。
- [ ] 文本模型不能被写成原生视觉模型；游戏页面/图片demo不作为图像能力证明。
- [ ] cheap responder与Nano-style responder公平比较；没有必要性证据不写“Jev is essential”。
- [ ] typed格式合法不等于无语义错误；模型confidence不自动等于校准后终局错误率。
- [ ] 没有复现官方RLCD就不称复现；CE/Brier与候选打分归属明确。
- [ ] 不保证引用量、投稿接收或“热点窗口”；项目名传播与科学claim分开。

## D. L2完整论文：首发之外的必需证据

- [ ] 两主张C1/C2对应五实验块，无第六个松散主故事。
- [ ] CUB大概念池和图像语义协议已真实验收；Derm作为条件扩展，明确case/patient边界。
- [ ] 独立lazy tree与ACO、至少一个适合的SEFA/BRiG强近邻完成同响应/动作/成本适配与验证；不是仅related work引用。
- [ ] 近邻版本/发表状态核对，原生方法与本项目adapted版本分开命名。
- [ ] live lazy执行实测，没有预先计算全量再伪装按需；cheap shared-head负对照保留。
- [ ] 匹配任务质量/成本的操作点在validation冻结，test不选最好λ。
- [ ] 错误耦合与batch成本机制有真实对应；toy图明显标analytical，不证明适应性本身。
- [ ] CI/bootstrap单位为独立group；多轨迹/多属性不膨胀n；seed和样本CI含义分开。
- [ ] 所有理论假设与实现对应；Hoeffding需要独立iid有界units，一般exchangeability不可替代。
- [ ] calibration只认证冻结候选；temperature/model selection留validation/inner-dev；M与hash固定。
- [ ] no-bypass结果不被写成语义正确性/因果/医疗安全保证。
- [ ] 正例、负例和错误早停均按固定规则选，展示总体失败频率。
- [ ] 论文数字全部追溯run ID；未跑项不填数字；结果不支持时使用 [条件结论分支](../paper/RESULT_BRANCHES.md)。
- [ ] 成本报告同时含研究训练成本、部署成本与额外审计成本，三者不混写。
- [ ] claim audit、citation audit、图表可读性/数据来源检查后，才冻结摘要和贡献。

## E. 真正发布前的外部操作清单

本地准备不等于授权代发。用户决定发布平台/仓库/署名/许可后，才执行外部上传。

- [ ] 明确项目名、作者与贡献者、代码许可；不替用户生成不存在的作者或机构。
- [ ] 检查秘密、医学访问口令、API key、用户绝对路径与不允许分发的样本。
- [ ] release只含可分发代码、配置、schemas、说明与许可允许的artifacts；原数据提供获取步骤。
- [ ] 给release打内容/版本哈希；列exact dependencies及支持环境；不使用“latest”作为可复现标识。
- [ ] 一条最小CPU smoke、一条已验证CEBaB real流程、完整limitations可直接访问。
- [ ] 引用文件只使用真实、已核实作者/标题/日期；没有论文DOI时不编造BibTeX或占用真实标识。
- [ ] 明确known gaps、issue模板、数据许可边界以及安全的失败处理；不要把路线图写成已实现能力。

最终发布判定：L0全部通过可发布工程预览；L1通过可发布CEBaB可复现原型；L2通过且主张证据充分才称论文实验包。任一层通过都不自动代表下一层通过。
