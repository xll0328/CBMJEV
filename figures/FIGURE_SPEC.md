# 论文图表施工图与审计

日期：2026-09-21。三张 SVG 为**可浏览草稿**，不是实验结果。真实数据前沿不画假点、假曲线或假误差条。Fig 3 的线只来自注明假设的解析 toy 公式；不能当真实数据优势。

可直接打开 [gallery.html](gallery.html)。正文最多四张图、两张表；默认双栏通宽，约 7 英寸。SVG 画布宽 1280，最小文字 22 px，缩到 7 英寸约 8.7 pt。终稿进一步在真实 LaTeX 版面检查。配色采用蓝 `#0072B2`、橙 `#D55E00`、蓝绿 `#009E73`、中性灰，颜色与线型/标签双编码。

## Fig 1：从固定预测向受限的测量过程

类型：solution overview，兼具 motivated example。文件：[fig1_measurement_boundary.svg](fig1_measurement_boundary.svg)。

为什么用比较式架构图：它一次说明三个关系——原输入只进入测量器、策略只基于已得证据、返回动作可以是一项/批次/STOP。单张医生流程图会把类比误画成专家监督事实；模块堆叠图则隐藏信息边界。

布局：1280×880。上行 all-at-once，小灰色支路；下行主系统，左侧 raw input / responder，右侧 acquired evidence / task head / controller；粗虚线垂直划分“可见 x”的组件。向左回环箭头只携带 query IDs，不携带 latent。底部三条测试 invariant。

箭头语义：实线为允许的数据流；橙色回环为控制动作；虚线为信任/信息边界，不表示概率关系。不存在 raw x→policy 或 raw x→task head 的箭头。若实现新增 metadata，每种 metadata 的来源必须写入图注和协议。

Caption 草稿：

> **Adaptive acquisition changes when evidence is measured, not what the bottleneck is allowed to observe.** Top: all-concept inference obtains every response in one batch. Bottom: the responder alone reads the raw input, while the controller and task head use only acquired typed observations. The controller requests a singleton or batch, or stops; query identities travel to the responder and observations travel back. The dashed boundary excludes direct raw-input and unqueried-response access. The three checks test this dependency restriction, not semantic correctness or clinical safety.

必须由实现验证：H 不含 y/metadata proxies；初始 query 输入无关；soft probabilities 单列；R 是否history-independent。图中 budget 指冻结 declared-cost budget，不是由当前输入真实 latency/tokens 更新的额外信息通道。图不宣称已经省时。

## Fig 2：同一组实验，两种横轴

类型：experimental results。文件：[fig2_dual_cost_template.svg](fig2_dual_cost_template.svg)。当前为无点、无数值的 PENDING 画板。

布局：1280×700。左右两个等高坐标轴，同一 y 范围；左 x = mean acquired concept groups，右 x = end-to-end latency (ms)。每一条线对应同一 policy family，同一 budget points，不能把左图的 query budget 数值复制到右图。All-batch 一般是一个点，不为了图面丰满连出虚构前沿。

系列：All-batch / static prefix / lazy tree / strong AFA / controller。线型与标记固定；主体不放 random，random 可进附录。若拥挤，将两数据集分成两行而非添加十几种方法。

字段：`dataset, seed, policy_id, responder_revision, classifier_id, budget_id, task_metric, task_metric_ci, mean_groups, mean_requests, latency_p50_ms, latency_p95_ms, serving_config, response_source`。

统计：同一组 test cases，group bootstrap 配对；跨训练 seed 的变异和样本 bootstrap 不是同一种 CI，不混写。x 轴 latency 也有重复计时波动，可用水平误差或补表。各数据集 y 轴分别固定合理范围，左右图必须相同；若截断纵轴需清楚标注并给全范围附图。这里不预填范围以免创造虚构结果。

Caption 模板：

> **[PENDING finding: query savings do/do not translate into measured runtime savings.]** Both panels evaluate the same frozen policies and responder on the same test examples. The left panel measures acquired concept groups; the right measures end-to-end live-inference latency under **[hardware, concurrency, warm/cold setting]**. Points are validation-selected operating points; **[interval definition]** describes uncertainty. All-batch includes shared input processing. Cache-replay timing is excluded from the runtime panel.

误导检查：不把GPU总时/单样本wall time混在同x轴；不在不同硬件上画一条方法对比；不省略all-batch；不测试集调λ后挑漂亮点；实际右图可能与左图排名相反，必须保留。

## Fig 3：噪声与成本的机制边界

类型：analytic motivated example + empirical mechanism。文件：[fig3_analytic_boundary.svg](fig3_analytic_boundary.svg)。当前左图为**解析 toy**，右图为真实实验待填面板。

主文目的：解释为什么同一获取规则在两种测量器上可能得出相反结论；不把 XOR 或有限样本界声称为首次。

左图：T4 的明确构造。两个独立 Bernoulli(1/2) 概念，y 为 XOR，观测经独立 flip probability ρ，ρ∈[0,1/2]。单项 Bayes error=1/2；双项 error=2ρ(1−ρ)；两项一批的净价值为正 iff `λ κ_batch < (1−2ρ)^2 / 2`。x=ρ，y=`λ κ_batch`，曲线下方说明 only this toy 的 acquire region，上方STOP。图上必须写 `ANALYTIC TOY — NOT EMPIRICAL`，不标方法名或真实dataset。

右图：计划中的 measured noise–cost mechanism map，维度按实际pilot确定，建议 real responder noise level / measured batch overhead ratio，色值 `Δ matched task-cost objective versus strongest baseline`。当前仅空框标 PENDING；不能根据理论曲线替真实 heatmap填色。若实际可控noise定义是翻转概率而不是真实respondererror，必须单独命名 synthetic。

辅助假设：图上的“ρ越高价值越低”只适用独立翻转；共同模式相关错误可能在 parity 中抵消。因此真实研究需估计联合错误而非只用平均conceptaccuracy解释结果。

Caption 草稿：

> **Complementarity can justify a batch even when no single query improves immediate prediction, but noise and batch cost determine whether that batch is worthwhile.** Left: an exact illustrative extension of the known XOR counterexample. Under independent symmetric observation noise, acquiring both concepts beats stopping precisely below the displayed boundary. This panel is analytical, not a fitted or empirical result. Static-pair and adaptive-pair policies coincide in this construction, so it does not demonstrate an instance-adaptive advantage. Right: **[PENDING: matched real-responder or explicitly synthetic mechanism results with units and uncertainty]**. The construction does not establish the novelty of coalition acquisition and does not predict every real responder's error structure.

近邻归属：[RouteCert](https://arxiv.org/abs/2608.15520) 已覆盖小联盟/互补与终局条件风险；最终引用以理论文档核实的 Appendix Proposition S2 为准。正文按 theory/THEORY_APPENDIX.md 的 T3、T4 假设写；完整证明与 T1/T2/T5/T6 放附录。

## Fig 4：分叉轨迹、干预重规划与失败

类型：case-based analysis。当前仅施工规格，必须有真实 logs 后再绘，不造病例。

布局：3 行 × 4 列。三行依次是两个具有不同后续查询的病例，以及一个高置信错误早停病例。列为 input缩略图/已有证据/后续action/终局与成本。若medicalimage不允许发布，图像留dataset授权方式或换CEBaB短文本，不擅自再分发。

选择规则：在冻结测试集上预先定义 strata，从每类按固定seed抽取；失败例至少一条。病例注释只引用数据已有labels和model日志，不增写医生判断。另给对全部样本的 failure-rate 表，不用三个例子证明普遍性。

字段：`sample_id, group_id, step, acquired_query_ids, typed_response, response_status, risk_stop, chosen_action, predicted_risk_after_action, actual_incremental_cost, final_prediction, target, error_type, intervention_mode`。风险显示先校准后的数值或标uncalibrated；不能把动作softmax叫作发生概率。

Caption 模板：

> **[PENDING finding about conditional branching and a concrete failure.]** The examples follow a prespecified selection rule. Each node displays only acquired concept evidence, and each edge shows the paid query and measured incremental cost. The third case is an error, included to expose **[specific failure mechanism]**. **[If correction is shown:]** Concept correction uses existing benchmark labels for a diagnostic intervention; it is not an observed clinical interaction or a causal claim.

## Table 1 / 2 与附录

- Table 1 主结果：相同成本点下所有强基线，真实runtime、query、macro-F1/accuracy、seeds；无需塞全部λ。
- Table 2 机制：oracle-vs-noisy训练、batch-vs-single、r-vs-V、hard-vs-soft、依赖审计pass/fail；未实施的设置写NOT_RUN。
- 附录：全前沿、所有seed、可靠性图及每bin样本数、预算/并发扩展、完整理论与小样本区间。

## 工具、审计与交付标准

草稿用手写 SVG + 本地 HTML；无外部字体、脚本、网络依赖。正式实证图从锁定 JSON/CSV 生成矢量PDF/SVG；不要在图形编辑器里手改数值。粗稿可导入Figma/PowerPoint，但最终脚本与数据应可重建。

| 审计 | 当前状态 | 终稿要求 |
|---|---|---|
| empirical truthfulness | PASS by construction：无伪造结果 | 所有点回溯到run ID |
| vector | SVG | SVG/PDF双格式 |
| type vs truth distinction | 明示 | 图注不得删 |
| color-blind access | 颜色+文字/线型 | 灰度打印复核 |
| font at final size | 1280wide / min22px / double-column | LaTeX实页复核≥8pt |
| no fake curves | Fig2/右Fig3空白 | 只能由真实日志填 |
| analytic toy labeling | Fig3显著标签 | 不能裁掉标签 |
| figure captions self-contained | draft已配 | 首句改成有证据的finding |
| layouts | 已用独立临时 profile 的 headless Chrome 渲染并逐张目视检查 | 最终 LaTeX 版面仍需复核 |

三张 SVG 已通过 `xmllint --noout`；PNG 浏览副本为同名 `.svg.png`。Quick Look 曾将宽图错误裁成正方形，已全部以正确尺寸的 Chrome 渲染覆盖该无效预览；源 SVG 未因此改变。当前严重性：CRITICAL=0（设计层面）；MAJOR=真实结果未运行、最终版面尚未冻结；MINOR=图中文字与最终术语尚可压缩。先做三件事：验证真实成本字段；锁Figure2的配对样本与operating points；取得一个成功和一个失败的完整合法transcript。
