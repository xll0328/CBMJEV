# 理论交付核验

核验日期：2026-09-21，T8 增补核验于 2026-09-23。范围仅 theory/；没有运行 GPU、真实数据或付费 API。

## 已运行

`python3 theory/sanity_check.py`：PASS。标准库、确定性种子和 Fraction 精确枚举。

- 11 个独立噪声率点的空/单/双 Bayes 风险等式。
- 三种相同 0.75 边际概念准确率的 pair Bayes error：0.375 / 0 / 0.5。
- 严格收益、并列边界、成本临界 `v=.125 → ρ_c=.25`。
- 2,000 组有限 action-risk/cost perturbation 的 regret inequality fixture。
- 查询少但成本更高的反例，及 batch 后可降低成本的模型内对照。
- 固定 transcript 的纯函数控制接口、合法 mask-only 标签信息。
- 边际校准/选择后失准反例。
- `M=20,δ=.05,n=203` 时 Hoeffding margin `0.1214796355`，即使零经验错误也无法认证 `α=.10`；`n=1000` margin `0.0547332831`。
- T8 的完整 29 候选、Boolean 错误、`λ=.03, τ=.5` softmax 模仿反例：期望效用最优 A 为 `0.51`，完美 soft CE 模仿偏好的 B 为 `0.55`；相应教师平均概率 `0.105373` 与 `0.112475`。这仅核对构造数值，不证明真实 CUB 分布出现反转。

检查器 JSON 明确标注 `is_real_dataset_experiment=false` 和 `is_formal_proof_verification=false`。有限数值自检不是一般数学证明；完整推导见附录。

review-trace 三个 JSON 元数据文件已用 `jq empty` 解析检查。LaTeX 片段的 proposition/proof 环境配对已人工/文本检查，其他推论在正文给出证明。2026-09-23 的 T8 修订已用本地缓存的 Tectonic 编译 `theory_preview.tex` 成功，输出 `theory_preview.pdf`，无 undefined/overfull 报告；`tests_cbmjev.test_choice_targets` 的 6 项测试通过，其中一项通过实际 Choice 教师接口复现 T8。独立 GPT-5.6 Terra 审稿代理核对了分母、期望效用、排序条件、KL 最小化和代码目标，未发现重大错误。该审阅不是形式证明，也不是 proof-checker Skill 原定 GPT-5.4 模型的完整审计；复核状态另记于证明义务账本。

## 尚未运行或不在本次证据范围

- `pdflatex` 仍不在当前 PATH；独立片段 `THEORY_APPENDIX.tex` 由 `theory_preview.tex` 引入并经 Tectonic 编译，不等同于论文整稿或正式模板验收。
- 没有形式证明助理认证（Lean/Coq 等）。
- 没有为 learned risk estimator 证明 uniform ε，也没有估计真实成本或医学可靠性。
- 初步 reviewer 只看了摘要主张；完整 prompt/response 与不同意之处保存在 [review-trace/REVIEW_RESOLUTION.md](review-trace/REVIEW_RESOLUTION.md)。最终全文由 root 统一独立审查。

所有未满足的经验前提和开放义务都在 [PROOF_OBLIGATION_LEDGER.md](PROOF_OBLIGATION_LEDGER.md) 单列。不能把“证明在假设内成立”改写成“实验已验证假设”。
