---
id: SE.RQ_SECTION_BINDING
slug: se-rq-section-binding
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-experiments, self-review]
domains: [se]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

每条 RQ 与其结果章节必须**三重冗余绑定**：

1. 结果小节标题**逐字复述该 RQ**（如 `RQ1: What distinguishes AI repos from general OSS?`）；
2. 小节开头一句 signpost：`To answer RQ1, we …`；
3. 一张 “results at a glance” 表带 **RQ 列**，使索引同时充当 RQ→finding→section 映射。

至少满足“标题复述 + signpost”两项；glance 表的 RQ 列为强烈推荐。

## Rationale

绑定让 reviewer 在任意入口（标题、小节首句、总表）都能回到对应 RQ，结果因此可跳读、可核对完备性。仅靠隐式 `H1→§h1` 对应会迫使 reviewer 自行重建映射，是常见的导航缺陷。**面向 empirical-SE 论文，通过 `se-*` profile 激活。**

## Check

- **LLM 语义检查**：每条 RQ 是否满足“标题复述 + signpost”两项；glance 表是否含 RQ 列
- **通过标准**：从结果章节任一小节标题即可直接读出它回答哪条 RQ

## Examples

### Pass

```latex
\subsection{RQ1: What distinguishes AI repositories from general OSS?}
To answer RQ1, we contrast the identity and execution-context layers against a matched
non-AI control. AI repositories carry an identity-layer vocabulary ...
% Table~\ref{tab:glance} 的首列即为 RQ 列
```

### Fail

```latex
\subsection{Identity vocabulary}
The AI signature is an identity-layer vocabulary ...
% 标题不含 RQ 标签、无 "to answer RQx" signpost；glance 表按 F1–F22 索引、无 RQ 列
```
