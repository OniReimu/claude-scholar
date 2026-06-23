---
id: SE.RESEARCH_QUESTIONS_EXPLICIT
slug: se-research-questions-explicit
severity: error
locked: false
layer: domain
artifacts: [text]
phases: [ideation, writing-experiments, self-review]
domains: [se, security]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

Empirical-SE 与 AI-security 论文，**当评估覆盖多个 contribution 或 ≥3 个*异质*评估维度时**，必须用**显式编号的 Research Question (RQ)** 驱动全文。单一主结果、同质评估（同一 metric 换 setting/benchmark）的论文可改用 Main Results / Ablation / Analysis，不强制 RQ。满足上述条件时：

1. **数量**：2–4 条（empirical-SE 典型）；多面的 security/systems 攻防评估可放宽至 ≤6，但每条仍须对应一个 contribution 或一个 validity/positioning 检查（robustness / baseline / end-to-end）；
2. 开放式问法（`What… / How… / To what extent…`），不写成陈述句结论；
3. 每条 RQ 紧跟一段 *justification*（Rationale / Motivation），说明为何值得问；
4. 放置位置随 venue：会议（ICSE/FSE/ASE/CCS）放 Introduction 末尾或 Evaluation 开头；期刊（TSE/TOSEM/EMSE）放独立 `Objective and Research Questions` 节。

若采用 pre-registered 方向性假设，可作为每条 RQ 下的 committed answer-shape 保留，但 RQ 在标题与 intro 中为 primary。

## Rationale

RQ 是 empirical-SE 论文的导航骨架：reviewer 据此判断 scope、贡献边界与结果完备性。把论点埋成 intro 段落里的陈述句 hypothesis 会让论文失去可导航性，是该领域最常见的 presentational gap。AI-security / 高数值实证类论文继承同一红利：当评估维度异质（viability / mechanism / defense / robustness / baseline / e2e 等）时，编号 RQ 把"实验是否 align claim"从源头锁死。**本规则面向 empirical-SE 与 AI-security 论文，通过 `se-*` / `security-*` profile 激活。**

## Check

- **LLM 语义检查**：
  - 是否存在显式编号 RQ（2–4 典型；多面 security/systems 评估可至 ≤6）
  - 若评估为单一主结果/同质，是否合理地未用 RQ（此时本规则视为 N/A）
  - 每条是否开放式问法 + 一段 justification
  - 放置位置是否符合 venue 惯例
- **通过标准**：reviewer 只读 Introduction 即可列出全部 RQ 及其动机

## Examples

### Pass

```latex
We ask four research questions, stated at the end of this section.

\smallskip\noindent\textbf{RQ1 (signature).} What dependency signature distinguishes
AI repositories from general OSS? \emph{Knowing the signature is a precondition for any
downstream measurement.}
% RQ2–RQ4 同构：开放问法 + 一句 justification
```

### Fail

```latex
In this paper we hypothesize that AI repositories carry an identity-layer vocabulary,
that discovery never plateaus, and that cards and code diverge.
% 论点埋成 intro 陈述句，无编号 RQ、无开放问法、无 justification
```
