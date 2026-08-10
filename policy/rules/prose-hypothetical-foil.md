---
id: PROSE.HYPOTHETICAL_FOIL
slug: prose-hypothetical-foil
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "(?:^|\\. )(An?) [a-z][a-z\\- ]{3,45}would\\b"
    mode: match
  - pattern: "\\b(Once you|If you (think|view|read|consider|imagine|take)|Suppose you|Imagine that|Consider what happens)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

直接陈述结果，不要先虚构一个不存在的对照物或假想读者来给真实主张搭台。禁止 `A method that only did X would stop there. Ours does Y.` 式的稻草人对照，禁止 `Once you view it as X, ...` 式的第二人称设问。

## Rationale

两种写法是同一个动作：作者先造一个场景，再把真实结论放进去，让结论显得比它自己更有分量。

稻草人对照的问题是它论证的是一个不存在的对手。读者要先接受"存在一个只会描述数据的方法"这个前提，才会觉得"我们的会预测"很了不起——而真正的证据（外推误差）自己就够，不需要这个前提。

第二人称设问（`Once you...` / `If you think of it as...`）把论文变成教学材料。学术论文的读者不需要被牵着走完一次推理，直接给结论和依据即可。

两者都是 AI 生成文本的高频修辞，因为模型被训练成"把内容讲得引人入胜"，而学术语域要的是"把内容讲得可核对"。

## Check

- **regex 搜索**：句首 `A/An <名词短语> would`；`Once you` / `If you` / `Suppose you` / `Imagine that`
- **检查范围**：`.tex` 正文
- **不适用**：
  - 真实的反事实分析（消融、失败模式讨论）中带具体对象的 `would`，如 `Removing the pilot would cost 32 paths per query`
  - 算法/协议描述中面向实现者的第二人称步骤（应放 `algorithm` 环境或附录）
  - 直接引语内部
  - 会议提供的样板文件（`checklist.tex` 等）：其中的 `If you obtained` / `If you answer` 是指令性用法，pattern 已收紧为只抓 `If you think/view/read/consider/imagine/take` 这类搭台动词
- **判断线**：`would` 的主语是不是一个**被造出来只为对照**的抽象物。是则违规，是真实存在的基线/变体则合规。

## Examples

### Pass

```latex
The beta-binomial prediction is within 0.8 to 2.5 pp of observed accuracy,
where the independence model diverges by 17 to 31 pp.
Reading $K$ paths as $K$ correlated observations makes the effective vote
count the quantity to measure.
```

### Fail

```latex
A statistic that only described the data would stop there. This one predicts.
Once you read $K$ paths as $K$ correlated observations, asking how many
independent votes they are worth is the natural next step.
```
