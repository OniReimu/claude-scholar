---
id: PROSE.ABSTRACT_AGENCY
slug: prose-abstract-agency
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
  - pattern: "\\b[A-Za-z]+'s (job|task|role|purpose|goal|mission|duty)\\b"
    mode: match
  - pattern: "\\b(model|method|analysis|framework|estimator|diagnostic|analogy|theory|equation|algorithm|approach|result|paper|metric) (wants|knows|decides|refuses|believes|cares|tries)\\b"
    mode: match
  - pattern: "\\bis (built|designed|meant) to (catch|hunt|chase|beat|kill|attack)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

抽象名词不做行动者。禁止给方法、模型、公式、类比赋予人类意图（`the analogy's job`、`the method wants to`），也禁止用比喻性动词描述它们与抽象对象的关系（`carries decades of validation`、`delivers votes`、`built to catch`）。改用字面动词：`has been validated`、`is worth`、`detects`。

## Rationale

拟人化把一个可核对的技术陈述换成一个意象。`The analogy's job was to raise the question` 听起来有画面，但它没说清是谁在什么时候基于什么提出了这个问题；`The analogy is where the question came from` 说清了。

比喻性动词的危害更隐蔽：它们读起来像是在陈述事实，实际上把关系模糊掉了。`The estimator carries decades of validation` 里 "carries" 既不是"被验证过"也不是"包含验证"，读者只能靠语感补。写成 `has been validated for decades` 之后，这句话变成可以被质疑、也可以被核实的断言。

同一个比喻动词在一份文档里复用（如 `carries` 出现两次指两件不同的事）是更强的信号：它说明作者在套模板而不是在陈述。

这类写法在 AI 文本中密度极高，因为"生动"是通用写作训练的目标，而学术语域要的是字面精确。

## Check

- **regex 搜索**：`X's job/role/purpose`；抽象主语 + 意志动词；`built/designed to catch|hunt|beat`
- **人工判断（regex 覆盖不到）**：抽象名词做主语时的比喻性动词，常见有 `carries`、`delivers`、`buys`、`captures`、`embraces`、`speaks to`、`lives in`。判断线是能否换成字面动词而不损失信息；能换就说明原来的是修辞。
- **重复检查**：同一比喻动词在文档中出现两次以上，即使单次可接受也应改写
- **不适用**：
  - 领域内已固化为术语的搭配（`the model learns`、`the classifier predicts`、`attention attends`）
  - 人类主语（`the authors want to`、`reviewers ask`）
  - 直接引语内部

## Examples

### Pass

```latex
The estimator has been validated for decades in survey statistics.
Thirty-two sampled paths are worth 1.12 independent votes.
High redundancy is the case the diagnostic detects.
The analogy is where the question came from.
```

### Fail

```latex
The estimator carries decades of validation in survey statistics.
Thirty-two sampled paths deliver 1.12 independent votes.
That is the case the diagnostic is built to catch.
The analogy's job was to raise the question.
```
