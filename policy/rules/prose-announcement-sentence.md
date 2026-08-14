---
id: PROSE.ANNOUNCEMENT_SENTENCE
slug: prose-announcement-sentence
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: [PROSE.RHYTHM_VARIANCE]
constraint_type: guidance
autofix: none
---

## Requirement

短句要承载**主张**，不要只做**预告**。禁止用"抽象名词 + 系动词 + 抽象形容词"的标签句开段，例如 "The difficulty is structural." "This can be made precise." "The reason is simple." "This deserves closer attention."

短句本身没有问题；问题是句子只宣布内容即将到来，而不给出内容。

## Rationale

人类领域专家写短的主题句，写的是可检验的论断；LLM 写短的主题句，写的是分类标签。"The difficulty is structural" 里的 "structural" 不可证伪也不可检验，读者读完这句什么都没得到，它只是在敲鼓预告后面有解释。

对照 UniProve（NDSS 投稿）中一个合格的短开段句：*"The gap is not only conceptual."* 它同样短，但指代的是前文已命名的具体对象（the gap），并且承载了一个转折论断（从理论转向实测），下一句立刻兑现。

这条与 `PROSE.SHORT_PUNCHY_FRAGMENTS` 互补：那条管"极短句独立成段制造戏剧效果"，这条管"短句有内容但内容是元话语"。也与 `PROSE.COPULA_DODGE` 相邻但不同：那条禁 "serves as" 替代 "is"，这条禁 "is + 抽象标签"这一整个句型。

## Check

**删除测试**：把这句删掉，段落是否损失任何信息？若否，它是预告，删掉或改写成承载内容的句子。

- **LLM 检查**:
  1. 句子主语是否为抽象名词（difficulty / tension / reason / issue / point / question / observation）
  2. 谓语是否为 be 动词
  3. 表语是否为抽象形容词或元描述（structural / subtle / simple / clear / precise / worth noting）
  4. 句子是否只描述"后面会讲什么"而非"是什么"
  5. 常见变体：`It is worth noting that...` / `This raises an important question.` / `The intuition is straightforward.`
- **排除**: 正式定义句（"A deletable unit is a designated time segment."）——那是在定义对象，不是贴标签

## Examples

### Pass

```latex
A CT model runs one shared vector field over every trajectory, and that single
field is where the difficulty comes from.

Stated formally, the tension is a claim about two subspaces. To first order, a
parameter edit preserves every retained output only if it lies in the null space
of the retained-output Jacobian.
```

机制直接出场；第二例的短句已经告诉读者"是两个子空间的事"。

### Fail

```latex
The difficulty is structural. A CT model applies one shared vector field
throughout every trajectory.

This can be made precise. To first order, a parameter edit preserves every
retained output only if it lies in the null space of the retained-output
Jacobian.
```

两个开头句删掉后信息零损失，它们只是在预告。

## Conflicts

与 `PROSE.RHYTHM_VARIANCE` 的张力裁决线：那条要求节内存在 <12 词的短句，本卡不与之矛盾——**短句应该存在，且必须承载主张**。修复本卡违规时用「改写成承载内容的短句」优先于「删除短句」，否则会把句长分布重新压平、触发 RHYTHM_VARIANCE。
