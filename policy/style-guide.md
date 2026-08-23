# Author Writing Style Guide — 整体性写作风格身份

> **⚠️ MANDATORY**: 所有论文写作任务（ml-paper-writing, writing-anti-ai, paper-self-review, review-response, post-acceptance）开始前必须加载本文件。本文件与 `policy/rules/` 同级权威，不可跳过。

> **定位**: 本文件定义作者的**整体性写作风格身份**——偏好动词、句式模板、段落组织、叙事逻辑。这些是写作时需要整体浸入的风格特征，无法拆成单条 pass/fail 的 rule card。与之互补的是 `policy/rules/`，每条规则有明确的 pass/fail 判定标准。

> **来源**: 基于 2019-2022 年（Pre-GPT 时期）发表论文的风格分析，代表作者真实的学术写作指纹。

---

## 1. 核心风格定位

**Classical Engineering / IEEE-Style Academic Writing**

- Problem-driven：快速切入研究问题，不讲故事
- 技术精确：用数据和公式说话，不用修辞
- 格式克制：不滥用 bold/italic/列表
- 逻辑显式：过渡词清晰标记逻辑关系
- 结构一致：段落和章节遵循固定模板

---

## 2. 偏好动词（Preferred Verbs）

写作时优先使用以下动词，不用花哨替代词：

| 功能 | 偏好动词 |
|------|---------|
| 提出方法 | propose |
| 研究问题 | investigate, study |
| 分析 | analyze |
| 建模 | formulate |
| 证明 | prove |
| 评估 | evaluate |
| 展示结果 | demonstrate |
| 对比 | compare |

**禁用**: revolutionize, introduce groundbreaking paradigm, pioneer, spearhead

---

## 3. 标准学术短语模板

### 3.1 Background（背景）

```
X currently relies on A, which holds only when B.
Deployments of X are constrained by A: every additional B costs C.
```

> **注意**：背景句从一个可验证的具体事实开篇，不从趋势断言开篇。`Recently, ... has attracted significant attention` / `With the rapid development of X, ...` / `In recent years, ...` 属于 `PROSE.AI_LEXICON` 的 formulaic openers 零容忍条款，写出来即违规。

### 3.2 Limitation（现有工作局限）

```
However, existing studies mainly focus on A and fail to address B.
Nevertheless, these approaches suffer from ...
```

### 3.3 Research Gap（研究空白）

```
However, there is a lack of ...
To the best of our knowledge, ...
```

### 3.4 Method Introduction（方法引入）

```
In this paper, we propose ...
To address this issue, this paper proposes ...
To tackle this challenge, we design ...
```

### 3.5 Contribution Signals（贡献信号）

```
The main contributions are summarized as follows:
This paper presents ...
We provide ...
```

### 3.6 Transition Toolkit（过渡词工具箱）

| 功能 | 偏好用法 |
|------|---------|
| 指出局限 | However, ... / Nevertheless, ... |
| 回应问题 | To address this issue, ... / In regards to this issue, ... |
| 展开细节 | Specifically, ... / Therein, ... |
| 对比 | In contrast, ... / They range from ... to ... |
| 因果 | Therefore, ... / Thus, ... |
| 同时 | Meanwhile, ... |

---

## 4. 句式偏好

### 4.1 方法→结果句式

优先使用 "By doing X, we enable Y to achieve Z" 结构：

```
By modeling the interaction among miners as a repeated game,
the proposed framework allows agents to optimize their strategies.
```

### 4.2 被动语态（IEEE 风格）

被动语态是本风格的正常特征，不需要刻意避免：

```
The problem is formulated as ...
The system is evaluated through simulations.
```

### 4.3 人称偏好

优先使用 "This paper proposes..." 而非 "We propose..."。两者均可接受，但前者更常用。

### 4.4 句子长度

典型区间 **25-35 词**。超过 35 词时考虑拆句。

---

## 5. 段落与章节结构

### 5.1 Introduction 段落组织

| 段落 | 内容 |
|------|------|
| 第一段 | Technology trend → Importance → Application |
| 第二段 | Existing work → Limitation |
| 第三段 | Paper contribution introduction |

**开篇句要求**：三段的**形状**不变，但第一段第一句必须携带具体的张力或 gap，不能是泛化的趋势陈述。"Technology trend" 是这一句要达到的**效果**，不是它的写法——写法是给出一个可验证的结构性事实，让趋势由这个事实自己带出来（例："Tabular deep learning discards feature-type metadata."）。泛化开场白（`With the rapid development of X, ...` / `X has attracted significant attention` / `In recent years, ...`）被 `PROSE.AI_LEXICON` 的 formulaic openers 条款零容忍禁止，同时它也无法为第二段的 Limitation 提供落点。

### 5.2 五步叙事逻辑

所有论文遵循：

```
Background → Problem → Gap → Method → Evaluation
```

段落级模板：

```
Existing ... → However ... → To address this issue ... → In this paper, we ... → Results demonstrate ...
```

### 5.3 Canonical Paragraph Template

五步：现有工作 → gap → 本文动作 → 具体做法 → 证据。

```
Existing X-based schemes assume A and therefore treat B as fixed.
However, B varies with C once the system is deployed, which invalidates
the assumption and leaves D unaddressed. To tackle this issue, this paper
proposes E. Specifically, we formulate D as F and develop G. Simulation
results demonstrate that E reduces H from 0.42 to 0.31 under the same
communication budget.
```

> **注意**：第一句直接落在现有工作的具体假设上，不铺垫趋势；最后一句给数字，不给 `significantly improves`（`PROSE.INTENSIFIERS_ELIMINATION`）。

---

## 6. 结构模板

### 6.1 Abstract 五段式

| 部分 | 句数 |
|------|------|
| Background | 1-2 句 |
| Problem tension | 1-2 句 |
| Method shift | 2-3 句 |
| Technical highlights | 1-2 句 |
| Results and implication | 1 句 |

推进逻辑：context → limitation → solution → evidence

**Background 句要求**：五段式的**形状**不变，但开头那 1-2 句 Background 必须具体到能自己生成 gap——读完它们，读者应当已经看得出下一段 Problem tension 从哪里来。泛化背景句（"领域 X 受到广泛关注"、"随着 X 的快速发展"）被 `PROSE.AI_LEXICON` 零容忍禁止，且它本来也承担不了生成 gap 的功能：它没有给出任何可以被证伪的东西。

> **与 Farquhar 公式的关系（作者未决事项，不由 agent 代决）**
> `ml-paper-writing` Step 3 教的是 Sebastian Farquhar 的 achievement-first 五句 abstract 公式：先说 achieved（"We introduce/prove/demonstrate..."），再说 why hard、how、evidence、最亮的那个数字。它把 achievement 放在第一句，与本节 background-first 的推进顺序不同。
> **两者在一个条件下兼容**：只要开头的 Background 句携带的是具体张力（上一段的要求），本节的五段式就仍然成立——差别只是把同一批信息按什么顺序铺开，而不是写不写 tension。
> **未决**：面向 NeurIPS / ICML / ICLR 这类 top-ML venue 时，是否整体切换到 achievement-first，是作者本人的取舍（涉及个人风格指纹与 venue 期待的权衡），不在 agent 的权限内。在作者明确决定之前，按本节 background-first 写。

### 6.2 Contribution Section 格式

```latex
The main contributions are summarized as follows:
\begin{enumerate}
  \item We formulate ...
  \item We propose ...
  \item We evaluate ...
\end{enumerate}
```

### 6.3 Equation 三步解释

1. 介绍左侧（LHS）概念
2. 给出公式
3. 逐项解释右侧（RHS）各项

### 6.4 Related Work 组织

按研究脉络（intellectual evolution）组织，不写孤立摘要：

```
✓ Early studies focused on ... However, these approaches ...
  Subsequent work attempted to ... Nevertheless, these methods still ...

✗ A did X. B did Y. C proposed Z.
```

---

## 7. 名词偏好

技术抽象名词优先使用：

- framework, mechanism, strategy, scheme
- architecture, formulation, model

示例：
```
a game-theoretic framework
a decentralized scheduling mechanism
```

---

## 8. 格式克制原则

- **Bold**: 仅用于首次定义的核心概念
- **Italic**: 仅用于强调术语
- **Bullet list**: 正文段落用连贯散文，不用列表（Contribution section 除外）
- 保持模板原生格式，不额外添加装饰

---

## 9. 叙事特征总结

```
problem → model → solution → simulation
```

- 最小化叙事修辞（minimal storytelling）
- 技术精确优先于说服力
- 结构清晰优先于文采
- 数据和指标驱动评估
