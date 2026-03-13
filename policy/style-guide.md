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
Recently, ... has attracted significant attention.
With the rapid development of X, Y has become increasingly important.
```

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

```
With the rapid development of X, Y has attracted significant attention.
However, existing studies mainly focus on A and fail to address B.
To tackle this issue, this paper proposes C. Specifically, we formulate
D as E and develop F. Simulation results demonstrate that the proposed
method significantly improves G compared with existing approaches.
```

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
