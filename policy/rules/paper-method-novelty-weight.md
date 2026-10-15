---
id: PAPER.METHOD_NOVELTY_WEIGHT
slug: paper-method-novelty-weight
severity: warn
locked: false
layer: core
artifacts: [text, figure]
phases: [writing-system-model, writing-methods, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [PAPER.METHOD_INSTANTIATION_SEPARATION]
constraint_type: guidance
autofix: none
---

## Requirement

Method 的篇幅跟着创新走。每一段先确定它的角色：

| 角色 | 是什么 | 应得的处理 |
|---|---|---|
| `problem-setup` | 输入、输出、目标、约束 | 写清楚，并且要**为新机制铺路**：点出这个问题里让标准方法失效的那个难点 |
| `novel-mechanism` | 撑起某个 `lead ≠ none` 主张的新设计 | 独立成段，可以是 Method 里最长的部分。必须写出三件事：它做什么、**为什么这样设计**（标准做法在这里为什么不行）、它在数据流里接在哪（输入从哪来、输出给谁） |
| `standard-component` | 沿用的已有技术 | 一句话加引用。只在本文偏离标准做法的地方展开，并说明偏离的理由 |
| `instantiation` | 实验取值、backbone、实现选择 | 搬到 Setup（`PAPER.METHOD_INSTANTIATION_SEPARATION`） |
| `evaluation-leak` | 结果、消融、数据集与 baseline 描述 | 搬回 Results / Setup |

**违规形态：**
- 新机制只有一句话，而标准组件或实现细节占了好几段。
- 新机制有描述，但没有"为什么这样设计"：读者知道它做了什么，不知道它为什么非做不可。
- **跳读测试不过**：只读 Method 各段首句和方法概览图的 caption，读者说不出哪一步是新的、为什么需要它。
- 问题定义给全了输入输出，却没有点出新机制要解决的那个难点，于是新机制出场时像是凭空冒出来的。

方法概览图应在视觉上区分新组件和标准组件，审稿人通常先看图、再读正文。

**不是删减许可。** 理解方法所需的标准组件仍然保留一句话加引用；实例化细节搬家，不删。本卡调整的是分配，不是总量。

## Rationale

审稿人读 Method 时最先要找的是：哪一部分是新的？为什么需要它？如果找这两个答案要先读完三段标准组件和一段超参，答案就会被淹没。另一种常见情况是新机制只写了一句话、细节却写得很全。这往往发生在多轮"补全复现细节"式的修改之后：每一轮都只往 Method 里加东西，于是篇幅比例一路倒向实现细节，论文越改越平庸。

设计理由是这里最容易缺的东西。"We use a gated aggregator" 告诉读者做了什么，"a mean aggregator lets one dominant client set the update direction, so the gate scales each client by its agreement with the median" 告诉读者为什么必须这样做。后者才是审稿人判断"这是不是真的创新"的依据。

## Check

- **前置**：`claim-architecture-review` 的 `spine.md` 已列出各主张及其 `lead`；新机制就是撑起 `lead ≠ none` 主张的那部分设计。
- **画像**：`uv run skills/claim-architecture-review/scripts/method_profile.py <paper>` 给出 Method 的总词数、段数、占正文比例和引用数。有范文画像（`architecture-review/exemplars.md`）时，对照范文的篇幅分配。没有范文时**不设固定阈值**，按下面的 LLM 判断执行。
- **LLM 检查**：
  1. 每段标 `method_role`，统计各角色所占的词数比例。
  2. `novel-mechanism` 段是否写了"做什么 / 为什么这样设计 / 接在数据流哪里"三件事？缺哪件，指出来。
  3. 跳读测试：只读各段首句和概览图 caption，能否说出哪一步是新的、为什么需要？
  4. `problem-setup` 是否点出了新机制要解决的难点？
  5. `standard-component` 是否展开超过一句话，且展开处不是本文的偏离？
- **结构执行**：`claim-architecture-review` 的 M-pass 产出各角色的篇幅分配和调整计划。"补写设计理由"是该 skill 允许提出的少数几个增补动作之一。

## Examples

### Pass

```latex
\noindent\textbf{Agreement-gated aggregation.} A mean aggregator lets
one client with a large update set the global direction, which is
exactly the failure Section~\ref{sec:problem} identifies under label
skew. We therefore scale each client update $\Delta_i$, produced by the
local step in Eq.~\eqref{eq:local}, by its cosine agreement with the
coordinate-wise median of all updates ...

\noindent\textbf{Local training.} Clients run standard FedAvg local
updates~\cite{mcmahan2017communication}.
```

### Fail

```latex
% 标准组件铺了三段、新机制一句带过、没有设计理由
\subsection{Local Training}
Each client trains a ResNet-18 with SGD ... [三段关于本地训练、
学习率调度与数据增强的细节]
\subsection{Aggregation}
We use a gated aggregator to combine the updates.
```

## Conflicts

- `PAPER.METHOD_INSTANTIATION_SEPARATION`：先由那张卡把实例化移出去，本卡再判剩下的篇幅怎么分配。
