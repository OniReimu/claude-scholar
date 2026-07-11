---
id: PROSE.PSEUDOCODE_ABSTRACTION
slug: prose-pseudocode-abstraction
severity: warn
locked: false
layer: domain
artifacts: [code, equation]
phases: [writing-methods, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

伪代码/算法块必须保持在数学抽象层级，不得堆砌冗长自然语言步骤。当正文或方法章节已经建立了数学 notation（符号、算子、集合、函数），伪代码须复用这套 notation 表达，而不是把同一操作再用大段文字重述。具体：

1. **抽象掉标准操作**：写 `θ ← θ − η∇L(θ)` 或 "Train with SGD"，而不是 10 行梯度更新细节。
2. **复用已建立的符号**：变量名在数学式、伪代码、正文三处保持一致；不在伪代码里另起一套白话名字。
3. **一行一语义**：每步是一个数学化的操作，而非一句解释性散文。

## Rationale

论文的伪代码是给已经读过 notation 的读者看的形式化摘要，不是教程。前面刚定义完符号，伪代码却退回自然语言，既冗长又制造符号与文字两套指称、增加读者对齐成本，也是 AI 生成伪代码的典型特征（把每步展开成完整英文句子）。数学抽象让核心贡献（哪一步是新的）一眼可辨。与 `PROSE.CRYPTO_CONSTRUCTION_TEMPLATE`（security 域的 Construction 写法）方向一致，但本条适用所有领域、聚焦"复用已建立 notation"。

## Check

- **LLM 检查**：定位 `algorithm`/`algorithmic`/`lstlisting` 环境。若正文/方法已定义数学 notation，核对伪代码是否复用该符号；步骤是否被写成完整自然语言句子（而非符号化操作）；是否展开了本应抽象的标准子过程（梯度更新、排序、标准库调用）。整块几乎无数学符号、全是英文散文即违规。

## Examples

### Pass

```latex
% 复用已建立的 notation：策略 π_θ、奖励 R、数据集 D
\begin{algorithmic}[1]
\State \textbf{Input:} dataset $\mathcal{D}$, policy $\pi_\theta$, step size $\eta$
\For{$t = 1 \dots T$}
  \State sample minibatch $B \sim \mathcal{D}$
  \State $\theta \gets \theta - \eta \nabla_\theta \mathcal{L}(\pi_\theta; B)$
\EndFor
\State \Return $\pi_\theta$
\end{algorithmic}
```

### Fail

```latex
% 前文已定义 π_θ、L、D，此处却全用散文重述、展开标准 SGD 细节
\begin{algorithmic}[1]
\State First, we take the whole training dataset and shuffle it randomly.
\State Then, for each example, we compute the model's prediction and compare
       it with the ground truth label to get an error value.
\State Next, we calculate how much each weight contributed to this error by
       taking derivatives, and we subtract a small fraction of that gradient
       from every weight so the model improves a little.
\State We repeat this whole procedure many times until the model converges.
\end{algorithmic}
```
