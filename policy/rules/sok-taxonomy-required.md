---
id: SOK.TAXONOMY_REQUIRED
slug: sok-taxonomy-required
severity: error
locked: false
layer: domain
artifacts: [text]
phases: [writing-background]
domains: [security, se, is]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

SoK 论文必须在 Background/Related Work 中给出明确的 taxonomy（分类体系），至少包含：

1. 分类维度（classification dimensions）
2. 各类别定义与边界（category definitions + boundaries）
3. 代表性工作到类别的映射逻辑（mapping rationale）

## Rationale

SoK 的核心贡献是“系统化组织知识”，taxonomy 是该贡献的最小结构单元。没有 taxonomy 时，综述容易退化为松散的逐篇罗列。**本规则仅在 SoK profile 激活时生效**。

## Check

- **LLM 语义检查**:
  - 是否存在显式 taxonomy（而非普通 related work 叙述）
  - 是否定义分类维度与类别边界
  - 是否说明代表性工作的分类依据
- **通过标准**: 读者可据此判断一篇新工作应归入哪个类别

## Examples

### Pass

```latex
\subsection{Taxonomy of Federated Unlearning}
We organize prior work along two dimensions: (D1) unlearning trigger granularity
(sample/class/client) and (D2) guarantee type (exact/approximate/certified).
Table~\ref{tab:sok-taxonomy} maps representative methods to these categories.
```

### Fail

```latex
\subsection{Related Work}
Paper A studies unlearning in federated settings. Paper B improves efficiency.
Paper C discusses security risks. Paper D proposes a new benchmark.
% 仅逐篇罗列，缺少 taxonomy 维度、类别定义与映射逻辑
```
