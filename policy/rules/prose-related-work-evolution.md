---
id: PROSE.RELATED_WORK_EVOLUTION
slug: prose-related-work-evolution
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

Related Work section 按**研究脉络（intellectual evolution）**组织，不写孤立摘要式罗列。

正确模式：
```
Early studies focused on X [1,2]. However, these approaches suffered from A.
Subsequent work [3,4] attempted to address this by B.
Nevertheless, these methods still face C, which motivates our approach.
```

错误模式：
```
A did X [1]. B did Y [2]. C proposed Z [3]. D extended W [4].
```

每个子话题段落应展示：前人工作 → 局限 → 后续改进 → 剩余问题 的演进逻辑。

## Rationale

孤立摘要式的 Related Work 只是文献列表，不帮助读者理解领域演进和本文定位。按脉络组织能清晰展示 research gap，也是审稿人评判论文 positioning 的重要依据。

## Check

- **LLM 检查**:
  1. 是否存在连续 3 句以上 "X did A. Y did B. Z did C." 的罗列模式
  2. 段落间是否有演进逻辑连接（however, subsequently, despite, building on）
  3. 每个子话题是否以当前局限/gap 结尾，引出本文动机

## Examples

### Pass

```latex
Early approaches to machine unlearning relied on full model
retraining~\cite{cao2015}. However, retraining from scratch is
computationally prohibitive for large-scale models. To mitigate this
cost, Bourtoule et al.~\cite{sisa} proposed SISA, which partitions
training data into shards. Nevertheless, SISA still requires
retraining entire shards, limiting its efficiency when unlearning
requests are frequent.
```

### Fail

```latex
Cao and Yang~\cite{cao2015} proposed a statistical query-based unlearning
method. Bourtoule et al.~\cite{sisa} proposed SISA training.
Golatkar et al.~\cite{golatkar} used Fisher information for unlearning.
Sekhari et al.~\cite{sekhari} provided theoretical guarantees.
```
