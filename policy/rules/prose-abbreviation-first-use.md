---
id: PROSE.ABBREVIATION_FIRST_USE
slug: prose-abbreviation-first-use
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

所有缩写在首次出现时必须展开全称，格式为 `Full Name (ABBR)`。

规则：
1. **Abstract** 和 **Body** 各自独立展开一次（Abstract 中展开过的缩写，在 Body 首次出现时需再次展开）
2. 展开后，后续一律使用缩写形式
3. 通用缩写无需展开：e.g., i.e., etc., API, GPU, CPU, RAM, URL, HTTP
4. 不要在标题（section/subsection heading）中首次定义缩写

## Rationale

读者可能只读部分章节。每个独立阅读单元（Abstract vs Body）都需要自包含的缩写定义。这也是学术出版的标准要求。

## Check

- **LLM 语义检查**: 扫描全文，列出所有缩写（2-5 个大写字母的 token），检查是否在 Abstract 和 Body 中各有一次展开定义
- **常见遗漏**: FL (Federated Learning), MU (Machine Unlearning), DP (Differential Privacy)

## Examples

### Pass

```latex
% Abstract
Machine unlearning (MU) enables models to forget specific training data.

% Body - Section 1
Machine unlearning (MU) has attracted significant attention due to
privacy regulations. MU aims to remove the influence of target data
from a trained model.
```

### Fail

```latex
% Body - 首次出现未展开
MU has attracted significant attention. The goal of MU is to ...
```
