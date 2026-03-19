---
id: PROSE.TENSE_CONSISTENCY
slug: prose-tense-consistency
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
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

各章节使用一致的时态：

| Section | 时态 | 示例 |
|---------|------|------|
| Related Work | 过去式 | "Smith et al. proposed..." |
| Method / System Design | 现在式 | "The algorithm computes..." |
| Results（报告数据） | 过去式 | "The accuracy reached 95.3%." |
| Results（讨论含义） | 现在式 | "This indicates that..." |
| Conclusion（总结贡献） | 过去式 | "We proposed a framework..." |
| Conclusion（未来方向） | 现在式/将来式 | "Future work includes..." |

同一段落内不应在无逻辑原因的情况下切换时态。

## Rationale

时态不一致是论文质量的常见扣分项。学术写作有明确的时态惯例，混用会让审稿人感到缺乏经验。

## Check

- **LLM 检查**: 逐章节验证时态一致性
- **关注点**: 同一段落内是否无故切换时态，Related Work 是否用了现在式描述已完成的工作

## Examples

### Pass

```latex
% Related Work - 过去式
Zhang et al.~\cite{zhang2021} proposed a federated unlearning framework
that removed target data through gradient ascent.

% Method - 现在式
The proposed algorithm first computes the influence function and then
removes the contribution of the target sample.
```

### Fail

```latex
% Related Work - 时态混用
Zhang et al.~\cite{zhang2021} proposes a federated unlearning framework
that removed target data. Their method achieves competitive results.
```
