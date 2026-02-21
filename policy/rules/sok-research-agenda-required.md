---
id: SOK.RESEARCH_AGENDA_REQUIRED
slug: sok-research-agenda-required
severity: error
locked: false
layer: domain
artifacts: [text]
phases: [writing-conclusion]
domains: [security, se, is]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

SoK 论文在 Conclusion/Discussion 末尾必须提出 research agenda（未来研究议程），至少覆盖：

1. 开放问题（open problems）
2. 关键挑战（technical barriers / evaluation gaps）
3. 可执行方向（actionable future directions）

## Rationale

高质量 SoK 不仅总结已有工作，还应定义下一阶段研究路线图，帮助社区形成可执行的问题清单。**本规则仅在 SoK profile 激活时生效**。

## Check

- **LLM 语义检查**:
  - 是否存在明确“future agenda / open problems”段落
  - 是否不是泛泛而谈（如“未来可以继续研究”）
  - 是否给出与 taxonomy/limitations 对齐的具体方向
- **通过标准**: 审稿人可以据此提炼出可独立立项的问题

## Examples

### Pass

```latex
\paragraph{Research Agenda.}
We identify three priorities: (1) standardized deletion verifiability benchmarks,
(2) threat-model-complete evaluations beyond honest-but-curious settings, and
(3) composable guarantees across cross-silo and cross-device regimes.
```

### Fail

```latex
In future work, we plan to improve this line of research.
% 过于笼统，没有 open problems / challenges / actionable directions
```
