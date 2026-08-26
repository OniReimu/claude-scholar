---
id: PROSE.VAGUE_QUANTIFIERS
slug: prose-vague-quantifiers
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: [PROSE.AI_LEXICON, PROSE.INFORMAL_VOCABULARY, PROSE.REGISTER_PRESERVATION]
constraint_type: guardrail
autofix: assisted
lint_patterns:
  - pattern: "\\b([Ss]ome|[Mm]any|[Ss]everal|[Nn]umerous|[Vv]arious) (studies|works|papers|methods|approaches|techniques|baselines|researchers|authors|applications|domains|scenarios|settings|datasets)\\b"
    mode: match
  - pattern: "(?i)\\b(a number of|a large amount of|a great deal of|plenty of|a wide range of|a variety of)\\b"
    mode: match
  - pattern: "\\b([Ee]xtensive|[Cc]omprehensive|[Tt]horough) (experiments|evaluations?|ablations?|analys[ei]s)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止使用无数据支撑的模糊量词。用具体数字或引用替代。

| 禁用 | 替代 |
|------|------|
| some researchers | Zhang et al.~\cite{} and Li et al.~\cite{} |
| many studies | over 30 studies (surveyed in~\cite{}) |
| several baselines | five baselines |
| a number of | 具体数字 |
| a large amount of | 具体数字 + 单位 |
| a wide range of | 具体范围 |
| extensive experiments | 直接列出数据集/设置（"three datasets (ImageNet, CIFAR-100, iNaturalist)"） |

**Pattern 只抓「量词 + 文献/方法类名词」组合与恒模糊短语**，不抓裸词。`some`/`many`/`several` 的裸词用法大量出现在合法语境——数学存在量词（"for some $\epsilon > 0$"）、固定搭配（"in many cases"）——裸词匹配的误报会淹没真命中。`a lot of` / `lots of` 归 `PROSE.INFORMAL_VOCABULARY` 管辖，本卡不重复收录。

## Rationale

模糊量词在学术写作中削弱精确度。审稿人会质疑 "many" 到底是多少。用数据说话是技术论文的核心原则。

"extensive/comprehensive experiments" 是同一问题的实验章节变体：AI 用形容词顶替实验清单。列出数据集名称与数量后，形容词自动多余。

## Check

- **regex 搜索**: 三组 pattern（量词+名词组合 / 恒模糊短语 / 实验形容词）
- **排除合法用法**: 引用了具体数据源的量词（如 "several studies~\cite{a,b,c}" 后随 3+ 个 citation key）可接受
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
We compare against five state-of-the-art baselines~\cite{a,b,c,d,e}.
Over 30 studies have investigated federated learning privacy (surveyed in~\cite{survey}).
```

### Fail

```latex
Many studies have investigated this problem.
We compare against several baselines.
```
