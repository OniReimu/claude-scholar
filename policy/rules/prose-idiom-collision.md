---
id: PROSE.IDIOM_COLLISION
slug: prose-idiom-collision
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [PROSE.INFORMAL_VOCABULARY, PROSE.ABBREVIATION_FIRST_USE]
constraint_type: guardrail
autofix: none
---

## Requirement

**任何技术短语，检查它是否与一个常用英文习语同形。** 同形时，改用无歧义的表述——即使技术义完全正确。

判据不是"这个短语对不对"，而是**审稿人第一遍会读成哪个意思**。技术义与习语义同形时，读者默认走习语义，因为那是更高频的解析。

**已知碰撞对**（每条都需要在使用前确认上下文能否消歧）：

| 技术短语 | 技术义 | 会被读成的习语义 | 无歧义写法 |
|---|---|---|---|
| `a fair bit` | 无偏随机比特（fair coin → fair bit） | 相当多 | `an unbiased random bit` |
| `a good deal` | 一份有利的合约 / 交易 | 大量 | `a favourable contract` |
| `on the order of` | 数量级 | 大约 | `of order $10^{-3}$` 或 `approximately` |
| `a number of` | 一个具体可数的数目 | 若干（模糊量词） | 给出数字 |
| `significant` | 统计显著 | 重要 | `statistically significant ($p<0.01$)` |
| `positive` | 正值 | 好的 / 有利的 | `positive-valued` 或 `nonnegative` |
| `by and large` | —（无技术义） | 大体上 | 该短语只有习语义，属 `PROSE.INFORMAL_VOCABULARY` 类 1 |

表格是起点不是全集。新领域会生出新的碰撞对（`fair`、`good`、`positive`、`regular`、`normal`、`sound`、`complete`、`tight`、`sharp` 这类日常形容词被术语化时最容易撞）。

## Rationale

这是一类**其他所有规则都覆盖不到**的缺陷：

- 不是**语域**问题——`a fair bit` 一点也不口语
- 不是**准确性**问题——在指"无偏比特"时它技术上完全正确
- 不是**缩写/定义**问题——`fair bit` 不需要定义，它的两个成分都是标准术语

它是**歧义**问题，而歧义的代价由审稿人承担：读者在第一遍读到 `A fair bit selects one fitted member per pair` 时，极可能解析为"相当多（的东西）选择了……"，然后卡住、回读、或者更糟——默默记下"这句话没读懂"。审稿意见里的"unclear"往往就是这么来的，而作者永远不知道是哪个短语造成的。

日常词被术语化是学术写作的常态（`fair`、`sound`、`complete`、`tight`），所以碰撞会持续产生。检查成本很低（一次自问），漏检成本是一条读不懂的关键句。

**实证**（CISU / ct_unlearning 稿件，NDSS 投稿，2026-08-17）：`A fair bit selects one fitted member per pair.` 这里 `fair bit` 指无偏随机比特，技术正确，但英文习语义是"相当多"。已改为 `An unbiased random bit`。这处缺陷不触发当时任何一条 PROSE 规则。

## Check

对每个由**日常词构成的技术短语**问三步：

1. **同形检查**：把这个短语单独拿出来，它在通用英语里有没有一个高频习语义？
2. **消歧检查**：如果有，紧邻的上下文（同句的定语、动词、单位、数学符号）是否足以把读者推向技术义？足够则可留。
3. **改写**：不足则换成无歧义表述——通常是把隐含的技术限定词显式写出来（`fair bit` → `unbiased random bit`）。

**检查范围**：`.tex` 正文与 caption。重点扫首次引入术语的位置（Introduction、System Model、算法描述）。

**提取正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。

**不做机械检测**：碰撞对是领域相关的开集，正则化会既漏又吵。这是一次低成本的自问，不是扫描任务。

## Examples

### Pass

```latex
% 显式写出技术限定词，习语义被排除
An unbiased random bit selects one fitted member per pair.

% 上下文足以消歧：单位与数学符号把读者推向技术义
The residual is of order $10^{-3}$, which is negligible relative to the
segment-level tolerance.

% 统计义被显式标注
The improvement is statistically significant ($p < 0.01$, paired test).
```

### Fail

```latex
% 技术义为"无偏比特"，但读者第一遍读成"相当多"
A fair bit selects one fitted member per pair.

% 技术义为"数量级"，读者读成"大约"
The error is on the order of the discretization step.

% 技术义为"统计显著"，读者读成"重要"
The improvement is significant across all three datasets.
```

## Conflicts

- `PROSE.INFORMAL_VOCABULARY` 类 1 收 `a fair bit` 的**口语量词义**（"相当多"）；本卡管它的**技术义与习语义同形**。同一字符串可能触发任一条，取决于作者的本意——作者本意是口语量词则归那条，本意是技术义则归本卡
- `PROSE.ABBREVIATION_FIRST_USE` 管缩写首次展开；本卡管的短语通常**不需要定义**，只需要换掉，两者不重叠
- `PROSE.INTENSIFIERS_ELIMINATION` 已禁 `significantly` 的空洞用法；本卡补的是 `significant` 在**统计义**下必须显式标注，否则与"重要"同形
