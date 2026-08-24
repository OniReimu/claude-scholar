---
id: PROSE.RULE_OF_THREE
slug: prose-rule-of-three
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-intro, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {max_per_paragraph: 1, max_list_sentences_per_paragraph: 2, max_items_inline_short: 4, max_items_inline_long: 2}
conflicts_with: [PROSE.RESTATEMENT_DILUTION, PROSE.COMMA_OVERUSE, PROSE.COLON_LIST_OVERUSE, PROSE.ELEGANT_VARIATION]
constraint_type: guidance
autofix: none
---

> **ID 说明**：本卡的 ID 保留 `RULE_OF_THREE` 以维持引用稳定，但**适用范围已不止三项并列**——它管的是**并列列举的密度与重复**。三项并列只是其中一种形态。

## Requirement

### 先说反向护栏（这条最容易被过度执行）

**技术散文里列举是合法且常见的。** 一个方法确实作用于四种张量维度，就该说出那四种。**本卡不要求减少信息，只要求同一组信息不被列举两次、也不在一段里堆成墙。**

首选修法永远是 **命名 + 引用**，不是删项：第一次列举时给这组东西一个名字，之后引用那个名字。这同时满足 `PROSE.ELEGANT_VARIATION`（全文用同一术语）。

### 四条判据

1. **三项并列密度**：同一段落中 "X, Y, and Z" 式三项并列最多 1 次（`max_per_paragraph`）。人类偶尔用是自然的，同段反复出现是 LLM 指纹。
2. **同一集合不得枚举两次**（本卡最常触发的一条）：一组对象在首次列举后必须**获得名字**；后文再提及时引用该名字，不重列一遍。第二次枚举几乎不携带新信息，只是把第一次的内容换个措辞重放——那属于 `PROSE.RESTATEMENT_DILUTION` 的命题层复述，在列举层的表现就是本条。
3. **单个列表的内联项数**：
   - **短项**（每项 ≤3 词：动词、单名词、缩写）内联至多 `max_items_inline_short`（默认 4）项。`may expand, duplicate, reorder, or rescale` 是合规的——四个是真正不同的操作，拆出去反而伤可读性。
   - **长项**（每项 >3 词的名词短语）内联至多 `max_items_inline_long`（默认 2）项。更多则改 `enumerate`、拆句、或给集合命名。判据是**读者能否一口气持住**：四个多词名词短语内联就是一堵墙。
4. **段落级列表密度**：一段中携带内联列表的句子最多 `max_list_sentences_per_paragraph`（默认 2）句。阈值刻意给得宽——超过它通常意味着该段在"逐项交代"而不是"推进论证"。

## Rationale

三项并列（Rule of Three）是 LLM 生成文本的强烈信号，这是本卡最初的范围。但实测暴露了原范围的两个洞。

**洞一：只盯三项，四项从定义上逃逸。** 一段真实的方法学段落里有三个列表——3 项、4 项、4 项——只有一个落在"三项并列"的定义内，另两个不受任何规则约束。而机械层能报的只有 `PROSE.COMMA_OVERUSE`（四项列表自然带 ≥4 个逗号），那是**副作用命中**，报的是逗号不是列举，作者按逗号去修会改错地方。

**洞二：修法本身在生产问题。** `writing-anti-ai` 曾写着 "Rule of three: prefer two or four items" —— 这是从通用 anti-AI 语境继承的（四项且其中一项奇具体能破坏三段式指纹），但在学术散文里，照它执行会把三项列表改成**四项**，制造出更长的列举墙，并与本卡原有的"三个以上用 enumerate"直接矛盾。同一件事，规则卡与执行它的 skill 指向相反方向。

**真正的缺陷形态**是密度与重复，不是项数本身。一段七句、四句带列表、且其中两句列的是同一组对象，读者被迫两次解析同一个集合——第二次没有新信息。这类段落逐句读全部合格，问题只在把段落作为整体读时浮现。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

**不发 `lint_patterns`**：内联列表与普通并列结构在正则上无法区分（`A, B, and C` 既可能是列举也可能是从句串联），而项的"长短"判定需要语义。本仓库实测过判断类规则上正则的代价（precision 0.00），此处不重蹈。

**逐段判定，四步**：

1. 数本段有多少句携带内联列表 → 超过 `max_list_sentences_per_paragraph` 即候选
2. 对每个列表：项是短项还是长项？项数是否超过对应阈值？
3. **集合比对**：本段（及本节）是否已经列举过同一组对象？是则第二次改为引用首次给出的名字
4. 逐个候选问：**这个列表是在推进论证，还是在逐项交代？** 交代型的合并成一句概括 + 附录/引用

**排除**：`enumerate` / `itemize` 环境内的列表项；contribution 列表；定义中穷举的形式化集合（如 $\{a,b,c\}$ 的语言化重述）；表格单元格。

## Examples

### Pass

```latex
% 短项四项内联：四个操作确实不同，合规
Under this abstraction, NSO may expand, duplicate, reorder, or rescale
coordinates along a feature axis.

% 集合首次列举时命名，后文引用名字而不重列
We use \emph{feature axis} to denote an internal tensor dimension whose
coordinates represent computational features: CNN channels, Transformer MLP
hidden units, and attention value coordinates. The same principle governs
every feature axis in this sense.
```

### Fail

```latex
% 同一集合枚举两次：第二次除了"同一原理适用于全部"没有新信息
We use feature axis to denote an internal tensor dimension whose coordinates
represent computational features, such as CNN channels, Transformer MLP hidden
units, or attention value coordinates.
...
The same principle governs CNN channels, Transformer MLP dimensions, attention
value-output paths, and supported residual-stream symmetries.

% 长项四项内联：四个多词名词短语堆在一句里，读者持不住
The framework covers gradient-based attribution methods, perturbation-based
saliency estimators, concept-level activation probes, and counterfactual
explanation generators.
```

## Conflicts

- `PROSE.RESTATEMENT_DILUTION` 拥有**命题层**复述（同一主张说两遍）；本卡第 2 条是它在**列举层**的表现（同一集合列两遍）。同一处可能两条都成立，以本卡的"命名 + 引用"修法为准，不重复计数
- `PROSE.COMMA_OVERUSE`（单句 ≤3 逗号）会在四项列表上**副作用命中**——它报的是逗号数，不是列举结构。**先按本卡判定**：列表合规（短项 ≤4）则改用分号或重述以满足逗号规则，**不得为了降逗号数而删项**
- `PROSE.COLON_LIST_OVERUSE` 管冒号引出的内联编号列举（`we: (1)...(2)...`）；本卡管不带冒号的并列列举密度，两者形态不同不重叠
- `PROSE.ELEGANT_VARIATION` 是本卡首选修法的约束：给集合命名后，全文必须一致使用该名字，不得再换同义说法
