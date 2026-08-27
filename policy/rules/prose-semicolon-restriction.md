---
id: PROSE.SEMICOLON_RESTRICTION
slug: prose-semicolon-restriction
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-intro, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: [PROSE.COMMA_OVERUSE, PROSE.SENTENCE_LENGTH, PROSE.EM_DASH_RESTRICTION]
constraint_type: guardrail
autofix: none
lint_targets: "**/*.tex"
---

## Requirement

正文段落中禁止分号。两个独立子句用分号连起来时，**拆成两句**。

```
before:  ... the momentum buffer amplifies rather than averages out; with
         $\beta=0.9$ we therefore align refreshes to 500-step boundaries.
after:   ... the momentum buffer amplifies rather than averages out. With
         $\beta=0.9$ we therefore align refreshes to 500-step boundaries.
```

不是把分号换成逗号——那是标点替换，不是修法。分号连接的两个子句各自独立成句，第二句首字母大写。

## Rationale

分号把两个完整命题压进一个句子，读者要在一口气里处理两个主谓结构。学术散文的默认单位是句子；能用分号连的地方，几乎总能用句号断开，而断开之后每一句都可以被单独引用、单独质疑。

密度上，作者本人的 pre-GPT 已发表论文为 **2.62 / 千词**，当前 draft 为 **5.60 / 千词**（2.14x）；公开 arXiv 语料同期为 1.32 → 1.80（1.36x）。

⚠️ **这两个数字不足以证明分号是 AI 痕迹，本卡也不以此为依据。** 两侧语料不可比（3 篇已发表 vs 8 份 draft，体裁不同），且本仓库今天已有四个"看起来有差别、盲评即塌"的信号。本卡是一条**作者选定的风格约束**（guardrail），不是一个被验证过的检测信号。若日后要把它升格为痕迹论据，必须先做盲评，标准与 `PROSE.ADHOC_COMPOUND_MODIFIER` 一致。

## Check

builtin（`lint_prose_semicolon_restriction`），不是纯 pattern。原因是纯正则会在数学记号上误报：`p(y \mid x; \theta)` 这类条件记号在 ML 论文里很常见，实测公开语料中**分号总数的 7–16% 位于行内数学内部**。builtin 先剥掉这些再判。

剥除顺序：

1. `\;`（LaTeX 细空格宏）——它不是标点；
2. 行内数学 `$...$`；
3. `algorithm` / `algorithmic` / `lstlisting` / `verbatim` / `tikzpicture` / `equation` / `align` 环境整块。

**豁免**：

- **列表项**（`itemize` / `enumerate` / `description` 内部）。以分号结尾的列表项不是"段落中间的分号"，是列表的分隔约定；
- `.bib` 文件与引用键；
- 直接引语、reviewer comment 原文、被批评的对象文本。

**报告已放行项**：报出每份文件被剥掉的数学分号数与列表分号数。只报违规，作者分不清"查过合格"与"没看到"。

## Examples

### Pass

```latex
The kernel spills to global memory at sequence length 4096. Throughput drops by
38\% at that point and stays flat beyond it.

The estimator is defined as $p(y \mid x; \theta)$ with $\theta$ fixed.
```

### Fail

```latex
The outer codebook quantizes each block to one of 256 centroids; the inner
codebook encodes the residual at 4 bits per dimension.
```

## Conflicts

- `PROSE.COMMA_OVERUSE` 此前把分号列为首选修法（"优先考虑拆句或用分号"）。本卡生效后该修法只剩**拆句**一项，那条卡已相应改写。两条同时命中一个长句时，先拆句——拆完两条都消失
- `PROSE.SENTENCE_LENGTH` 同理：原文写"拆成多个短句，或使用分号连接两个独立子句"，后半句与本卡直接冲突，已删去。**一个 55 词的分号长句要拆成两句，不是加一个分号**
- `PROSE.EM_DASH_RESTRICTION` 是同一类错误的另一半：那条禁 em-dash 但曾把"逗号插入语"列为合法替代，于是破折号挂着的尾巴换个标点原样留下。两条卡现在共用同一个判据——**替换标点不算修法，结构必须变**
