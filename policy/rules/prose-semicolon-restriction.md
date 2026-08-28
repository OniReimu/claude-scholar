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
conflicts_with: [PROSE.CAUSAL_CONNECTIVE, PROSE.COMMA_OVERUSE, PROSE.EM_DASH_RESTRICTION, PROSE.RHYTHM_VARIANCE, PROSE.SENTENCE_LENGTH, PROSE.SHORT_PUNCHY_FRAGMENTS, PROSE.THEATRICAL_SPLIT]
constraint_type: guardrail
autofix: none
lint_targets: "**/*.tex"
---

## Requirement

正文段落中禁止分号。修法**两级**，按语义关系选，不是只有断句一条路：

**① 句法改变**——把两个子句真正接成一句。可用 `PROSE.CAUSAL_CONNECTIVE` 已有的阶梯（**从属化** `Because A, B` / **关系从句** `, which ...`），以及**逗号 + 连接词**（`whereas` / `while` / `but` / `and`）。准入判据见 Check，四条同时成立才用。

**② 断句**——其余一律拆成两句，第二句首字母大写。

```
before:  ... the momentum buffer amplifies rather than averages out; with
         $\beta=0.9$ we therefore align refreshes to 500-step boundaries.
after:   ... the momentum buffer amplifies rather than averages out. With
         $\beta=0.9$ we therefore align refreshes to 500-step boundaries.
```

**`;` → 裸逗号是禁止的，`;` → 逗号 + 连接词不是。** 前者是 comma splice，本身不合语法，且只换了标点；后者引入了一个承载语义的连接词，是句法改变。本卡此前的措辞（"不是把分号换成逗号"）没有区分这两者，读起来像"一律只能断句"——那是错的，见 Rationale 末段。

## Rationale

分号把两个完整命题压进一个句子，读者要在一口气里处理两个主谓结构。学术散文的默认单位是句子；能用分号连的地方，几乎总能用句号断开，而断开之后每一句都可以被单独引用、单独质疑。

密度上，作者本人的 pre-GPT 已发表论文为 **2.62 / 千词**，当前 draft 为 **5.60 / 千词**（2.14x）；公开 arXiv 语料同期为 1.32 → 1.80（1.36x）。

**只给断句一条路会系统性制造新违规。** 一篇全稿去分号（47 处，39 处可测）的实测回归：拆出的第二子句词数中位数 **10 词**，**19/39** 短于 10 词，**8/39** 短于 8 词，**6/39** 前后两句皆短于 12 词。最差一例正是本卡直接产出的——

```
The point-estimate rule passes.  Interval equivalence does not.
        4 词                            4 词
```

而这恰好是 `PROSE.THEATRICAL_SPLIT` 点名的违规型（`One might expect X. It does not.`），**那张卡的规定修法就是用 `but` / `yet` / `, and` 合并回一句**。本卡此前的措辞把这条路堵死，于是两张卡直接对冲：这边逼你拆，拆完撞那边，那边叫你接回去。补上第 ① 级修法就是为了解开这个环。

⚠️ **前面两个密度数字不足以证明分号是 AI 痕迹，本卡也不以此为依据。** 两侧语料不可比（3 篇已发表 vs 8 份 draft，体裁不同），且本仓库今天已有四个"看起来有差别、盲评即塌"的信号。本卡是一条**作者选定的风格约束**（guardrail），不是一个被验证过的检测信号。若日后要把它升格为痕迹论据，必须先做盲评，标准与 `PROSE.ADHOC_COMPOUND_MODIFIER` 一致。

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

**用 ① 的四条准入判据**（同时成立才用，否则走 ②）：

1. 拆出来的**第二子句 < 10 词**；
2. 两句关系是**对比 / 让步 / 互补**，连接词承载真实语义，不是当胶水；
3. 断句会产生**连续短句**，或第二句读起来像回声碎片；
4. 合并后**逗号总数 ≤ 3**——这是与 `PROSE.COMMA_OVERUSE`（阈值 ≥4）的协同点。

第 4 条是实测逼出来的：一处两半句各带三项列表的句子合并后达 5 个逗号，必须退回 ②。没有这条，本卡只会把分号违规换成逗号违规。

**不适用 ①**：并列的规格罗列（两个 seed、两条配置项、两组统计量）。那类拆开更清楚，硬接反而糊。参考比例——实测 39 处里 **32 处**仍然走 ②，① 是少数情形，不是新的默认。

⚠️ **连接词必须按语义分布，不得统一。** 全篇统一成 `while` 只是把分号指纹换成 `while` 指纹，并制造新的均质化（撞 `PROSE.RHYTHM_VARIANCE`）。实测那 7 处回接的分布是 `whereas`×2 / `while`×2 / `but`×2 / `and`×1。**优先沿用稿件已有的同型句式**（style-guide 的 Do NOT Over-Correct §1），不要自造。

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

两个修法各自的适用面：

```latex
% 规格罗列 -> 断句（②）。两半各自完整，拆开更清楚。
The outer codebook quantizes each block to one of 256 centroids. The inner
codebook encodes the residual at 4 bits per dimension.

% 对比关系且第二子句只有 4 词 -> 逗号 + 连接词（①）。
% 断成 `The point-estimate rule passes. Interval equivalence does not.`
% 会直接生成 PROSE.THEATRICAL_SPLIT 的违规型。
The point-estimate rule passes, whereas interval equivalence does not.
```

## Conflicts

- `PROSE.COMMA_OVERUSE` 此前把分号列为首选修法（"优先考虑拆句或用分号"）。本卡生效后该修法只剩**拆句**一项，那条卡已相应改写。两条同时命中一个长句时，先拆句——拆完两条都消失
- `PROSE.SENTENCE_LENGTH` 同理：原文写"拆成多个短句，或使用分号连接两个独立子句"，后半句与本卡直接冲突，已删去。**一个 55 词的分号长句要拆成两句，不是加一个分号**
- `PROSE.SHORT_PUNCHY_FRAGMENTS` 与 `PROSE.THEATRICAL_SPLIT` 是**本卡批量执行时的下游违规**：只给断句一条路，会把长句拆成两个短句，第二个常常是 4–8 词的回声碎片。THEATRICAL_SPLIT 的规定修法（用 `but` / `yet` / `, and` 合并回一句）与本卡此前的措辞直接对冲——这就是补第 ① 级修法的原因。执行顺序：先按本卡定级，**用 ① 的那些不会走到那两条**
- `PROSE.CAUSAL_CONNECTIVE` 处理的是同一句法现象（两个独立子句粘在一起）的另一种标点形式。两卡的修法阶梯**对齐**：从属化 / 关系从句 / 断句是共用的，本卡另加逗号 + 连接词一档。两卡同为 `autofix: none`，理由相同——连接词的选择取决于语义关系，机械替换会改错意思，比留着原标点更糟
- `PROSE.RHYTHM_VARIANCE`：① 的连接词若全篇统一（例如一律 `while`），只是把分号指纹换成连接词指纹，并压低句长方差。按语义分布，并优先沿用稿件已有句式
- `PROSE.EM_DASH_RESTRICTION` 是同一类错误的另一半：那条禁 em-dash 但曾把"逗号插入语"列为合法替代，于是破折号挂着的尾巴换个标点原样留下。两条卡现在共用同一个判据——**替换标点不算修法，结构必须变**
