---
id: PROSE.REGISTER_PRESERVATION
slug: prose-register-preservation
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: [PROSE.INFORMAL_VOCABULARY, PROSE.ABSTRACT_AGENCY, PROSE.VAGUE_QUANTIFIERS, PROSE.HEDGING_DISCIPLINE, PROSE.ELEGANT_VARIATION, PROSE.RHYTHM_VARIANCE]
constraint_type: guardrail
autofix: none
---

## Requirement

### 先给修复规则（这是决策程序，不是品味判断）

**当一个短语必须被替换时，用这篇稿子在别处已经使用的措辞。**

在触发本规则的九处实测案例里，**七处**的正确改法是某个更早草稿或另一节里已经存在的短语。先搜稿子再造词有三个后果：改法可核对（能指出来源）、`PROSE.ELEGANT_VARIATION` 顺带满足、agent 拿到的是检索任务而不是审美任务。

只有当这个概念全文别处都没出现过时，才自己造措辞。

### 再给禁令

**简化或压缩一个句子，永远不得降低它的语域。** 一个替换即使**准确**，只要语域低于原文，就是违规。

**本规则的执行范围是 diff，不是 document。** 判定输入是一次编辑 pass 的改前 / 改后文本；只审这次改动过的 span。未被改动的句子无论多口语都不归本规则管——那是作者的选择，归 `PROSE.INFORMAL_VOCABULARY`。

> 对全文做这一类扫描的实测结果是 **precision 0.00 / recall 0.00**（35 条命中，真阳性 0；同一文本里真实存在的 2 处缺陷一条没抓到）。全文扫描不仅无效，而且会在一个 session 内把整份 guardrail 弄成噪声然后被关掉。

### Drift 分类

第 1 类可以列表；第 2–5 类不能，而实测九处里它们占了六处。

1. **口语词汇**——`stuff`、`a lot of`、`kind of`、`for good`、`on heads`。黑名单唯一够得着的一类，九处里占三处。归 `PROSE.INFORMAL_VOCABULARY` 的词表兜底。
2. **短语动词顶替拉丁语源动词**——`drives off` 代 `deters`、`calls for` 代 `requires`、`pay for the checking` 代 `fund the checking`。**判据**：本领域是否已有一个单词动词在用？有就用那个。
3. **伪装成简化的精确度损失**——`pay peers` 代 `route micropayments to peers`、`too heavy a one` 代 `a sanction set too high`。结果不口语，只是**更含糊**。**判据**：说出替换掉的那个指称对象。说得出来，就把它放回去。
4. **谓语位置的比喻**——`institutions hold levers`、`RQ4 closes the loop`、`the literature has the right instinct`。与 `PROSE.ABSTRACT_AGENCY` 重叠，交叉引用而非重复判定。
5. **正式对象的对话式框架**——`the whole point is`、`we just`、`either way`、`so we also observe it directly`。按上下文裁决，其中部分是合法连接词。

### 排除（必须放行，每条都是实测假阳性）

| 模式 | 为什么合法 |
|---|---|
| `good state`、`good-state attempts`、`Good-state conditional dispute bound` | 已定义的技术术语（状态切片 $G_t$），不是评价性形容词 |
| `\big\|`、`\Big[`、`\bigl(` | LaTeX 尺寸宏；朴素的 `\bbig\b` 会把它们全吃掉 |
| 行首的 `and` / `so` / `also` | LaTeX 源码折行，不是句首连接词。**匹配前必须先接行** |
| `rules out` / `rule out` | 标准博弈论惯用语（`rules out profitable one-shot deviations`） |
| `cheap signal`、`cheap guess`、`costly effort` | 本文在 model 节自己定义的对比术语 |
| `plug in a measurable quantity` | 标准的组合性/模块化表述 |
| `clean union bound` | 既有数学惯用法 |
| `several`、`various`、`a number of` | 归 `PROSE.VAGUE_QUANTIFIERS`，**不得重复报告** |
| 校准过的 hedge（`suggests`、`is consistent with`） | `PROSE.HEDGING_DISCIPLINE` 拥有且**优先级更高** |
| 逐字引语与被引文献标题 | 从来不是本文作者的语域 |

## Rationale

**语域是编辑动作的属性，不是词的属性。**

不存在一个包含 `pay for the checking too` 的有限词表。这个短语是普通英语，正确，清晰。它之所以错，只是**相对于它替换掉的东西**、以及相对于周围稿件的语域而言。

其余每一条 PROSE 规则都能在孤立的句子上判定，这一条不能。信号是一个 **delta**，所以执行点必须是 diff。

**压缩在结构上就会把语域往下拽。** 压缩优化的是词数，而最便宜可砍的词恰恰是那些精确的多音节词，因为它们的大白话替代品更短：

- `route micropayments to peers`（5 词）→ `pay peers`（2 词）
- `a sanction set too high deters`（6 词）→ `too heavy a one drives off`（6 词，但名词没了）
- `permanently excluding`（2 词）→ `excluding … for good`（3 词，而且读起来像口语）

所以 drift 不是失误，它是这个优化目标的**预期输出**，除非语域被单独计分，否则每一次压缩 pass 都会复发。去术语化操作同理：把一个术语换成大白话释义，正是那个落进对话式英语的动作。`PROSE.RHYTHM_VARIANCE` 推的是同一个方向——拉高句长方差最省事的办法就是写短口语句。

**实证**（payment-free-MAS / ARGUS 手稿 §1，一次压缩 pass，968 → 728 词，2026-08-15）：九处语域违规全部渲染进 PDF，`PROSE.INFORMAL_VOCABULARY` 的五条 lint pattern 命中 **0/9**。其中 #2（`pay peers`）和 #6（`too heavy a one`）根本不口语，只是更不精确——任何以"口语词"为范围的规则永远看不见它们。

关键在于：这些违规是**正确执行其他规则的产物**。agent 被要求去掉术语墙并砍掉三分之一篇幅，它做的每一次单独替换都站得住脚，聚合起来是一次语域塌陷。

作者是靠通读抓到全部九处的，原话：「我知道是理解简单，但是好像不太像是 academic tone 啊，anti-ai 不代表要 avoid academic/professional 吧？」

## Check

### 无机械检测器（刻意的）

不要为本规则编写 `lint_patterns`。§2.3 已实测：35 条命中、真阳性 0、真实缺陷漏报 2/2。发布一个正则版本会在正确文本上每篇产生约 35 条警告并且仍然抓不到缺陷，这是让整份 guardrail 文件被禁用的最快方式。

### LLM 检查（diff 范围，每个改动 span 两问）

输入：一次编辑 pass 的改前与改后文本。对每个改变的 span：

1. **替换测试**：新措辞是否把一个领域术语、一个拉丁语源动词、或一个具名对象换成了日常释义？如果是，这个释义是**更精确**了，还是只是**更短**了？
2. **修复**：在本次 pass 未触碰的章节里搜索这个概念是怎么措辞的，优先采用那个措辞。只有当该概念全文别处都不存在时才自造。

**报告格式**（四列，缺一不可）：

```
original → replacement → suggested wording → source of the suggestion
```

写出 source 是这条发现可被核对的原因；没有 source 的建议只是另一次品味判断。

### 工作流闸门（关键）

**register check 未通过之前，不得报告词数或压缩百分比。**

在 ARGUS 那次 session 里，agent 在作者读到正文之前三次把「968 → 728 words, −25%」作为成功指标报告出来——**正是那个数字让这次 pass 看起来已经完成了**。数字先于质量出现，就会替代质量成为验收标准。

## Examples

### Pass

```text
% 改前 → 改后：语域保持，且措辞取自稿件其他章节
route micropayments to peers  →  route micropayments to peers（未改动）
a sanction set too high deters  →  a sanction set too high deters entry
% 未被本次 pass 触碰的句子，即使口语，也不由本规则判定
```

### Fail

```text
original: route micropayments to peers
replacement: pay peers
→ 第 3 类，精确度损失：micropayment 这个指称对象消失了

original: a sanction set too high deters
replacement: too heavy a one drives off
→ 第 3 类 + 第 2 类：名词被代词 one 顶替，deters 被短语动词顶替

original: permanently excluding a verifier
replacement: excluding a verifier for good
→ 第 1 类，口语惯用语

original: Institutional instruments impose a cost
replacement: Institutions hold levers
→ 第 4 类，谓语位置的比喻（并触发 PROSE.ABSTRACT_AGENCY）
```

## Conflicts

- `PROSE.INFORMAL_VOCABULARY` 拥有**词表层**（第 1 类）与**未改动文本**的判定；本规则拥有**改动 span** 的语域 delta。同一处命中时以本规则的报告为准，不重复计数
- `PROSE.ABSTRACT_AGENCY` 拥有比喻性谓语的通用判定；第 4 类交叉引用它，不重复立案
- `PROSE.VAGUE_QUANTIFIERS` 拥有 `several` / `various` / `a number of`，本规则不得重复报告
- `PROSE.HEDGING_DISCIPLINE` 拥有校准 hedge 的存废，**优先级高于本规则**：不得以"语域"为由删除或强化一个校准过的 hedge
- `PROSE.RHYTHM_VARIANCE` 要求句长有落差，而最省事的加落差方式是写短口语句——本规则是它的对向约束：**拉方差不得靠降语域**
- `PROSE.ELEGANT_VARIATION` 要求术语全文一致；本规则的修复规则（复用稿件已有措辞）天然满足它
- **非学术语域不适用**：博客/社交/newsletter 的语域刻意口语化，`writing-anti-ai` 已按语域分流；grant/fellowship 语域归 `grant-application-writing`
