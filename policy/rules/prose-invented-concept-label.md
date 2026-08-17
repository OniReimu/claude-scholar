---
id: PROSE.INVENTED_CONCEPT_LABEL
slug: prose-invented-concept-label
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [ideation, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [PROSE.ELEGANT_VARIATION, PROSE.ABBREVIATION_FIRST_USE]
constraint_type: guardrail
autofix: none
---

## Requirement

不要把刚刚描述完的现象顺手封装成一个看起来像既有术语的复合名词（`the supervision paradox`、`workload creep`、`the alignment gap`、`the X effect`）。

带定冠词的概念标签只在两种情况下允许：

1. **有出处**——该术语在文献中已存在，首次出现处给引用（`the deployment gap~\cite{...}`）；
2. **是本文的命名贡献**——首次出现处显式声明命名动作（`we refer to this as ...` / `we call this ...`），给出可判定的定义，此后全文一致使用该名称，缩写遵守 `PROSE.ABBREVIATION_FIRST_USE`。

两者都不满足时，用普通语言把现象说清楚，不要给它起名。

## Rationale

模型被训练成"把观察提升为概念"，于是它在描述完一个现象后会自动生成一个术语化的标签，语气上暗示这是学界共识。论文里这会造成两种具体伤害。

第一是可信度：审稿人读到 `the supervision paradox` 会当作既有概念去检索，检索不到就会认为作者在冒充文献或在虚张声势——这类质疑一旦出现在 review 里，代价远大于那个标签带来的一点点简洁。

第二是解释被顶替：命名会让作者以为已经解释过了。`workload creep` 读起来像一个机制，实际上它只是给"负载随时间上升"换了个更贵的说法，读者拿到标签却没拿到原因、条件和边界。

真正的命名贡献要付三样成本——定义、证据、全文一致使用。这三样都在的时候命名是好写作；三样都不在的时候它只是修辞。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

逐处扫描形如 `the <修饰语> (paradox|effect|problem|gap|trap|dilemma|principle|law|phenomenon|tax)` 与 `<名词> (creep|drift|collapse|debt)` 的复合标签，对每一处问三问：

1. 有文献出处并已引用吗？
2. 首次出现处有显式命名声明和定义吗？
3. 全文是否一致使用（没有在别处换成同义说法）？

三问全否 → 违规，改写成对现象的直接描述。仅第 3 问否 → 不是本卡问题，交 `PROSE.ELEGANT_VARIATION`。

**检查范围**：`.tex` 正文，重点扫 Introduction、Discussion 与 Related Work——这三处是标签最容易滋生的地方。

## Examples

### Pass

```latex
We refer to this behaviour as \emph{cross-principal capability laundering}
(CPCL): an agent acquires a capability under one principal's authorisation
and exercises it under another's. CPCL is observable whenever the two
principals share a tool namespace, and Section~\ref{sec:eval} measures it
across four vendors.
```

```latex
Accuracy degrades as the deployment distribution drifts away from the
training distribution, and the gap widens by 3.1 pp per month.
```

### Fail

```latex
This is the supervision paradox: the more closely the operator monitors the
agent, the less the agent learns to recover on its own. Such workload creep
is a well-known constraint on autonomy.
```

## Conflicts

- `PROSE.ELEGANT_VARIATION` 要求术语全文一致——本卡管**该不该造这个术语**，ELEGANT_VARIATION 管**造了之后有没有一致使用**，先过本卡再过它
- `PROSE.ABBREVIATION_FIRST_USE` 管缩写首次展开；合法的命名贡献必须同时满足两卡
- `SOK.TAXONOMY_REQUIRED` 要求 SoK 给出 taxonomy，taxonomy 的类别名是**有意的命名贡献**，按本卡第 2 类处理（显式声明 + 定义 + 一致使用），不因本卡而回避命名
