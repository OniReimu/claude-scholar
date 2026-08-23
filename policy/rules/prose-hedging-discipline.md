---
id: PROSE.HEDGING_DISCIPLINE
slug: prose-hedging-discipline
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: [PROSE.REGISTER_PRESERVATION, PROSE.SELF_UNDERMINING]
constraint_type: guidance
autofix: none
---

## Requirement

**动词强度必须匹配证据强度，双向执行。**

**方向 1 — 不 under-claim（hedge 过度）。** Hedging 词（may, might, could, possibly, potentially）仅用于：

1. **真正不确定的推测**: 尚无实验证据的假设
2. **泛化声明**: 超出实验范围的推论
3. **未来方向**: 可能的后续工作

**已有实验证据支撑的结论禁止 hedge。** 实验结果用确定性语言陈述。

**方向 2 — 不 over-claim（动词强于证据）。** 实证工作 *shows* / *provides evidence* / *improves ... by N*；它不 *prove* / *demonstrate* / *establish* / *confirm* / *guarantee* 普适真理。检查点：

- prove / demonstrate / establish / confirm / guarantee 出现时，结论是否真的被数学证明或穷尽验证？只有实验支撑的，降级为 show / provide evidence / improve by N
- 比较级主张（more robust / outperforms all）是否带数字、图表或引用锚点？没有则补锚点或收窄范围（"on these three datasets, our method matches or exceeds ..."）
- 模糊幅度（a large improvement）→ 数字或区间（"a 2--6\% improvement over the strongest baseline"），并注明 metric 与 comparator；比较对象领先者，不是 trivial baseline

**校准红线（防反向矫正）**：修方向 1 时不得制造方向 2 的违规。校准过的 hedge（suggests / is consistent with / we hypothesize / may indicate，用于真不确定的主张）是学术写作的**正确**形态，把 "the results suggest X" 改成 "the results prove X" 是制造 over-claim，不是修复。

## Rationale

过度 hedging 削弱论文说服力，让读者质疑作者对自己结果的信心；over-claiming 则直接给审稿人递刀（"prove" 一个只有三个数据集支撑的结论，reviewer 一句 "overclaimed" 就能压分）。两个方向是同一条轴——动词强度对证据强度的映射——只修一边会把文本推到另一边。

AI 生成文本两个方向都常见：默认语气倾向 demonstrate/significantly 的 over-claim，被提示"严谨一点"后又滑向 may/potentially 满篇的 over-hedge。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **LLM 检查（方向 1）**:
  1. Results section 中是否对有数据支撑的结论使用了 may/might/could
  2. Conclusion 中对已完成工作的总结是否过度 hedge
  3. Method section 中对设计选择是否不必要地用了 potentially/possibly
- **LLM 检查（方向 2）**:
  1. 逐个扫描 prove/demonstrate/establish/confirm/guarantee，核对证据类型（证明 vs 实验）
  2. 每个比较级主张（outperform/exceed/more robust/faster）是否有数字/图表/引用锚点在同句或紧邻句
  3. "significantly" 是否有统计检验支撑（p 值/检验名），没有则删或换具体幅度
- **排除**: 数学环境内的 prove/establish（定理证明是字面用法）；引用他人主张的转述

## Examples

### Pass

```latex
% Results - 有数据，不 hedge
The proposed method outperforms all baselines by at least 5.2\%.

% Discussion - 推测，合理 hedge
This improvement may stem from the regularization term, which could
prevent catastrophic forgetting in non-target classes.

% 比较主张带锚点、范围收窄
On these three datasets, our method matches or exceeds the strongest
baseline (Table 2).
```

### Fail

```latex
% 方向 1：有数据却 hedge
The proposed method may potentially outperform the baselines.
Our results could possibly suggest an improvement of 5.2\%.

% 方向 2：动词强于证据
We prove that our method significantly outperforms all prior approaches.
This demonstrates that our framework is universally superior.
```

## Conflicts

- `PROSE.SELF_UNDERMINING` 管与证据强度无关的情绪与自贬措辞（`unfortunately`、`merely`、`far from practical`），本卡管动词强度与证据强度的校准。校准正确的 hedge（`suggests` / `we hypothesize`）以本卡为准，不得因「听起来示弱」被改强——那是制造 over-claim
