---
id: EXP.SCOPE_EXCLUSION_CONSISTENCY
slug: exp-scope-exclusion-consistency
severity: warn
locked: false
layer: core
artifacts: [text, table, figure]
phases: [writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: lint_script
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: none
lint_targets: "**/*.tex"
coverage_note: "[A] pairs are located, not judged: each pair needs a verdict (consistent / contradiction / scope too broad). Not enumerable: -- and N/A cells, implicit inclusion (all ten schemes without naming the excluded one), statements with no extractable object (listed for hand check), and prose totals against table column sums. [B1]-[B3] are mechanical."
---

## Requirement

每一句**否定范围陈述**都必须与报告**同一对象**的每一张表、每一张图、每一个正文计数一致。否定范围陈述包括：not supported / unsupported、excluded、reported for X only / no X claim、abstains、does not apply、outside the scope、not evaluated、N/A，以及表格里的 `--`、N/A 和带脚注的例外格。

1. **排除在生成器里实现**：一个对象被排出某个总体（不做 ownership 结论、不支持某种编辑），排除必须写进产出该计数的生成脚本，并记录原因；表格、正文、摘要里的数字都引用重新生成的输出。禁止只在附录写一句「X 仅报告 recovery」而让表格和汇总数字照旧把 X 算进去。
2. **汇总计数处处一致**：同一总体的计数（`k of n`、`k/n`、`p% of the n <noun>`）在 abstract、intro、正文、conclusion、appendix（含 extended appendix）之间一致，并与表格的合计一致；百分比在所示精度下等于 `100k/n`。
3. **范围改动是整篇的改动**：任何一处排除或范围措辞改了，所有报告该对象或该总体的位置必须在同一次修改里重读、重算；只看 diff 的 review 不算核查。

## Rationale

真实事故（一篇白盒水印论文，2026-11）：附录写明某个方案在某个 LLM 上「只报告 recovery，不做 ownership 结论」，但 ownership 主表和 abstract / intro / conclusion 的汇总数字仍把这些 run 算在内。每个数字都有合法的 file:line 来源，错的是**总体**。同一篇里，一张表的脚注说某方案在 residual duplication 下「not supported」，而附录的一个 adaptive arm 照样给它打分并接受；extended appendix 的一段在表格重新生成后（33 / 51）仍保留旧的汇总数（32 / 50）。三次都是 diff-scoped review 漏掉的：没人改的表，没人拿已声明的规则去重读。

`EXP.MULTIRUN_AGGREGATE_CONSISTENCY` 保证来源工件可信，`CITE.CLAIM_SUPPORT_REQUIRED` 保证转录保真；本规则管的是第三层：**陈述的范围与被计数的总体是同一个总体**。来源可信、转录无误的数字，仍可能数错了对象。

## Check

- **lint（`bash policy/lint.sh --rule EXP.SCOPE_EXCLUSION_CONSISTENCY <dir>`）**，实现见 `policy/scripts/scope-consistency.pl`：
  - **[A] 定位器**：抽出每句否定范围陈述（含 caption 和 tablenotes），取其中的命名对象（`\textit{}` / `\texttt{}`、带数字的标识如 `Qwen2.5-1.5B`、驼峰名如 `HufuNet`），再找其他位置报告该对象数字的表格行和正文句子，按「同文件出现范围词 + 共享对象 + 共享内容词」排序，每句陈述打印前 5 对（`SCOPE_TOPN=0` 打印全部），其余文件列在信息行里。**只定位，不判定**：大多数配对本来就一致。
  - **[B1]** 同一个有定冠词的总体（`the 1,200 attacked configurations`、`k of the n <noun>`）在不同文件里分母不同。同一文件内的不同子集不报。
  - **[B2]** 紧挨 `k/n` 或 `k of n` 的百分比在所示精度下不等于 `100k/n`（接受四舍五入和截断）。
  - **[B3]** 正文 `k of the n`（n ≥ 1000，`SCOPE_B3_MIN_DEN` 可调）与使用同一分母的表格格子对不上分子。
  - regex 无法覆盖的部分（`coverage_note` 已声明）：`--` / N/A 格、隐式包含（「all ten schemes」没点名被排除者）、抽不出对象的陈述（lint 列出位置，逐条手查）、正文总数与表格列和的比较。
- **paper-self-review 的判定协议**：对 [A] 的每一对写一行裁决，三选一，**不许整批放行**：
  - **consistent**：报告位置不在被排除的总体里（例如 recovery 表报 recovery），或已按排除重算（写出依据：表注、生成器参数）。
  - **contradiction**：报告位置把被排除对象算进了声明它被排除的那个总体。处置：改生成器、重生成、更新所有引用该输出的数字。
  - **scope too broad**：陈述比实际做的更宽（例如说「not supported」而附录实际打了分）。处置：收窄陈述到真实的排除范围，或真的排除。
  - [B1]-[B3] 每条要么改到一致，要么说明两个数字属于不同总体并让名词短语把区别说出来。
  - 修改任何排除、范围措辞或汇总数后，**重跑整个目录**，不只看 diff。

## Examples

### Pass

```latex
% 生成器排除 riga21 的 Qwen run 并记录原因；表注、正文都引用同一输出
\item[b] Ownership counts exclude \textit{riga21}, whose Qwen runs show recovery only.
% abstract 与 appendix 用同一个总体
... restores ownership verification in 99.9\% of the 1,170 attacked configurations.
Of the 1,170 attacked models, 29 remain accepted before recovery.
```

### Fail（陈述排除了对象，计数却没排除）

```latex
% apdx: 只报告 recovery，不做 ownership 结论
there we report \textit{riga21} for recovery only and make no ownership claim for it.
% ownership 主表仍把这些 run 计入 120/120
riga21 & 120/120 & 11/120 & 8/120 & 120/120 \\
% 解决方式：在生成器里排除，重生成表格与所有汇总数
```

### Fail（汇总数漂移）

```latex
% 表格重生成后为 51/10160，extended appendix 仍写旧数
Under Tier-2, 50 of the 10,160 evaluation runs are accepted.
```

## Conflicts

无直接冲突。与 `EXP.MULTIRUN_AGGREGATE_CONSISTENCY`（来源可信）和 `CITE.CLAIM_SUPPORT_REQUIRED`（转录保真）互补：本规则管陈述的范围与计数的总体是否一致。
