---
id: EXP.MULTIRUN_AGGREGATE_CONSISTENCY
slug: exp-multirun-aggregate-consistency
severity: error
locked: false
layer: core
artifacts: [table, figure, text]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: none
---

## Requirement

凡是聚合了**多个实验 run** 的数字产物（结果表、对比图、正文中的跨 run 对比数字），必须产自**机器生成的 aggregate 工件**，不得从记忆、聊天记录或日志人工转录。合格的 aggregate 工件必须携带三类字段：

1. **每 run 的 validity 判定**（如 PASS / WARN / FAIL）——validity=FAIL 的 run 不得进入任何论文数字；
2. **跨 run must-match 一致性判定**：哪些实验设定字段（超参、底座、数据版本、评测协议）要求一致、实际是否一致、违规明细；
3. **provenance**：run 标识 + 上游产物代际标识（如 `base_build_id`）。

一致性判定为 INCONSISTENT 时只有两条出路：重跑对齐，或在 caption / 正文**显式披露差异**（例如 "Mistral uses r32/3ep while Llama uses r64/5ep due to ..."）。禁止用行文技巧把设定不一致的 run "写顺"成同质对比。

生成工具不限：参考实现为 `exp aggregate <family>.spec.json` → `paper_aggregates/<family>.aggregate.json`（适用于任何带 results sidecar 的目录树，本地或集群均可运行）；项目自有脚本亦可，只要输出含上述三类字段。结果表 `.tex` 中以注释记录来源：`% source: paper_aggregates/<family>.aggregate.json`。

## Rationale

真实事故类别：同一张表聚合的 run 后来被发现实验设定不一致（LoRA rank / epochs 因模型而异）、或混入了坏底座产物与 validity=FAIL 的结果——直到写作后期对数字甚至审稿阶段才暴露。`CITE.CLAIM_SUPPORT_REQUIRED` 保证「表 ↔ 源文件」的转录保真；本规则保证**源文件本身可信**（run 有效、互相可比、可溯源）。两层缺一不可：保真地转录一份被污染的源，产出的仍是错误的论文。

## Check

- **LLM 语义检查**：
  - 每个聚合多 run 的结果表 `.tex` 是否含 `% source: ...aggregate...` 来源注释（或等效的源工件引用）
  - 正文中的跨 run 对比数字能否追溯到某个 aggregate 工件
  - 若 aggregate 的 consistency verdict 为 INCONSISTENT，caption 或正文是否有对应差异的显式披露
  - 是否存在无任何源工件的"凭空数字"（疑似手工转录）
- **判定原则**：数字可追溯 + run 互相可比 + 不一致必披露，三者同时满足才 Pass

## Examples

### Pass

```latex
% source: paper_aggregates/v75_qwen35_main.aggregate.json (verdict: CONSISTENT)
\input{tables/tab_qwen35_main}
```

```latex
% source: paper_aggregates/crossmodel.aggregate.json (verdict: INCONSISTENT, lora_rank differs)
\caption{Cross-model comparison. Mistral uses r32/3ep while Llama uses
r64/5ep; per-base recalibration is disclosed in Appendix~C.}
```

### Fail

```latex
% no source artifact anywhere; numbers typed from a chat transcript
\begin{tabular}{lcc}
Method & Forget & Retain \\
ECO & 0.015 & 0.972 \\
\end{tabular}
```

```latex
% aggregate says INCONSISTENT (mixed lora_path), but caption presents runs
% as a homogeneous comparison with no disclosure
\caption{All methods evaluated under identical settings.}
```

## Conflicts

无直接冲突。与 `CITE.CLAIM_SUPPORT_REQUIRED` 互补（转录保真 vs 源可信）；与 `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` 正交（该规则管非实跑结果的披露，本规则管实跑结果的聚合资格）。
