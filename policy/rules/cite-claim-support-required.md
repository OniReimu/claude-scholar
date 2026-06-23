---
id: CITE.CLAIM_SUPPORT_REQUIRED
slug: cite-claim-support-required
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
lint_patterns:
  - pattern: "\\[CLAIM NOT VERIFIED\\]"
    mode: match
  - pattern: "\\[QUOTE NOT VERIFIED\\]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

每个挂在**具体事实性 claim**（关于 prior work / 数据 / 方法的断言）上的引用，必须在一个**可访问的来源 span** 中有可验证的支撑。按 claim 类型分流：

- **Direct quote（直接引语）**：必须有逐字原文 + page/locator。
- **Paraphrase（转述某文献的具体结论/数字/方法）**：必须有语义匹配的来源 span + page/section locator。
- **General-contribution（仅引文献的整体存在/主题，不挂具体断言）**：走 `<!-- policy:CITE.VERIFY_VIA_API -->` 的存在性/metadata 验证即可，不要求 span-level locator。

支撑缺失、不可访问或与原文冲突时，必须**弱化、替换，或标记未解决**：claim 用 `[CLAIM NOT VERIFIED]`，引语用 `[QUOTE NOT VERIFIED]`。这些 marker 与 `[CITATION NEEDED]` 同性质——进入 self-review / revision / camera-ready 前必须清零（lint 会命中未解决 marker）。

**Failure-safe（不可违反）**：

- 绝不编造逐字引语、页码或"implied by"式的伪支撑。
- 无法访问来源时，只允许降级为 general-contribution 引用并标 `[CLAIM NOT VERIFIED]`，不得把具体断言强行挂上去。
- 若发现被引文献其实不支撑该 claim，优先**指出作者真正想引的那篇**（suggest the real paper），而不是把错引硬推过去。

## Rationale

`CITE.VERIFY_VIA_API` 保证"这篇文献存在且 metadata 正确"，但保证不了"这篇文献真的支撑你这句话"——后者才是 reviewer 实际惩罚的失败模式（misattribution、over-claim、把不存在于原文的结论安到引文头上）。本规则把已存在于 skill 散文里的 claim-level 核查升格为可判定契约：语义判断走 self-review，未解决 marker 走硬 lint backstop。方法借鉴 RefChecker（MIT，cascade 抽取 + 多源 active verification）与 claim-faithfulness 审计范式（per-claim verbatim grounding）；本规则只编码方法，不复制其文本。规则本身**不**充当"能可靠验证论文"的 LLM 语义闸门——它只要求人/LLM 在 self-review 做核查、并对无法核实处留可被 lint 抓到的诚实 marker。

## Check

- **lint 强制（`policy/lint.sh`）**：`.tex` 中禁止存在未解决的 `[CLAIM NOT VERIFIED]` / `[QUOTE NOT VERIFIED]` marker（regex backstop，不做语义判断）。
- **self-review 语义协议**：逐条核查每个 direct quote 和每个挂具体断言的引用。对每条记录 `citation key + claim text + 类型(quote/paraphrase/general) + locator(page/§) + status`。
  - quote：比对逐字原文，确认 locator。
  - paraphrase：确认来源 span 语义匹配，记录 §/page。
  - 同时跑 cascade 抽取（regex/BibTeX → API → 仅对脏条目用 LLM）+ 多源核验（Semantic Scholar / OpenAlex / web）+ 撤稿/勘误检查（详见 `citation-verification` skill）。
- **通过标准**：成稿中零未解决 marker；每个具体断言都能指到一个真实、可访问、语义匹配的来源 span。

## Examples

### Pass

```latex
% 转述：来源 span + locator 已核实
Prior work reports a 40\% citation error rate in AI-generated
references~\cite[\S4.2]{liu2026integrity}.

% 无法访问来源时诚实降级 + 留 marker（草稿期）
Some surveys suggest the trend is accelerating~\cite{x2025survey}. [CLAIM NOT VERIFIED]
```

### Fail

```latex
% 把一个原文没有的具体数字安到引文头上（无 locator、无法核实）
\cite{vaswani2017attention} proves that residual streams leak 3.0 bits/token.

% 成稿仍保留未解决 marker（lint 直接命中）
The channel survives multi-turn use~\cite{anon2026covert}. [QUOTE NOT VERIFIED]
```
