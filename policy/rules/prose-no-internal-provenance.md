---
id: PROSE.NO_INTERNAL_PROVENANCE
slug: prose-no-internal-provenance
severity: error
locked: false
layer: core
artifacts: [text, table, figure]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {drafting_severity: warn}
conflicts_with: [EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE, EXP.RESULTS_STATUS_DECLARATION_REQUIRED]
constraint_type: guardrail
autofix: assisted
lint_targets: "**/*.tex"
---

## Requirement

**先说 provenance 该去哪，再说正文不许写什么——只知道"禁止"的作者会照写不误，然后指望 review 抓。**

| 内容 | 归属 |
|---|---|
| 数字来自哪个文件的哪一行 | number-provenance ledger（仓库内部） |
| 复现所需的数据、脚本、列名 | released artifact（匿名仓库 / DOI） |
| 论文正文、caption、表格 | **引用 artifact 本身**，永不出现路径 |

一个数字的来源链是 `ledger → artifact → 论文引用 artifact`。论文是链条的终点，不是中间态的展示窗。

正文（含 caption、表格单元格、notation table、appendix）不得出现以下**七类**内部工作痕迹：

1. **内部产物名**——脚本名（`plot_fig3.py`）、内部预览路径（`figs/_preview.png`）、renderer/工具链名、DPI 自检备注、artifact bundle 备注。
2. **写作过程元话语**——`this draft`、`in the camera-ready version we will`、`(placeholder)`、`[TODO]` 残留。
3. **placeholder / 验证状态标记**——`[CITATION NEEDED]`、`[CLAIM NOT VERIFIED]` 等 workflow 标记，投稿/camera-ready 前必须全部清除或落实。
4. **数据来源路径**——把结果目录/数据文件当作所报数字的 provenance 写进论文（`\path{experiments/results/rc1/rc1.csv}`）。**发生频率最高，且对作者而言最像在守规矩**。与第 1 类的区别：第 1 类是图**怎么渲染出来的**，第 4 类是数字**从哪来的**。
5. **数据 schema 标识符**——CSV 列名、DataFrame key、config key、run ID，出现在正文、caption 或作为表格的一整列。判据：读者需要这个字符串**才能看懂主张**，还是只有**重跑我们的代码**时才用得上？后者即违规。
6. **内部 fixture / case 名**——golden-file 名、测试用例 ID、实验代号、论文从未定义过的 internal claim-register ID。**最阴险的一类**，因为它读起来像领域术语（"As Golden G4, ..."）。
7. **修订叙事**——预设了作者自己作品存在更早版本的陈述："the old bound is retracted"、"legacy notation superseded by"、"renamed to avoid collision with"。与第 2 类的区别：第 2 类关于**写作过程**，第 7 类关于**研究history**；多轮 audit 流程会大量产生第 7 类。

**例外（必须放行，否则检测器会吵到被关掉）**：

| 模式 | 为什么合法 |
|---|---|
| `\includegraphics{figures/...}`、`\input{sections/...}` | LaTeX 源码管道，不渲染进 PDF |
| `\label{}` / `\ref{}` / `\cite{}` 的 key | 不渲染 |
| `\url{https://anonymous.4open.science/...}` | artifact 链接是**要求**，不是泄漏 |
| 模型标识符：`gpt-4o`、`Llama-3.1-70B`、`claude-haiku-4.5` | 关于评测对象的科学事实 |
| 随机种子、超参、网格取值 | 可复现性事实，本就该进论文 |
| 长得像代码的领域术语："L1 commit"、数学意义的 "branch"、"kernel" | 术语 |
| "corrected coefficient"、"harness-level estimate" | 技术限定语，不是修订叙事 |
| `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` / `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` **要求**的状态披露 | 科学 claim 的一部分，那两条规则优先 |
| 论文主题本身就是 artifact evaluation / 审计方法学，需要实名引用内部工件 | 主题豁免 |

## Rationale

**根因是范式冲突，不是马虎——这决定了规则必须怎么措辞。**

一个有纪律的研究项目**要求**内部 provenance：number-provenance ledger、"没有 `file:line` 的数字不许进表"这类 gate。当 agent 在**同一个 turn 里**算完一个值、紧接着写下报告这个值的句子时，来源路径正活在它的工作上下文里，把它写出来的感觉是**在遵守项目主动奖励的美德**。

Agent 是在把一条内部规范导出到外部产物上。提示词用什么语言是次要的；触发条件是**同 turn 计算即写作**——agentic 流程每天都在做，人类作者极少这样写。

因此规则的第一句必须是"provenance 该放哪"，而不是"不许写"。给了搬运目的地的作者会合规；只被告知禁止的作者写完等 review。

**实证**（payment-free-MAS / ARGUS 手稿，ACM ASIA CCS 投稿，2026-08-15 扫描）：一篇已经过多轮 `writing-anti-ai` 打磨、十轮外部专家审计的 ~23 页稿件里，仍有 **11 处**开发痕迹渲染进了 PDF，其中一处是印在 evaluation 正文里的本地文件系统路径。全部清除后论文短了一页（24 → 23）。

**分布决定了扫描范围**：11 处里只有 **3 处在正文**，其余 8 处在 caption、notation table 和 appendix。任何只查正文散文的检查会漏掉三分之二——这就是 `R5`/`lint_targets` 覆盖全部 `.tex` 而非仅正文段落的原因。

本条是 `PROSE.*` 中唯一 `layer: core` 的规则：其余 PROSE 规则约束文风偏好，归 `domain` 层；本条约束投稿完整性/卫生底线（工作痕迹漏进正文即是硬伤，与 domain、venue 无关），归 `core`。

**severity 为什么是 error**：Rationale 自己管泄漏叫"硬伤"，却长期挂 `warn`，这个错配是它 11 次失效的一部分。submission / camera-ready 阶段按 `error` 执行；起草期允许通过 profile 覆盖降级为 `warn`（`locked: false`，`params.drafting_severity` 记录该默认值）。

（第 1–3 类改编自 DELONG-L/Academic-Paper-Skills 的 paper body versus audit trail 约定，MIT。第 4–7 类与检测器规格来自上述 ARGUS 扫描。）

## Check

### 机械检测（`policy/lint.sh` 内建，无需 LLM）

剥离注释、`\includegraphics{}`/`\input{}`/`\url{}`/`\label{}`/`\ref{}`/`\cite{}` 跨度，以及 EXP 披露 caption 之后，对 `.tex` 逐行匹配，输出 `file:line` 与命中片段：

| ID | 模式 | 置信度 | 覆盖类 |
|---|---|---|---|
| P1 | `\\path\{[^}]*\}` | 近乎确定——手稿里出现 `\path` 几乎必然是泄漏 | 4 |
| P2 | `(experiments\|results\|scripts\|src\|data\|notebooks)/[A-Za-z0-9_./-]+` | 高 | 4 |
| P3 | `\.(csv\|py\|jsonl\|json\|sh\|ipynb\|log\|pkl\|npz\|yaml)\b` | 高 | 1, 4 |
| P4 | `\\texttt\{[^}]*[a-z0-9]+\\?_[a-z0-9]+[^}]*\}`（`\texttt` 组内含 snake_case） | 高 | 5 |
| P5 | `retract\|supersede[sd]\|legacy\|no longer (used\|the original version)\|old (bound\|formula\|version)\|previously we\|earlier draft\|to avoid collision with` | 中——需人工裁决 | 7 |

P1–P4 属 `autofix: assisted`：工具无法知道替代措辞，但能精确指出位置。

### 未定义标识符子检查（LLM，两阶段）

第 6 类的一般化，也是本规则最有价值的部分。

1. **机械抽取候选**：渲染进 PDF 的正文中匹配 `[A-Z][A-Za-z]*-?[A-Z]?\d+\b` 的 token（`Golden-G4`、`C6`、`RC3`、`P7`、`V0`）。脚本：`policy/scripts/extract-undefined-identifiers.sh`
2. **LLM 裁决**：对每个不同候选，问——它在全文任何地方被定义过吗？定义、定理 caption、notation table 行、或首次出现处的显式解释，任一即可。
3. **报告未定义者**：读者无法解析的标识符，要么是内部泄漏，要么是真实的定义遗漏，两种都需要作者处理。

ARGUS 扫描中此检查同时抓到 `Golden-G4`（泄漏）与 `C6`（claim-register 标签，用了九次从未定义——一个真实缺陷）；`V0`–`V3` 因首次出现处有定义而正确放行。

### 联动

与 `CITE.CLAIM_SUPPORT_REQUIRED` 联动：其产生的 `[CLAIM NOT VERIFIED]` 标记在 submission gate 前必须清零。

## Examples

### Pass

```latex
Figure~\ref{fig:pipeline} shows the three-stage pipeline; exact
per-stage latencies are reported in Table~\ref{tab:latency}.
Replication data and analysis code are available at
\url{https://anonymous.4open.science/r/Argus-B943}.

\includegraphics[width=\linewidth]{figures/fig_f3_stats.pdf}
The Llama-3.1-70B arm uses seed 42; the L1 commit carries
weighted mass rather than the unweighted count.
```

### Fail

```latex
% 第 4 类：数据来源路径（正文）
Dispersion is reported in \path{experiments/results/rc1_pdisp/rc1_pdisp.csv}.

% 第 5 类：schema 标识符（caption）
\caption{Coverage by arm. Source columns: \texttt{empirical\_rate, empirical\_ucl}.}

% 第 6 类：未定义 fixture 名（appendix prose）
As Golden G4, at $\tilde r=0$ the substitution branch is inactive.

% 第 7 类：修订叙事（notation table）
The old refined general bound is retracted; legacy window-budget
notation is superseded by the per-round cap, and the variants were
named V0--V3 to avoid collision with the theorem labels C1/C2.

% 第 1--3 类：渲染 provenance + 元话语 + 残留标记
\caption{Pipeline overview (rendered by autofigure2.py at 300 DPI,
preview checked in figs/_preview.png). In this draft we use
placeholder numbers pending the final run. [CLAIM NOT VERIFIED]}
```

## Conflicts

- `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` / `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` **要求**在 caption 与小节声明里披露结果状态（simulated / projected / 非实跑）。那是科学 claim，不是工作痕迹——**两条 EXP 规则优先**，检测器必须放行其要求的披露文本
- `ANON.DOUBLE_BLIND_ANONYMIZATION` 管匿名性。一个会去匿名化的仓库 URL 是它的问题，不是本条的问题；本条只关心路径/schema/fixture/修订叙事是否泄漏
