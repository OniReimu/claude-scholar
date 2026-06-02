---
id: FIG.SELF_CONTAINED_CAPTION
slug: fig-self-contained-caption
severity: info
locked: false
layer: core
artifacts: [figure, text]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
deprecated_by: writing-convention
---

> **⚠️ Deprecated**: 本规则作为写作层面的 guideline 保留参考，不再作为代码级硬约束检查。
> Caption 质量属于论文写作范畴，由 `ml-paper-writing` / `paper-self-review` skill 在 review 阶段覆盖。

## Requirement

Caption 的内容取决于图的类型，分两类处理。

### 非实验图（概念图 / 系统图 / 架构图 / 流程图 / pipeline）

Caption 必须自包含（self-contained），无需阅读正文即可理解。应包含三要素：

- **what** — 图展示什么
- **how** — 关键组件、数据流或设计意图（概念图没有"实验设置"，这里指读懂图所需的结构说明）
- **finding / intent** — 该图要传达的核心信息

### 实验图 / 表格（results figures & tables）

Caption 只负责说明**"这是什么"（what）**，保持简洁，不堆砌三要素。其余信息分流：

- **实验配置细节（how）** → 用 `threeparttable` 放进表格 footnote（`\begin{tablenotes}`），不写进 caption
- **实验结果描述与解读（finding）** → 写进正文 / takeaway（见 `EXP.TAKEAWAY_BOX`），不写进 caption

即：实验图表的 caption ≠ 三要素全塞。把"配置"下沉到 threeparttable footnote，把"结论"上浮到正文，caption 只保留身份标识。

## Rationale

非实验图的读者常先看图后读正文，self-contained caption 提升可读性。但实验图表若把 setup 和结论全塞进 caption，会让 caption 臃肿、与正文重复，且违反多数 venue 的紧凑排版习惯。threeparttable 让配置细节就近可查又不占正文，正文 / takeaway 负责解读，职责清晰。

## Check

- **判断图类型**: 概念 / 系统 / 架构 / 流程 → 非实验图；results / ablation / benchmark 图表 → 实验图表
- **非实验图**: caption 是否含 what/how/intent；是否只写一句空泛标识或"见正文"
- **实验图表**: caption 是否过度堆砌 setup 与结论；setup 是否应下沉到 threeparttable footnote；结论是否应在正文 / takeaway

## Examples

### Pass — 非实验图（concept / system diagram）

```latex
\caption{Overview of our framework. Raw inputs are encoded by a
shared backbone (left), routed through the gating module (center),
and aggregated by the consensus head (right) to produce the final
prediction. The dashed path denotes the optional refinement loop.}
% what: framework overview
% how: backbone -> gating -> consensus head; dashed = refinement loop
% intent: 让读者读懂整条数据流
```

### Pass — 实验表格（caption 只说 what，config 进 threeparttable，结论进正文）

```latex
% requires \usepackage{threeparttable}
\begin{table}[t]
\caption{Accuracy on CIFAR-10 and ImageNet.}   % 只说"这是什么"
\label{tab:main}
\centering
\begin{threeparttable}
\begin{tabular}{lcc}
\toprule
Method & CIFAR-10 & ImageNet \\
\midrule
Baseline & $93.2$ & $76.1$ \\
Ours     & $\textbf{95.8}$ & $\textbf{79.3}$ \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item Backbone: ResNet-50. Optimizer: SGD (lr=0.1, momentum=0.9),
200 epochs, batch size 256. Best per column in \textbf{bold}.
\end{tablenotes}
\end{threeparttable}
\end{table}

\textbf{Takeaway:} Ours improves over the baseline on both datasets,
with the largest gain (+3.2\%) on ImageNet.   % 结论在正文，不在 caption
```

### Fail — 实验图表 caption 塞满 setup + 结论

```latex
% 把 how + finding 全堆进实验表 caption
\caption{Accuracy comparison on CIFAR-10. We train ResNet-50
with SGD (lr=0.1) for 200 epochs. Our method (blue) outperforms
all baselines by 2.3\% on average.}
% setup 应进 threeparttable footnote；结论应进正文 / takeaway；caption 只留"这是什么"
```

### Fail — 非实验图 caption 太空

```latex
\caption{Our framework.}          % 概念图也至少要让人读懂结构
\caption{See text for details.}   % 依赖正文，无法独立理解
```
