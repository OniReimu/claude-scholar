---
id: FIG.HEATMAP_LABEL_ABBREVIATION
slug: fig-heatmap-label-abbreviation
severity: warn
locked: false
layer: core
artifacts: [figure]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

热量图（heatmap）遵循与表格同样的密度纪律。当行/列标签过长（典型是左侧的方法名/类别名太长）时：

1. **图内用简写**——轴标签换成短代号（如 `M1`、`FT`、`RAG`），不让长名字撑宽画布；
2. **caption 里给全称对照**——在 caption 中列出简写→全称的映射；
3. **保持单栏**——不为容纳长标签而拉成双栏 `figure*`，也不把字号压到不可读。

## Rationale

长标签会把 heatmap 的绘图区挤扁、迫使拉宽成双栏或缩小字号，破坏可读性。图内简写 + caption 定义全称，既保住单栏紧凑度又不损失信息，与 `FIG.SELF_CONTAINED_CAPTION`（caption 自包含）一致：caption 本就该承载读懂图所需的说明。与 `FIG.COLUMN_WIDTH_JUSTIFICATION`（图单栏优先）同源——热量图默认单栏，长标签不是上双栏的理由。

## Check

- **LLM 检查**：定位 heatmap（`make_heatmap`/`imshow`/`pcolormesh` 生成的图或 caption 含 "heatmap"）。核对：轴标签是否过长导致画布被拉宽或字号压缩；若用了简写，caption 是否给出全称映射；是否为长标签升成 `figure*` 双栏。长标签直接铺满且无 caption 对照，或为此上双栏，即违规。

## Examples

### Pass

```latex
% 图内简写，caption 定义全称，单栏
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{heatmap_transfer}
  \caption{Cross-method transfer. Rows/cols use short codes:
    FT (full fine-tuning), RAG (retrieval-augmented generation),
    ICL (in-context learning), LoRA (low-rank adaptation).}
  \label{fig:heat}
\end{figure}
```

### Fail

```latex
% 长标签直接铺满、为容纳它拉成双栏、caption 无对照
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{heatmap_transfer}
  \caption{Cross-method transfer results.}
  % 图内 y 轴："Full Fine-Tuning", "Retrieval-Augmented Generation",
  % "In-Context Learning", "Low-Rank Adaptation" —— 长名字撑宽画布
  \label{fig:heat}
\end{figure*}
```
