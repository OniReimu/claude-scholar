# Base Prompt Template

图像生成 prompt 的组合模板。Claude 在 Step 4 (Compose) 阶段使用此模板将各部分拼接成最终 prompt。

---

## 组合规则

最终 prompt 由以下 4 部分按顺序拼接：

```
[1. Base Context]
[2. Layout Fragment]
[3. Style Fragment]
[4. Content Section]
```

每部分之间用空行分隔。

---

## 1. Base Context（固定）

所有学术图表共用的基础要求：

```
Generate a publication-quality academic figure illustration.
This is a conceptual/illustrative diagram (NOT a data chart or plot).
Requirements:
- Professional academic quality suitable for a top-tier conference or journal paper.
- All text labels must be clearly legible, properly sized, and spelled correctly.
- Use vector-style clean rendering (no photorealistic textures).
- Maintain consistent visual hierarchy: titles > component labels > annotations.
- Ensure sufficient contrast between text and background.
- No watermarks, logos, or decorative borders.
- The figure should be self-explanatory with minimal external context needed.
```

## 2. Layout Fragment

从 `layouts.md` 中选择对应布局的 "Prompt 片段" 部分，原样插入。

## 3. Style Fragment

从 `styles.md` 中选择对应风格的 "Prompt 片段" 部分，原样插入。

## 4. Content Section

由 Claude 根据用户描述生成的结构化内容：

```
Content specification:
- Title: {figure title}
- Components: {numbered list of modules/blocks with labels}
- Connections: {list of arrows/flows between components}
- Groupings: {which components belong to the same group}
- Annotations: {important labels, formulas, or notes on the figure}
- Emphasis: {which elements should be visually prominent}
```

---

## 宽高比规范

在 prompt 末尾追加宽高比说明：

```
Aspect ratio: {ratio} (e.g., 16:9 landscape, suitable for a full-width figure in a two-column paper).
```

常用宽高比:
- `16:9` — 全宽 figure（默认，适合大多数论文 Figure 1）
- `4:3` — 稍窄的 figure
- `1:1` — 正方形 figure（适合 comparison 或小图）
- `3:2` — 宽幅 figure

---

## 完整示例

```
Generate a publication-quality academic figure illustration.
This is a conceptual/illustrative diagram (NOT a data chart or plot).
Requirements:
- Professional academic quality suitable for a top-tier conference or journal paper.
- All text labels must be clearly legible, properly sized, and spelled correctly.
- Use vector-style clean rendering (no photorealistic textures).
- Maintain consistent visual hierarchy: titles > component labels > annotations.
- Ensure sufficient contrast between text and background.
- No watermarks, logos, or decorative borders.
- The figure should be self-explanatory with minimal external context needed.

Layout: System overview diagram for an academic paper (Figure 1 style).
Structure: Show the complete system as interconnected modules flowing left-to-right.
Each module is a distinct rounded rectangle with a clear label.
Use directional arrows to show data/information flow between modules.
Group related modules with dashed boundary boxes and group labels.
Include input on the left and output on the right.

Visual style: Modern gradient academic illustration.
Use soft gradient fills for modules (light-to-medium saturation).
Rounded rectangles with generous corner radius.
Subtle drop shadows for depth.
Clean sans-serif typography for all labels.
White or very light gray background.

Content specification:
- Title: Retrieval-Augmented Generation Framework
- Components:
  1. Query Encoder (transforms user query into embedding)
  2. Document Index (vector database of document chunks)
  3. Retriever (top-k similarity search)
  4. Context Builder (combines retrieved passages)
  5. LLM Generator (produces final answer)
- Connections:
  - Query → Query Encoder → Retriever
  - Document Index → Retriever
  - Retriever → Context Builder
  - Context Builder → LLM Generator → Answer
- Groupings:
  - "Retrieval Module": Query Encoder, Document Index, Retriever
  - "Generation Module": Context Builder, LLM Generator
- Annotations: "Top-k passages" on Retriever→Context Builder arrow
- Emphasis: LLM Generator as the central component

Aspect ratio: 16:9 landscape, suitable for a full-width figure in a two-column paper.
```
