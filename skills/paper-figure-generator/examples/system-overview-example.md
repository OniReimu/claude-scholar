# System Overview 示例：RAG Framework

展示如何使用 paper-figure-generator 生成一个 Retrieval-Augmented Generation 系统总览图。

---

## 用户输入

> 帮我生成一张 RAG 系统的 Figure 1，展示从用户查询到生成回答的完整流程。系统包含 query encoder、向量检索、context builder 和 LLM generator 四个核心模块。

## Step 1: Analyze — 提取方法描述

从用户描述中提取：

- **组件**: Query Encoder, Document Index, Retriever, Context Builder, LLM Generator
- **关系**: 线性流 + 双输入到 Retriever
- **分组**: Retrieval Module (Query Encoder + Index + Retriever), Generation Module (Context Builder + LLM)
- **输入/输出**: User Query → Final Answer

推荐布局：`system-overview`（Figure 1 风格，适合展示整体框架）

## Step 2: Prepare — 创建 method.txt

创建 `figures/rag-framework/method.txt`:

```text
Retrieval-Augmented Generation (RAG) Framework

The system takes a user query as input and produces a grounded answer as output.

The Query Encoder transforms the user query into a dense embedding vector (d=768).
The Document Index is a pre-built vector database containing chunk embeddings from
the knowledge base.

The Retriever performs top-k similarity search over the Document Index using the
query embedding, returning the most relevant passages. The Query Encoder and
Document Index together form the Retrieval Module.

The Context Builder concatenates the retrieved passages with the original query
to create an enriched prompt. The LLM Generator produces the final answer
conditioned on this enriched context. The Context Builder and LLM Generator
form the Generation Module.

Data flows from left to right: User Query → Query Encoder → Retriever ← Document Index,
then Retriever → Context Builder → LLM Generator → Final Answer.
```

## Step 3: Setup — 检查安装

```bash
# 检查 AutoFigure-Edit 是否已安装
ls skills/paper-figure-generator/.autofigure-edit/autofigure2.py
```

## Step 4: Generate — 运行生成

```bash
bash skills/paper-figure-generator/scripts/generate.sh \
  --method_file figures/rag-framework/method.txt \
  --output_dir figures/rag-framework
```

输出:
- `figures/rag-framework/figure.png` — 光栅预览
- `figures/rag-framework/final.svg` — 可编辑 SVG

## Step 5: Finalize — SVG 转 PDF

```bash
uv run python -c "import cairosvg; cairosvg.svg2pdf(url='figures/rag-framework/final.svg', write_to='figures/rag-framework/figure.pdf')"
```

在 LaTeX 中使用:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/rag-framework/figure.pdf}
  \caption{Overview of the Retrieval-Augmented Generation framework.
  The system retrieves relevant passages from a document index and
  augments the LLM input with retrieved context.}
  \label{fig:rag-overview}
\end{figure}
```
