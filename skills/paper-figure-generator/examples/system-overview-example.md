# System Overview 示例：RAG Framework

展示如何使用 paper-figure-generator 生成一个 Retrieval-Augmented Generation 系统总览图。

---

## 用户输入

> 帮我生成一张 RAG 系统的 Figure 1，展示从用户查询到生成回答的完整流程。系统包含 query encoder、向量检索、context builder 和 LLM generator 四个核心模块。

## Step 1: Analyze — 提取结构化内容

从用户描述中提取：

- **组件**: Query Encoder, Document Index, Retriever, Context Builder, LLM Generator
- **关系**: 线性流 + 双输入到 Retriever
- **分组**: Retrieval Module (Query Encoder + Index + Retriever), Generation Module (Context Builder + LLM)
- **输入/输出**: User Query → Final Answer

## Step 2: Select — 推荐布局和风格

推荐选择:
- **布局**: `system-overview`（Figure 1 风格，适合展示整体框架）
- **风格**: `modern-gradient`（NeurIPS/ICML 风格，美观现代）

## Step 3: Structure — 创建 content.md

创建 `figures/rag-framework/content.md`:

```markdown
# RAG Framework - Figure Content

## Title
Retrieval-Augmented Generation Framework

## Components
1. Query Encoder - Transforms user query into dense embedding vector
2. Document Index - Pre-built vector database of document chunk embeddings
3. Retriever - Performs top-k similarity search over document index
4. Context Builder - Concatenates retrieved passages with original query
5. LLM Generator - Produces final answer conditioned on enriched context

## Connections
- User Query → Query Encoder (input)
- Query Encoder → Retriever (query embedding)
- Document Index → Retriever (document embeddings)
- Retriever → Context Builder (top-k passages)
- User Query → Context Builder (original query)
- Context Builder → LLM Generator (enriched prompt)
- LLM Generator → Final Answer (output)

## Groupings
- "Retrieval Module": Query Encoder, Document Index, Retriever
- "Generation Module": Context Builder, LLM Generator

## Annotations
- "Top-k passages" label on Retriever → Context Builder arrow
- "Dense embedding" label on Query Encoder → Retriever arrow
- "d=768" dimension annotation on embedding arrows

## Emphasis
- LLM Generator: largest block, central visual weight
- Retriever: second emphasis, key bridge component
```

## Step 4: Compose — 组装完整 prompt

将 base template + system-overview layout + modern-gradient style + content 拼接，保存为 `figures/rag-framework/prompt.md`。

## Step 5: Generate — 运行脚本

```bash
npx -y bun skills/paper-figure-generator/scripts/main.ts \
  --promptfiles figures/rag-framework/prompt.md \
  --output figures/rag-framework/figure.png \
  --ar 16:9
```

输出: `figures/rag-framework/figure.png`
