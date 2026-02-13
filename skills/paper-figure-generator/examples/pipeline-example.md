# Pipeline 示例：数据预处理流程

展示如何使用 paper-figure-generator 生成一个多阶段数据处理 pipeline 图。

---

## 用户输入

> 画一个 NLP 数据预处理 pipeline 的图，从原始文本到模型输入，包括分词、清洗、embedding、data augmentation 几个阶段。用 clean minimal 风格。

## Step 1: Analyze

- **组件**: Raw Text, Tokenizer, Cleaner, Embedding Layer, Data Augmenter, Model Input
- **关系**: 严格线性流，每阶段输出下一阶段输入
- **分组**: 无明显分组，纯线性 pipeline
- **数据变换**: Text → Tokens → Clean Tokens → Embeddings → Augmented Embeddings → Batched Tensors

## Step 2: Select

- **布局**: `pipeline`（多阶段处理链，最佳匹配）
- **风格**: `clean-minimal`（用户指定，Nature/Science 风格）

## Step 3: Structure — content.md

```markdown
# NLP Data Preprocessing Pipeline - Figure Content

## Title
Text Preprocessing Pipeline

## Components
1. Raw Text - Unprocessed input documents and sentences
2. Tokenizer - Splits text into subword tokens (BPE/WordPiece)
3. Text Cleaner - Removes noise, normalizes unicode, handles special chars
4. Embedding Layer - Maps token IDs to dense vector representations
5. Data Augmenter - Applies augmentation (synonym replacement, back-translation)
6. Model Input - Final batched tensor ready for training

## Connections (left to right)
- Raw Text → Tokenizer: "sentences"
- Tokenizer → Text Cleaner: "token sequences"
- Text Cleaner → Embedding Layer: "clean token IDs"
- Embedding Layer → Data Augmenter: "embeddings [B, L, d]"
- Data Augmenter → Model Input: "augmented [B, L, d]"

## Groupings
(none - pure linear pipeline)

## Annotations
- Tensor shape "[B, L, 768]" on Embedding output
- "BPE vocab=32K" annotation on Tokenizer
- "p=0.3" augmentation probability on Data Augmenter

## Emphasis
- Embedding Layer: key transformation step
- Data Augmenter: novel contribution (if applicable)
```

## Step 4: Compose

拼接 base + pipeline layout + clean-minimal style + content。

## Step 5: Generate

```bash
npx -y bun skills/paper-figure-generator/scripts/main.ts \
  --promptfiles figures/nlp-pipeline/prompt.md \
  --output figures/nlp-pipeline/figure.png \
  --ar 16:9 \
  --provider google
```
