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

推荐布局：`pipeline`（多阶段处理链，最佳匹配）

## Step 2: Prepare — method.txt + 参考图片

创建 `figures/nlp-pipeline/method.txt`:

```text
Text Preprocessing Pipeline

A multi-stage processing pipeline that transforms raw text into model-ready tensors.

Stage 1: Raw Text - Unprocessed input documents and sentences enter the pipeline.

Stage 2: Tokenizer - Splits text into subword tokens using BPE/WordPiece algorithm
(vocabulary size 32K). Input: sentences. Output: token sequences.

Stage 3: Text Cleaner - Removes noise, normalizes unicode characters, and handles
special characters. Input: token sequences. Output: clean token IDs.

Stage 4: Embedding Layer - Maps clean token IDs to dense vector representations.
Output tensor shape: [B, L, 768] where B is batch size, L is sequence length.

Stage 5: Data Augmenter - Applies augmentation techniques including synonym replacement
and back-translation with probability p=0.3. Input: embeddings [B, L, d].
Output: augmented embeddings [B, L, d].

Stage 6: Model Input - Final batched tensor ready for model training.

The pipeline flows strictly left to right, with each stage transforming the data
representation for the next. The Embedding Layer is the key transformation step
converting discrete tokens to continuous representations.
```

用户指定 clean minimal 风格，提供一张 Nature Methods 论文的参考图片：

## Step 3: Setup

```bash
ls skills/paper-figure-generator/scripts/.venv/bin/python
```

## Step 4: Generate

```bash
# 带风格迁移的生成
bash skills/paper-figure-generator/scripts/generate.sh \
  --method_file figures/nlp-pipeline/method.txt \
  --output_dir figures/nlp-pipeline \
  --use_reference_image \
  --reference_image_path path/to/nature-style-reference.png
```

输出:
- `figures/nlp-pipeline/figure.png` — 光栅预览
- `figures/nlp-pipeline/final.svg` — 可编辑 SVG

## Step 5: Finalize

```bash
uv run python -c "import cairosvg; cairosvg.svg2pdf(url='figures/nlp-pipeline/final.svg', write_to='figures/nlp-pipeline/figure.pdf')"
```

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/nlp-pipeline/figure.pdf}
  \caption{Text preprocessing pipeline. Raw text is tokenized (BPE, vocab=32K),
  cleaned, embedded into dense vectors ($d{=}768$), and augmented ($p{=}0.3$)
  before being fed to the model.}
  \label{fig:preprocessing}
\end{figure}
```
