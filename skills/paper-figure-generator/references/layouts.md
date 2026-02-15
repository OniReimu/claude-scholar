# Academic Figure Layouts

5 种学术论文常用图表布局。这些布局用于指导如何构建 `method.txt` 输入文件 — AutoFigure-Edit 从方法文本生成图表，因此布局选择影响你如何描述系统。

> **Note**: 以下 "Prompt 片段" 不再直接用于 API 调用，而是作为撰写 method.txt 时的参考指南 — 确保描述覆盖布局所需的所有结构元素。

---

## 1. System Overview (`system-overview`)

**适用场景**: Figure 1，方法总览图，用于论文开头展示整体框架。

**结构元素**:
- 主模块框（3-6 个）：代表系统核心组件
- 数据流箭头：展示信息在模块间的流动
- 输入/输出标注：明确系统的输入和输出
- 可选虚线分组框：将相关模块归组

**Prompt 片段**:
```
Layout: System overview diagram for an academic paper (Figure 1 style).
Structure: Show the complete system as interconnected modules flowing left-to-right or top-to-bottom.
Each module is a distinct rounded rectangle with a clear label.
Use directional arrows to show data/information flow between modules.
Group related modules with dashed boundary boxes and group labels.
Include input on the left/top and output on the right/bottom.
Keep the layout clean with consistent spacing between elements.
```

**示例论文**:
- Transformer 的 encoder-decoder 架构图
- RLHF pipeline 的 reward model + PPO 训练流程图
- RAG 系统的 retrieval → augmentation → generation 流程图

---

## 2. Pipeline (`pipeline`)

**适用场景**: 多阶段数据处理流程，展示数据从原始输入到最终输出的变换过程。

**结构元素**:
- 阶段块（Stage blocks）：每个阶段一个方框
- 连接箭头：展示数据流向，可标注数据变换
- 中间数据表示：展示每阶段输出的数据形态
- 分支/合并点：支持并行路径

**Prompt 片段**:
```
Layout: Multi-stage processing pipeline diagram.
Structure: Arrange stages as a horizontal chain of processing blocks connected by arrows.
Each stage block contains: stage name at top, key operations listed below.
Between stages, show intermediate data representations (tensors, embeddings, text, etc.) on arrows.
Support branching paths that merge later if needed.
Use consistent block sizes and arrow styles throughout the pipeline.
```

**示例论文**:
- NLP 预处理 pipeline：Raw Text → Tokenize → Embed → Encode → Classify
- 数据增强 pipeline：Original → Augment → Filter → Train

---

## 3. Threat Model (`threat-model`)

**适用场景**: 安全论文中的威胁模型图，展示攻击者、防御者和系统实体间的关系。

**结构元素**:
- 角色图标/框：攻击者（红色调）、防御者（蓝色调）、系统（中性色）
- 交互箭头：攻击向量（红色虚线）、防御措施（蓝色实线）
- 信任边界：虚线框划分信任域
- 资产标注：关键保护目标

**Prompt 片段**:
```
Layout: Threat model diagram for a security/privacy paper.
Structure: Show three types of entities - attacker (red-tinted), defender (blue-tinted), and system components (neutral).
Draw trust boundaries as dashed enclosures separating different trust domains.
Use red dashed arrows for attack vectors and blue solid arrows for defense mechanisms.
Label critical assets and data flows that need protection.
Position the attacker outside trust boundaries, defender components inside.
```

**示例论文**:
- 联邦学习中的 Byzantine 攻击者模型
- 差分隐私中的 trusted curator vs untrusted curator

---

## 4. Comparison (`comparison`)

**适用场景**: 对比图，展示本方法与 baseline 的区别，或不同方法间的并排对比。

**结构元素**:
- 并排面板（2-3 列）：每列代表一种方法
- 对齐的组件行：确保可比较的元素水平对齐
- 差异高亮：用颜色或标注突出关键差异
- "vs" 或分隔线：清晰分隔不同方法

**Prompt 片段**:
```
Layout: Side-by-side comparison diagram showing differences between approaches.
Structure: Arrange two or three approaches as parallel columns with a clear separator.
Align corresponding components horizontally so differences are immediately visible.
Use subtle color coding to distinguish approaches (e.g., blue for "Ours", gray for "Baseline").
Highlight key differentiators with accent colors or callout annotations.
Include a shared input at the top and distinct outputs at the bottom of each column.
```

**示例论文**:
- Standard Attention vs Sparse Attention vs Flash Attention 对比
- 传统训练 vs 蒸馏训练的架构差异

---

## 5. Architecture (`architecture`)

**适用场景**: 详细的神经网络架构图或系统架构图，展示内部组件和连接细节。

**结构元素**:
- 层/组件块：每个神经网络层或系统组件一个块
- 跳跃连接：ResNet-style 的 skip connections
- 维度标注：tensor shapes、通道数
- 堆叠重复：用 "×N" 表示重复模块

**Prompt 片段**:
```
Layout: Detailed architecture diagram showing internal structure of a model or system.
Structure: Show layers and components stacked vertically (bottom-to-top for neural networks).
Each block shows: component name, key parameters (e.g., dimensions, kernel size).
Use skip connections (curved arrows bypassing blocks) where applicable.
Indicate repeated blocks with "×N" notation and a bracket.
Show tensor dimensions along connection paths.
Color-code different types of operations (attention=blue, FFN=green, normalization=orange).
```

**示例论文**:
- Vision Transformer 的 Patch Embedding + Transformer Encoder 详细结构
- U-Net 的 encoder-decoder 对称架构
