# 图表视觉自检闭环（render → read → fix → re-render）

> 改编自 [DELONG-L/Academic-Paper-Skills](https://github.com/DELONG-L/Academic-Paper-Skills)（MIT）的 visual-qa 工作流。

画图工具的通病：**画完就结束，没人回看成图**。文字被裁、图例压住数据、子图编号乱放、符号变方框——这些问题全部漏到投稿。本闭环要求：每张数据图在导出最终矢量版**之前**，渲染 PNG 预览并**实际读图**逐条自检。

```
绘制 → ① 渲染 PNG 预览 (150 dpi) → ② 用 Read 工具读图，逐条过 8 项清单
                                          ↓ 发现问题
        ④ 通过后才导出矢量图 ← ③ 回源头改图 → 重渲 → 再读 ←┘
```

## 为什么必须渲染 PNG 再"看"

- 矢量 PDF/SVG 无法直接检查像素层面的重叠与遮挡——先栅格化。
- 程序能查的有限（缺字、越界是确定性的），但"图例正好压住一簇数据点""两条标注叠在一起""配色发灰分不开"是**感知性**问题，只有把图当图像看才能发现。
- Claude Code 的 `Read` 工具可直接读 PNG——这个闭环**必须实际执行**，不是 checklist 装饰。

## 操作流程

1. **渲染预览**：`fig.savefig("figs/_preview.png", dpi=150, bbox_inches="tight")`——150 dpi 足够看清文字与重叠。在导出最终 PDF/SVG **之前**做。
2. **读图自检**：用 `Read` 读 `figs/_preview.png`，逐条对照下方清单。不要扫一眼说"看起来不错"——一项一项过。
3. **回源头改**：发现问题回脚本改，不在预览图上修补。
4. **重渲再读**：每改一处重渲一次。全过后才导出矢量图。

## 读图自检清单（8 项）

1. **乱码/方框**：负号、`±`、`×`、希腊字母、上下标有没有缺字？（缺负号 → `axes.unicode_minus=False`）
2. **文字被裁切**：标题、轴标签、图例、旋转后的长刻度标签有没有被画布边缘切掉？（→ `bbox_inches='tight'` / 缩短标题）
3. **文字遮盖/重叠**：图例压住数据？标注互相叠？x 轴刻度挤成一团？（→ 图例移图外 `bbox_to_anchor=(1.02,1)`；刻度 `rotation=30` 或减刻度数）
4. **子图编号对齐**：a/b/c/d 横看一条线、竖看一条线？字号风格一致？
5. **子图间距**：子图互相侵入？y 轴标签伸进邻居？colorbar 压住数据？（→ `constrained_layout=True`）
6. **配色与灰度**：各类别区分得开？有没有红绿对比？灰度打印仍可分？（→ colorblind-safe 调色板 + 线型/marker 冗余编码，见 `FIG.COLORBLIND_SAFE_PALETTE`）
7. **数据完整性**：数据点、曲线、误差棒顶端有没有被轴范围切掉？（→ 放宽 `set_xlim/ylim` 或 `ax.margins(0.05)`）
8. **跨子图一致性**：同一变量在多个子图同色、同 marker？共享含义的轴范围一致？

## 循环纪律

- **每改一处就重渲一次**——不要一次改五处然后猜结果。
- **最多 3 轮**：3 轮过不了多半是图型选错或维度太多，回头重选图型/拆图。
- **留痕**：把每轮发现的问题和改法简要告诉用户。
- 剩余问题需用户显式接受才可豁免（如"标签确实密，但这是数据本身决定的"）。

## 与既有规则的关系

本闭环是执行手段，不新增规则；它落地的是 `FIG.COLORBLIND_SAFE_PALETTE`、`FIG.FONT_GE_24PT`、`FIG.NO_IN_FIGURE_TITLE`、`FIG.SELF_CONTAINED_CAPTION`、`FIG.VECTOR_FORMAT_REQUIRED` 的实检。概念图（`paper-figure-generator` 产物）同样适用第 1–3、6 项：接受 `final.svg` 前先 Read 渲染出的 PNG。
