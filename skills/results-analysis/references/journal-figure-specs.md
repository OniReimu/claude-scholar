# 期刊/会议图表硬规格速查

> 改编自 [DELONG-L/Academic-Paper-Skills](https://github.com/DELONG-L/Academic-Paper-Skills)（MIT）的 journal-specs。
> **画图前先查目标 venue**：把栏宽、字号、DPI、字体、矢量格式偏好定下来，再按最终物理尺寸出图。

## 跨 venue 速查表

| Venue | 单栏 (inch) | 双栏 (inch) | 图内字号 (pt) | 推荐字体 | 位图 DPI | 矢量首选 |
|---|---|---|---|---|---|---|
| Nature 系 | 3.5 (89mm) | 7.2 (183mm) | 5–7 | Helvetica/Arial | ≥300（线条 600） | EPS/PDF |
| Science | 2.2 (55mm) | 7.2 | 5–7 | Helvetica/Arial | ≥300 | PDF/EPS |
| PNAS | 3.42 | 7.0 | 6–8 | Helvetica/Times | 300（黑白 600） | PDF/EPS |
| IEEE（含 S&P/TDSC） | 3.5 | 7.16 | 8–10 | Times（图内可 Helvetica） | 600（线条图） | PDF/EPS |
| ACM（CCS/KDD 等） | 3.33 | 7.0 | ≥7 | Libertine 配套 sans | 300+ | PDF |
| ML 会议（NeurIPS/ICML/ICLR） | 5.5（正文单栏版面） | — | ≥ 正文脚注字号 | 与正文一致 | 300+ | PDF |

## 各 venue 的坑

- **Nature**：极其强调"按最终物理尺寸出图"——投稿系统按 mm 计算字号是否合规。子图标签 **a, b, c**（小写、加粗、左上角）。行宽 0.25–1 pt（matplotlib 默认 1 pt 偏粗，建议 0.6）。
- **Science**：单栏极窄（2.2 in），数据多时选 1.5 栏（4.7 in）而不是硬塞单栏。子图标签 **A, B, C**（大写）。
- **IEEE**：**黑白可读是硬要求**（会议印刷常黑白）——线型 + marker + 颜色三重冗余编码。子图标签 (a) (b) (c)。
- **ML 会议**：无独立图规范，但审稿人在 100% 缩放下读 PDF——图内最小字号不小于正文脚注字号，字号可读性由 `scientific-figure-making` 的 `FigureStyle` 接管（原 `FIG.FONT_GE_24PT`，已退役）。
- **arXiv**：宽松，但 PDF 字体未嵌入会在预览里变方块——`pdffonts` 检查全部 embedded。

## 出图纪律

> 适用范围：本纪律适用于有 hard-spec 的 venue（Nature 系/Science/PNAS/IEEE/ACM）；ML 会议走 `scientific-figure-making` 的 `FigureStyle` 默认口径，不受本表约束。

1. `figsize` 直接设为目标物理尺寸（如 Nature 单栏 `figsize=(3.5, 2.6)`），**不要**画大图再缩——缩放会让字号失控。
2. 字体全局指定（Helvetica/Arial 族），并确认负号/希腊字母渲染正常。
3. 导出 PDF（矢量）为主，PNG 300 dpi 作 review 副本。
4. 交叉核对 `FIG.VECTOR_FORMAT_REQUIRED`、`FIG.COLORBLIND_SAFE_PALETTE`；字号可读性由 `scientific-figure-making` 的 `FigureStyle` 接管（原 `FIG.FONT_GE_24PT`，已退役）；出图后走 `figure-visual-qa.md` 闭环。

## 提交前检查工具

- **Nature**：投稿系统自动检 DPI/尺寸/字号
- **IEEE**：PDF eXpress 检查字体嵌入
- **arXiv/通用**：`pdffonts figure.pdf` 确认无 Type 3 / 未嵌入字体
