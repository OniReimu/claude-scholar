# Visual Styles

3 种学术图表视觉风格，每种包含美学描述和 prompt 片段。

---

## 1. Modern Gradient (`modern-gradient`)

**默认风格**，适用于 NeurIPS/ICML/ICLR 2024-2025 风格的顶会论文。

**视觉特征**:
- 渐变色模块背景（浅色到中等饱和度）
- 圆角矩形（border-radius: 12-16px 等效）
- 柔和阴影（subtle drop shadow）
- 半透明分组框
- 清晰的 sans-serif 字体（类似 Inter, Helvetica）

**Prompt 片段**:
```
Visual style: Modern gradient academic illustration.
Use soft gradient fills for modules (light-to-medium saturation).
Rounded rectangles with generous corner radius.
Subtle drop shadows for depth.
Clean sans-serif typography for all labels.
White or very light gray background.
Color palette: use harmonious gradients - blue (#4A90D9 to #7BB3F0), green (#4CAF50 to #81C784),
purple (#7E57C2 to #B39DDB), orange (#FF9800 to #FFB74D), teal (#009688 to #4DB6AC).
Semi-transparent group boundaries with light fill.
Smooth, anti-aliased arrows with consistent weight.
```

**推荐色板**:
| 用途 | 渐变色 |
|------|--------|
| 主模块 | `#4A90D9 → #7BB3F0` (蓝) |
| 数据处理 | `#4CAF50 → #81C784` (绿) |
| 特殊模块 | `#7E57C2 → #B39DDB` (紫) |
| 高亮模块 | `#FF9800 → #FFB74D` (橙) |
| 辅助模块 | `#009688 → #4DB6AC` (青) |

---

## 2. Clean Minimal (`clean-minimal`)

**适用于** Nature/Science/Cell 等高影响期刊，追求极简清晰的视觉效果。

**视觉特征**:
- 纯色填充（flat colors），无渐变
- 高对比度配色（深色文字 + 彩色模块）
- 细线条（1-2px stroke）
- 大量留白（generous whitespace）
- 严格的网格对齐

**Prompt 片段**:
```
Visual style: Clean minimal academic illustration, Nature/Science journal style.
Use flat, solid color fills for modules - no gradients, no shadows.
High contrast: dark text (#333333) on colored backgrounds.
Thin, precise lines (1-2px) for borders and arrows.
Generous whitespace between all elements.
Strict grid alignment for all components.
Color palette: limited to 3-4 colors - blue (#3B7DD8), coral (#E8604C),
gold (#D4A843), gray (#8C8C8C), with white (#FFFFFF) backgrounds.
Sharp corners or very slight rounding (2-4px).
Simple arrowheads, no decorative elements.
Typography: clean, professional, consistent sizing.
```

**推荐色板**:
| 用途 | 颜色 |
|------|------|
| 主模块 | `#3B7DD8` (蓝) |
| 次要模块 | `#E8604C` (珊瑚红) |
| 强调 | `#D4A843` (金) |
| 中性 | `#8C8C8C` (灰) |
| 背景 | `#FFFFFF` (白) |

---

## 3. Technical Blueprint (`technical-blueprint`)

**适用于**工程导向的论文，强调技术精确性和系统化呈现。

**视觉特征**:
- 深色或蓝灰色背景
- 网格线/参考线可见
- 方正的块状元素（sharp corners）
- 等宽字体标注（monospace labels）
- 工程图纸/蓝图美学

**Prompt 片段**:
```
Visual style: Technical blueprint / engineering schematic style.
Dark background (#1A1A2E or #0D1B2A) with light-colored elements.
Visible grid lines in very subtle color for alignment reference.
Sharp-cornered rectangles with thin bright borders.
Monospace or technical font for labels and annotations.
Accent colors: cyan (#00D4FF), green (#00FF88), yellow (#FFD700), white (#FFFFFF).
Wire-style connections with right-angle routing.
Component blocks with pin-style connection points.
Circuit-board or architectural drawing aesthetic.
Data flow shown with animated-style dashed lines or glowing edges.
```

**推荐色板**:
| 用途 | 颜色 |
|------|------|
| 背景 | `#1A1A2E` (深蓝灰) |
| 主元素 | `#00D4FF` (青) |
| 数据流 | `#00FF88` (绿) |
| 强调 | `#FFD700` (金黄) |
| 文字 | `#FFFFFF` (白) |
