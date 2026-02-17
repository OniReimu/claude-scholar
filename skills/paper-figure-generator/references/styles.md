# Visual Styles — Reference Image Guide

AutoFigure-Edit 支持通过 `--reference_image_path` 进行风格迁移。提供一张参考图片，生成的图表会模仿其视觉风格。

当前内置默认参考图（`generate.sh` 自动使用）：
- `skills/paper-figure-generator/.autofigure-edit/img/reference/sample3.png`（primary）
- `skills/paper-figure-generator/.autofigure-edit/img/reference/sample2.png`（fallback）

---

## 如何使用风格迁移

```bash
bash scripts/generate.sh \
  --method_file figures/my-figure/method.txt \
  --output_dir figures/my-figure \
  --use_reference_image \
  --reference_image_path path/to/reference-style.png
```

---

## 推荐风格方向

### 1. Modern Gradient

**适用于**: NeurIPS/ICML/ICLR 2024-2025 风格的顶会论文。

**视觉特征**:
- 渐变色模块背景（浅色到中等饱和度）
- 圆角矩形
- 柔和阴影
- 清晰的 sans-serif 字体

**参考图片来源**: 从目标会议的 best paper 中截取 Figure 1 作为参考图片。

**推荐色板**:
| 用途 | 渐变色 |
|------|--------|
| 主模块 | `#4A90D9 → #7BB3F0` (蓝) |
| 数据处理 | `#4CAF50 → #81C784` (绿) |
| 特殊模块 | `#7E57C2 → #B39DDB` (紫) |
| 高亮模块 | `#FF9800 → #FFB74D` (橙) |
| 辅助模块 | `#009688 → #4DB6AC` (青) |

---

### 2. Clean Minimal

**适用于**: Nature/Science/Cell 等高影响期刊。

**视觉特征**:
- 纯色填充，无渐变、无阴影
- 高对比度配色
- 细线条（1-2px）
- 大量留白
- 严格的网格对齐

**参考图片来源**: Nature Methods 或 Science 论文中的示意图。

**推荐色板**:
| 用途 | 颜色 |
|------|------|
| 主模块 | `#3B7DD8` (蓝) |
| 次要模块 | `#E8604C` (珊瑚红) |
| 强调 | `#D4A843` (金) |
| 中性 | `#8C8C8C` (灰) |
| 背景 | `#FFFFFF` (白) |

---

### 3. Technical Blueprint

**适用于**: 工程导向论文，强调技术精确性。

**视觉特征**:
- 深色或蓝灰色背景
- 网格线可见
- 方正元素
- 等宽字体
- 工程图纸美学

**参考图片来源**: IEEE 系统架构论文或工程蓝图风格的示意图。

**推荐色板**:
| 用途 | 颜色 |
|------|------|
| 背景 | `#1A1A2E` (深蓝灰) |
| 主元素 | `#00D4FF` (青) |
| 数据流 | `#00FF88` (绿) |
| 强调 | `#FFD700` (金黄) |
| 文字 | `#FFFFFF` (白) |

---

## 选择参考图片的建议

1. **从目标会议 Best Paper 中选择**: 最直接的方式 — 截取同会议同年度获奖论文的 Figure 1
2. **保持风格一致性**: 同一篇论文的所有概念图应使用相同参考图片
3. **分辨率要求**: 参考图片建议 ≥ 800px 宽，清晰可辨
4. **避免过于复杂的参考图**: 选择结构清晰、元素分明的图片效果更好
5. **不提供参考图也可以**: AutoFigure-Edit 有默认风格，适合大多数场景
