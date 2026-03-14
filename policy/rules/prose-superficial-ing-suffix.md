---
id: PROSE.SUPERFICIAL_ING_SUFFIX
slug: prose-superficial-ing-suffix
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
lint_patterns:
  - pattern: ",\\s*(highlighting|underscoring|emphasizing|showcasing|reflecting|symbolizing|contributing to|fostering|cultivating|encompassing)\\s"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止在句末用逗号接 "-ing" 分词短语来添加浮浅分析。这些尾部分词短语通常不携带实质信息，只是 AI 用来制造"深度感"的填充。

禁用模式：`, highlighting...`, `, underscoring...`, `, emphasizing...`, `, showcasing...`, `, reflecting...`, `, symbolizing...`, `, contributing to...`, `, fostering...`

如果确实需要说明意义，用独立句子并提供具体证据。

## Rationale

尾部 -ing 分词短语是 AI 生成文本的高频特征。它们通常附加空洞的意义声明（"highlighting its importance", "reflecting broader trends"），不携带可验证的信息。人类学术写作极少这样写。

## Check

- **regex 搜索**: 匹配逗号后接禁用 -ing 动词的模式
- **排除**: 句中合法的分词结构（如 "By highlighting X, we show Y"）
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
The proposed method reduces memory consumption by 40\%.
This reduction enables deployment on edge devices with limited RAM.
```

### Fail

```latex
The proposed method reduces memory consumption by 40\%, highlighting
its potential for deployment on resource-constrained edge devices and
underscoring the importance of efficient memory management in modern
deep learning systems.
```
