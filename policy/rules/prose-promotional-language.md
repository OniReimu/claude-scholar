---
id: PROSE.PROMOTIONAL_LANGUAGE
slug: prose-promotional-language
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
  - pattern: "\\b(exciting|remarkable|revolutionary|groundbreaking|dramatically|game-changing|cutting-edge|unprecedented|transformative)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止使用推销性、情绪化的形容词和副词。学术论文保持中性、技术性的语气，用数据和实验结果说话。

禁用词：exciting, remarkable, revolutionary, groundbreaking, dramatically, game-changing, cutting-edge, unprecedented, transformative

偏好词：significant, critical, challenging, promising（在有数据支撑时使用）

## Rationale

推销性语言削弱学术可信度。Pre-GPT 时期的 IEEE 风格论文极少使用情绪化修饰，审稿人对此类用词敏感。

## Check

- **regex 搜索**: 匹配禁用词列表
- **检查范围**: 所有 `.tex` 文件正文区域
- **注意**: "novel" 单独使用时允许（"a novel method"），但 "novel and groundbreaking" 组合违规

## Examples

### Pass

```latex
The proposed method achieves 15.3\% improvement over the state-of-the-art baseline.
```

### Fail

```latex
We present a groundbreaking and revolutionary framework that dramatically
transforms the landscape of federated learning.
```
