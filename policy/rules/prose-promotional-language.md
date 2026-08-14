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
params: {novel_max_per_file: 2, tbook_max_per_file: 1}
conflicts_with: [PROSE.AI_LEXICON]
constraint_type: guardrail
autofix: assisted
lint_patterns:
  - pattern: "\\b(exciting|remarkable|revolutionary|groundbreaking|dramatically|game-changing|cutting-edge|unprecedented|transformative)\\b"
    mode: match
  - pattern: "\\bnovel\\b"
    mode: count
    threshold: 2
    threshold_param: novel_max_per_file
  - pattern: "(?i)\\bto the best of our knowledge\\b"
    mode: count
    threshold: 1
    threshold_param: tbook_max_per_file
  - pattern: "(?i)\\bfor the first time\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止使用推销性、情绪化的形容词和副词。学术论文保持中性、技术性的语气，用数据和实验结果说话。

禁用词：exciting, remarkable, revolutionary, groundbreaking, dramatically, game-changing, cutting-edge, unprecedented, transformative

偏好词：significant, critical, challenging, promising（在有数据支撑时使用）

**Novelty padding**（同类问题的自夸变体）：

- `novel` 全文 ≤ `novel_max_per_file`（默认 2）次。新颖性由「与最近先行工作的具体差异」证明，不由形容词证明
- `to the best of our knowledge` 全文 ≤ 1 次。第二次出现说明作者在用套话顶替 gap 论证
- `for the first time` 逐处审查：改写为具体差异（"prior calibration work is offline; we study the online setting"），首创性主张交给证据

## Rationale

推销性语言削弱学术可信度。Pre-GPT 时期的 IEEE 风格论文极少使用情绪化修饰，审稿人对此类用词敏感。

Novelty padding 是同一失败的结构化版本：AI 起草的 Introduction 高频堆叠 novel / to the best of our knowledge / for the first time，把「新在哪里」的论证换成断言。审稿人对这三个短语已经形成条件反射，出现即扣分预备。

## Check

- **regex 搜索**: 推销词 match；novel 与 to-the-best-of-our-knowledge 用 count 阈值；for the first time 逐处 match
- **检查范围**: 所有 `.tex` 文件正文区域
- **注意**: "novel" 单独使用时允许（"a novel method"），但 "novel and groundbreaking" 组合违规；novelty 类命中的修复方向是写出与 closest prior work 的具体差异，不是删词了事

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
