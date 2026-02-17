---
id: BIBTEX.CONSISTENT_CITATION_KEY_FORMAT
slug: bibtex-consistent-citation-key-format
severity: warn
locked: false
layer: venue
artifacts: [bibtex]
phases: [writing-background, self-review, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
lint_patterns:
  - pattern: "@\\w+\\{(?![a-z]+_\\d{4}_[a-z])"
    mode: match
lint_targets: "**/*.bib"
---

## Requirement

所有 BibTeX citation key 必须遵循统一格式：`lastname_year_firstword`（如 `vaswani_2017_attention`、`devlin_2019_bert`）。禁止使用随意命名（如 `ref1`、`Paper2023`、`smith:2020:deep`）。

## Rationale

统一的 citation key 格式便于搜索、排序和维护 `.bib` 文件。不一致的命名在多人协作时尤其混乱。

## Check

- **regex 检查**: `.bib` 文件中的 citation key 是否匹配 `[a-z]+_\d{4}_[a-z]+` 格式
- **检查范围**: 所有 `.bib` 文件中的 `@type{key,` 行
- **pattern**: `@\w+\{(?![a-z]+_\d{4}_[a-z])` 匹配不符合规范的 key

## Examples

### Pass

```bibtex
@article{vaswani_2017_attention,
  title   = {Attention Is All You Need},
  author  = {Vaswani, Ashish and others},
  journal = {NeurIPS},
  year    = {2017}
}

@inproceedings{devlin_2019_bert,
  title     = {BERT: Pre-training of Deep Bidirectional Transformers},
  author    = {Devlin, Jacob and others},
  booktitle = {NAACL},
  year      = {2019}
}
```

### Fail

```bibtex
@article{ref1,
  title = {Attention Is All You Need},
  ...
}

@inproceedings{BERT2019,
  title = {BERT: Pre-training of Deep Bidirectional Transformers},
  ...
}

@article{smith:2020:deep,
  title = {Deep Learning for NLP},
  ...
}
```
