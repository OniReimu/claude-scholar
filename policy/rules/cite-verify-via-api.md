---
id: CITE.VERIFY_VIA_API
slug: cite-verify-via-api
severity: error
locked: true
layer: core
artifacts: [bibtex]
phases: [writing-background, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: manual
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

所有 BibTeX 条目必须通过 API（Semantic Scholar、CrossRef 或 arXiv）编程获取并验证。禁止从记忆中生成 BibTeX。无法验证的引用必须标记 `[CITATION NEEDED]` 并通知研究者。

## Rationale

AI 生成的引用约 40% 存在错误；虚构引用构成学术不端。API 验证确保引用信息（标题、作者、年份、DOI）与实际文献匹配。

## Check

- **人工审查**: BibTeX 来源是否来自 API 获取（而非 LLM 记忆生成）
- **标记检查**: 检查是否存在 `[CITATION NEEDED]` 标记
- **有效性验证**: 验证每个引用的 DOI/URL 是否有效、可访问

## Examples

### Pass

```bibtex
% 通过 Semantic Scholar API 获取，DOI 已验证
@inproceedings{vaswani2017attention,
  title     = {Attention is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and ...},
  booktitle = {NeurIPS},
  year      = {2017},
  doi       = {10.5555/3295222.3295349}
}
```

```latex
% 无法通过 API 验证的引用，正确标记
Recent work~\cite{unknown2024method} shows ... [CITATION NEEDED]
```

### Fail

```bibtex
% 从记忆中编造：作者名拼写错误、年份不对、DOI 不存在
@inproceedings{vaswani2018attention,
  title     = {Attention is All You Need},
  author    = {Vaswany, Ashish and Shazer, Noam},
  booktitle = {ICML},
  year      = {2018},
  doi       = {10.1234/fake.doi.000}
}
```
