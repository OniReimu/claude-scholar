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
enforcement: lint_script
params: {}
conflicts_with: []
---

## Requirement

所有 BibTeX 条目必须通过 API（Semantic Scholar、CrossRef 或 arXiv）编程获取并验证。禁止从记忆中生成 BibTeX。无法立即验证的引用必须临时标记 `[CITATION NEEDED]` 并通知研究者；在进入 self-review / revision / camera-ready 前必须清零该标记。

## Rationale

AI 生成的引用约 40% 存在错误；虚构引用构成学术不端。API 验证确保引用信息（标题、作者、年份、DOI）与实际文献匹配。

## Check

- **lint 强制检查 (`policy/lint.sh`)**:
  - 禁止存在未解决的 `[CITATION NEEDED]` 标记
  - 禁止明显占位/幻觉模式（如 `ref1` key、`...` 作者列表、`TODO/TBD` 标题）
  - 每个 BibTeX 条目必须至少包含 `doi` / `url` / `eprint` 之一（用于 API 可追溯验证）
- **人工复核**: 对 metadata 做最终核验（标题、作者、年份、venue 是否与 API 返回一致）

## Examples

### Pass

```bibtex
% 通过 Semantic Scholar API 获取，DOI 已验证
@inproceedings{vaswani2017attention,
  title     = {Attention is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {NeurIPS},
  year      = {2017},
  doi       = {10.5555/3295222.3295349}
}
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

```latex
% 提交前仍保留未解决标记（lint 会直接报错）
Recent work~\cite{unknown2024method} shows ... [CITATION NEEDED]
```
