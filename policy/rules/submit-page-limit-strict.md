---
id: SUBMIT.PAGE_LIMIT_STRICT
slug: submit-page-limit-strict
severity: error
locked: false
layer: venue
artifacts: [text]
phases: [self-review, camera-ready]
domains: [core]
venues: [neurips, icml, iclr, acl, aaai, colm]
check_kind: manual
enforcement: doc
params:
  neurips: 9
  icml: 8
  iclr: 9
  acl: 8
  aaai: 7
  colm: 9
conflicts_with: []
---

## Requirement

论文主体内容必须严格遵守会议页数限制：

| 会议 | 页数限制 |
|------|---------|
| NeurIPS | 9 页 |
| ICML | 8 页 |
| ICLR | 9 页 |
| ACL | 8 页 |
| AAAI | 7 页 |
| COLM | 9 页 |

参考文献和附录不计入页数限制，但主体内容是硬限制。

## Rationale

超页限制导致自动 desk rejection，投稿系统强制执行。写作时需留出约 0.5 页余量以应对最后调整。

## Check

- **人工检查**: 编译后 PDF 的页数
- **主体内容**: 检查 main content 最后一页是否超出限制
- **建议**: 写作过程中持续编译检查页数，留 0.5 页余量

## Examples

### Pass

```
NeurIPS 投稿：主体正好 9 页，references 和 appendix 另起新页。
```

### Fail

```
NeurIPS 投稿：主体 10 页，超出 1 页限制。
```
