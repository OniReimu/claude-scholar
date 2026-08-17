---
id: PROSE.RHYTHM_VARIANCE
slug: prose-rhythm-variance
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {min_stdev_words: 10, max_band_share: 0.55}
conflicts_with: [PROSE.ANNOUNCEMENT_SENTENCE, PROSE.THEATRICAL_SPLIT, PROSE.SHORT_PUNCHY_FRAGMENTS]
constraint_type: guidance
autofix: none
---

## Requirement

一节内的句长必须有落差。散文段落的句长标准差应 ≥ **10 词**，且落在 15–30 词区间的句子占比应 ≤ **55%**。每段既要允许 8–12 词的断言，也要允许 35–45 词的展开。

均质化本身就是 AI 痕迹。执行 `PROSE.SENTENCE_LENGTH`（单句 ≤35 词）时**不得**把所有句子压到同一长度带——那条规则是上限，不是目标值。

## Rationale

人类学术写作的句长呈宽分布：作者按论点的复杂度决定句子长度，简单断言写短，多条件的技术陈述写长。LLM 生成（以及机械执行"每句一个意思、≤N 词"的改写指令）会把分布压成窄峰，读起来像节拍器。

这是最难自查的 AI 痕迹之一，因为**逐句读每一句都合格**，只有在分布层面才暴露。实测对照：一篇经三轮数学审查的稿件，被机械拆句改写后的 Introduction 为 sd=6.3、零个 >35 词句、69% 落在 15–30 词带；同一稿件未被拆句改写的 Method 节为 sd=18.0。读者报告前者"AI 味明显更重"。

修复方向是双向的：把被拆碎的从句合并回长句，同时把关键论断压成短句。只做其中一半会把峰移位而不是展宽。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **量化检查（推荐）**: 剥离公式/图表/命令后按句切分，计算句长标准差与 15–30 词区间占比
- **LLM 检查**:
  1. 连续三句是否长度接近（差值 <5 词）
  2. 整节是否找不到任何 <12 词的句子
  3. 整节是否找不到任何 >32 词的句子
  4. 改写指令是否只含长度上限而无落差要求
- **排除**: 参考文献、表格单元格、算法伪代码、caption

参考脚本（按需调整剥离规则）:

```bash
python3 - <<'PY'
import re, statistics as st, sys
t = open(sys.argv[1] if len(sys.argv)>1 else 'sec.tex').read()
t = re.sub(r'(?m)^%.*$','',t)
t = re.sub(r'\\begin\{(equation|figure|table|algorithm)\*?\}.*?\\end\{\1\*?\}','',t,flags=re.S)
t = re.sub(r'\\(cite|ref|eqref|label)\{[^}]*\}','',t)
t = re.sub(r'\$[^$]*\$','X',t); t = re.sub(r'\\[a-zA-Z]+\*?','',t); t = re.sub(r'[{}~]',' ',t)
s = [x for x in re.split(r'(?<=[.?!])\s+',t) if len(x.split())>3]
l = [len(x.split()) for x in s]
band = sum(1 for x in l if 15<=x<=30)/len(l)
print(f"n={len(l)} sd={st.pstdev(l):.1f} band={band:.0%} short={sum(1 for x in l if x<12)} long={sum(1 for x in l if x>32)}")
PY
```

## Examples

### Pass

```latex
A CT model runs one shared vector field over every trajectory, and that single
field is where the difficulty comes from. Input inside the target interval moves
the hidden state, that state keeps evolving after the interval ends, and the
segment can therefore influence outputs well downstream of itself. Training
deepens the coupling. The loss on the segment updates the same field and readout
that produce retained outputs at other times and on other trajectories, so an
edit aimed at the segment lands on the dynamics that carry everything else.
```

句长 22 / 33 / 4 / 38 词，sd 高，读起来有呼吸。

### Fail

```latex
The difficulty is the temporal coupling created by the CT dynamics. A CT model
applies one shared vector field throughout every trajectory. The input inside a
target interval changes the evolving hidden state. During training, the loss on
that interval also updates the same vector field and readout. Once the segment
has been absorbed, a post-hoc edit acts on the retained dynamics as well.
```

句长 11 / 12 / 11 / 15 / 15 词，全部落在同一带，节拍器感。

## Conflicts

与 `PROSE.ANNOUNCEMENT_SENTENCE` / `PROSE.THEATRICAL_SPLIT` / `PROSE.SHORT_PUNCHY_FRAGMENTS` 的张力裁决线：**本卡要求短句存在，那三条约束短句的内容。** 合规短句 = 承载可核对主张的短句（"Training deepens the coupling."）。修本卡时新增的短句必须直接过那三条的门槛——如果为了拉开句长落差而写出预告句或两拍式反驳，等于把一个违规换成另一个。
