# Evidence Log — 规则的证据与失效状态

这个文件存在的理由：本 skill 和 `policy/rules/` 里的绝大多数规则是**断言式**的——"不要写 X"，但没有记录"凭什么"和"什么时候这条会失效"。断言不可证伪，于是规则只会累积不会退役，错的那条会一直留着。

规则要能被推翻，就必须有三样东西：**来源、日期、样本**。这个文件记录这三样。

## Entry schema

每条证据一个小节，字段固定：

| 字段 | 含义 |
|---|---|
| `claim` | 被支持或被推翻的具体主张（对应某条 rule ID 或 SKILL.md 的某节） |
| `source` | 一手实验 / 外部 skill 包 / 文献 / 逸事 |
| `date` | 观察日期（绝对日期，不写"最近"） |
| `instrument` | 检测器与版本、审稿人、或"作者自评" |
| `register` | 被测文本的语域（社交帖 / newsletter / 学术长文 / 邮件） |
| `n` | 样本量。**必须写，哪怕是 1** |
| `verdict` | `supported` / `refuted` / `untested-here`（外部结论，我们未复现） |
| `expiry` | 什么条件下这条作废 |

## 已导入的外部证据（未在学术语域复现）

来源：外部 skill 包 `anti-ai`（用户于 2026-08 提供的 zip；含 `SKILL.md` + `references/tells.md` + `references/wordbank.md` + `evals/evals.json`）。以下结论全部来自该包作者的实验记录，**我们没有复现过任何一条**。

### E1 — 表层风格改写对统计检测器无效

- `claim`: 修完读者层 tell 不能让文本通过统计检测器
- `source`: 外部 skill 包 `anti-ai`
- `date`: 2026-07-22
- `instrument`: Pangram v3.3.2
- `register`: 社交帖 / newsletter / 逸事段落
- `n`: 11 次改写（punchy、rambling、全小写、带错别字、ESL 腔、模仿特定人笔迹、63 词最短版等）
- `verdict`: `untested-here`（外部结论：全部 100% AI，High Confidence，无梯度）
- `expiry`: 检测器换代即失效
- **我们的用法**：写进 SKILL.md「两个不同的目标」一节，作为**认知分层**使用，不作为任何规则的依据

### E2 — Interleave protocol 的比例阈值

- `claim`: 真人底稿逐句保留、AI 句只插在其间、AI 词占比 ≤ ~40% 且不连续两句 AI，文本仍被判为人类
- `source`: 同上
- `date`: 2026-07-22
- `instrument`: Pangram v3.3.2
- `register`: 一份真人手写底稿
- `n`: 4 个比例点（13% / 35% / 45% 严格交替 → Human；70% 含连续 AI 块 → 100% AI）
- `verdict`: `untested-here`
- `expiry`: 检测器换代；或在学术长文语域复测失败
- **我们的用法**：写进 SKILL.md「Interleave Protocol」一节，**明确标注学术语域未验证**。这是本次导入中唯一跨语域可用的技术

### E3 — Discourse fracture（明确不采纳）

- `claim`: 把文本改成"现场写、计划中途放弃"的散漫独白可让检测器判为人类
- `source`: 同上
- `date`: 2026-07-22
- `instrument`: Pangram v3.3.2
- `register`: 社交帖 / rant，119–155 词
- `n`: 4 次通过 + 2 次受控 A/B 失败（篇幅过长、覆盖原文全部论点、实体密度过高会翻车）
- `verdict`: `untested-here`
- **不采纳的理由**：该技术要求全小写、掉撇号、`!!`、自我打断、口语转述对白、≤160 词。学术语域**不可能**承载这套写法，该包作者本人也承认 "a discourse-fractured white paper is absurd to a human reader"。整段技术不进本 skill，仅在此备案，防止日后有人重新提议
- `expiry`: 不适用（不采纳）

### E4 — 显著性标记（significance-marking）单变量实验

- `claim`: "that's the part that got me" 这类"告诉读者哪里重要"的元评论句，单独一项就足以触发 AI 判定
- `source`: 同上
- `date`: 2026-07-22
- `instrument`: Pangram v3.3.2
- `register`: 社交帖
- `n`: 1（删掉两句、其余一字不改 → 判定从 100% AI 翻为 100% Human）
- `verdict`: `untested-here`
- **我们的用法**：不立卡。学术语域的对应物（"值得注意的是"、"这正是问题所在"）已由 `PROSE.FILLER_PHRASES` 与 `PROSE.ANNOUNCEMENT_SENTENCE` 覆盖。备案是因为它是整份外部证据里方法上最干净的一条（单变量、其余不变），值得作为将来做实验的模板

### E5 — academic-humanizer 导入（学术语域对口，推理导出）

- `claim`: 学术文本的 AI 修复需要双向校准（over-claim 与 over-hedge 同轴）+ 反向矫正护栏（保留合法学术构造）
- `source`: 外部 skill `academic-humanizer` v0.3.3（github.com/AIScientists-Dev/academic-humanizer，MIT，用户于 2026-08 提供）
- `date`: 2026-08-15（导入日）
- `instrument`: 无实验数据——该包为 SOP 型 skill，主张来自审稿惯例与写作经验，非受控测试
- `register`: 学术论文 / proposal（与本 skill 完全对口）
- `n`: 0（无实测）
- `verdict`: `untested-here`（采纳为 doctrine，理由在于与审稿实践一致，非实验证据）
- **导入内容**：HEDGING_DISCIPLINE 双向化（over-claiming verbs + 比较主张锚点 + 幅度区间）；PROMOTIONAL novelty padding（novel 超频/to the best of our knowledge/for the first time）；VAGUE_QUANTIFIERS extensive-experiments 模式；AI_LEXICON formulaic openers + 连接词密度；SKILL.md「Do NOT Over-Correct」护栏（校准 hedge/被动语态/we/术语逐字保留）；语域分流（学术 vs 随意的"人味"定义相反）；self-review 新增 contribution 套话与 citation dumping 检查
- **明确不导入**：Layer 6 proposal mode——`grant-application-writing` skill 已覆盖且更深，只加了路由指针；Layer 5 voice matching——`policy/style-guide.md` 已是更强形态
- `expiry`: 无实验数据支撑的 doctrine，若与实际审稿反馈冲突则逐条推翻

### E6 — Policy 冲突/冗余审计（2026-08-15，一手）

- `claim`: 词表类规则存在互相触发的修复循环与 FP 洪水源
- `source`: 一手审计（全量 34 条 PROSE 卡交叉比对）
- `date`: 2026-08-15
- `instrument`: 人工交叉比对 + regex 探针
- `n`: 34 条规则
- `verdict`: `supported`，已修复
- **发现并修复**：(1) INFORMAL 的 `\bsmaller\b` 禁用与自己替代表矛盾（smaller 本是规范用词）→ 移除；(2) INFORMAL autofix "a lot of"→"many" 的产物被 VAGUE_QUANTIFIERS 禁用 → 撤销该 autofix，替代表删去 numerous 建议；(3) VAGUE 的裸词 `\bsome\b` 命中数学存在量词（"for some ε>0"）→ 收窄为量词+名词组合；(4) SENTENCE_LENGTH 的 lint pattern 语义错误（统计的是文件句数>35 而非句长>35 词）→ 换为 ≥36-token 行内长句匹配，标注硬换行召回局限；(5) 短句集群四卡（RHYTHM vs ANNOUNCEMENT/THEATRICAL/SHORT_PUNCHY）conflicts_with 全空 → 补互指 + 裁决线；(6) writing-anti-ai「Personality and Soul」教注入 I/幽默与学术语域冲突 → 语域分流
- **机器化**：validate.sh 新增 Section 5c（fix-emission safety），修复循环从此由 CI 兜底

## 本 skill 新增规则的验证状态

2026-08-14 从外部包 diff 出的 4 条规则，全部为**推理导出**（reader-facing，理由写在各自 Rule Card 的 Rationale），尚无实测：

| Rule ID | 来源 | 验证状态 | 下一步 |
|---|---|---|---|
| `PROSE.AI_LEXICON` | 外部 wordbank + 学术语域裁剪 | 未验证 | 跑 `evals/evals.json` case 0–1，看误伤率（尤其 `robust`/`framework` 类术语是否被错杀） |
| `PROSE.FRACTAL_SUMMARY` | 外部 tells.md「fractal summaries」 | 未验证 | 在真实投稿 draft 上跑 regex，统计命中数与误报数 |
| `PROSE.INVENTED_CONCEPT_LABEL` | 外部 tells.md「invented concept labels」 | 未验证 | 需要一次审稿人反馈作为证据（review 里出现"检索不到这个概念"即为 supported） |
| `PROSE.RESTATEMENT_DILUTION` | 外部 tells.md「one-point dilution」 | 未验证 | 用删除测试在已接收论文上做对照：作者自己的 pre-GPT 论文复述率应显著低于 AI draft |

## 加新条目的规矩

1. **一次只动一个变量。** E4 是模板：删两句、其余一字不改。改五处再测，测出来什么都不能归因
2. **`n` 必须写。** 一次观察就写 `n: 1`，不要用"多次测试"这种说法糊过去
3. **`verdict: refuted` 要跟动作。** 推翻了就去改或退役对应的 Rule Card，并在 Card 里注明本条 entry
4. **外部结论一律 `untested-here`**，除非我们自己复现过。别把别人的实验写成我们的
5. **失效检查**：任何带检测器版本号的 entry，版本变了就重测或降级为历史记录
