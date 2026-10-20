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

### E7 — 首轮 eval 运行（2026-08-15，一手）

- `claim`: 优化后的 skill 在 7 个学术语域用例上行为正确，且新增能力（检测器预期管理、过度矫正护栏）是 delta 的全部来源
- `source`: 一手实验——skill-creator eval harness，7 case × 2 版本（当前 vs 2026-08-13 快照）并行盲跑
- `date`: 2026-08-15
- `instrument`: 14 个独立 subagent（claude-fable-5），39 条断言（机械 regex + 人工复核），结果在 `writing-anti-ai-workspace/iteration-1/benchmark.json`
- `register`: 学术论文（Related Work / System Model / 实验分析 / Discussion / 中文摘要 / 咨询问答）
- `n`: 每格 1 次（n=1，方向性证据，非统计证据）
- `verdict`: `supported`
- **结果**: 新版 39/39（100%），旧版 37/39（94.9%）。旧版仅有的两个失分：(1) eval-4 未给 interleave 路径，且 agent 自由发挥了检测器民科（"perplexity/burstiness、深度改写通常能降分"——与 E1 直接矛盾）；(2) eval-6 把 "The results suggest that" 当双重 hedge 删掉——恰是 Do NOT Over-Correct 护栏防的那个动作，新版引用校准红线显式保留
- **AI_LEXICON 误伤检验通过**: eval-0 双版本都在改写中原样保留 loss landscape / robust / optimize / training trajectories，术语豁免有效
- **已知混杂**: 旧版快照只冻结了 SKILL.md+references，policy/ 目录共享且是当前版——旧版 run 读到了新 rule cards 和新 guardrail-checklist（eval-2 旧版读了 08-14 才存在的 prose-invented-concept-label.md）。**真实 delta 被低估**；eval 0/1/2/3/5 打平证明的是 policy 层的承载力，不是 skill diff 无效
- **观察（待跟踪）**: 两个版本的 eval-0 都往改写里加了原文没有的机制细节（implicit regularization / flat minima）——Related Work 压缩语境下可接受，但这是 fabrication-adjacent 倾向，下轮 eval 应加断言盯住
- `expiry`: 换模型或改 skill 后需重跑

### E8 — 解释式语域六原则（2026-10-19，一手）

- `claim`: AI 时代论文和人写论文的文风差异，根子在语域，而不是个别词。先按六条原则改，再按 anti-AI 改，能把文风推向人写的方向；只做通用润色不能
- `source`: 一手实验（p2g 仓库 `spike-runs/1019-calib/style_roots/`、`rewrite/`）
- `date`: 2026-10-19
- `instrument`:
  - 句法：spaCy en_core_web_sm 3.8；
  - 文风：p2g 文风分，由虚词 logistic 模型加 lint 措辞标记组成，按 ChatGPT 之前 95% 的论文 ≥ 80 锚定，人写论文落在 82–96；
  - 改写：每一步都是看不到检测器、看不到词表的全新子代理（claude-opus-5-5）
- `register`: 学术长文。测量用 280 篇 arXiv 论文，覆盖 7 个领域，ChatGPT 之前和 2025 年以后各 140 篇；改写用一篇论文（arXiv 2608.17050）的 Introduction、Background 和 Methods 三节，约 2,500 词
- `n`: 测量 280 篇；改写每种做法 n=1
- `verdict`: `supported`（方向性证据）
- **测量结果**（7 个领域方向一致）：
  - 作者作主语 −40%（全文）；主语分工（前 3,000 词）：`we` ×0.76，被动 ×0.75，系动词主干 ×0.71，名词主语配主动动词 ×1.39（集中在把结论归给事物的动词），7/7 领域同向；
  - 名词串 +22%，名词化 +20%；
  - 情态和助动词约 −1/3，被动 −26%；
  - 目的不定式 −30%；
  - 范围介词增多；
  - 对照句式合计翻倍（每千词 1.00 → 2.06），其中 `rather than` ×7.3，`instead of` ×0.5；
  - 带解释词的 `rather than` 句从每百万词 26 升到 233，而带解释词的句子占比只从 12.6% 升到 14.1%。
- **改写结果**（文风分）：
  - 原文 33；
  - 通用润色 33；
  - 5 条原则 39；
  - 5 条 + anti-AI 56；
  - 5 条 + anti-AI 再改两轮 54 / 54；
  - **6 条 + anti-AI 71**；
  - 再单独执行一次第 2 条 70：名词串拆得更开了，但拆的时候换上了范围介词，收益抵消了，所以第 2 条要求拆完更具体。
- **内容保真**：引用、ref、公式、数字都不变，只删了 (1)(2)(3) 列表编号。唯一的问题是 6 条版本删掉了一个属于 ③ 的比较论断（readers 和 backbone identity 的对比），所以第 6 条第 ③ 种写成「必须保留并就地给结果」。
- **局限**：只有一篇论文，每种做法只改了一次，单点有几分的波动。56 和 71 之间的差距大于这个波动。
- `expiry`: 换改写模型，或者在第二篇论文上复测时方向不一致，就降级

### E9 — 逐章迭代改写（2026-10-20，一手）

- `claim`: 整篇重跑完整 skill 不会再涨（E8），但逐章、第 2 轮起只按六原则改，能继续把文风推向人写的方向，前三轮都在涨；再往后，涨的多半是把某些词推到人写论文之外
- `source`: 一手实验（p2g 仓库 `spike-runs/1019-calib/rewrite-6g/`、`rewrite-edge/`）
- `date`: 2026-10-20
- `instrument`:
  - 文风：p2g 文风分（同 E8），另算一个扣掉超出人写范围加分的分数：某个虚词的频率超出 ChatGPT 之前论文的 5%–95% 区间时，超出部分带来的加分不算；
  - 改写：每章每轮一个全新子代理（claude-opus-5-5），看不到检测器；主代理每轮回退越界改动并核对引用、ref、公式、数字
- `register`: 学术长文，两篇作者自己的期刊稿，6g 约 13,000 词、edge 约 7,000 词
- `n`: 3 篇。6g、edge 各 4–5 轮；anti_nso 2 轮，用 `writing-anti-ai-loop` 按 skill 原样跑
- `verdict`: `supported`（方向性证据）
- **结果**（文风分；括号里是扣掉超范围加分以后的分数）：
  - 6g：57 → 61 → 69 → 75 → 75（73），第 4 轮不涨，停；
  - edge：48 → 57 → 60 → 60 → 78（72）→ 75（72）；
  - anti_nso（S&P 投稿版，卡在 18 页上限，各章不许变长）：48（46）→ 50（48）→ 50（48），第 2 轮不涨，还超到 19 页，停，采用第 1 轮。不许变长时只涨了 2 分；六原则要补主语、写出句间关系，这些都要花词数；
  - edge 第 4 轮的 78 是虚高：真条件全改成了 `When` 从句，`when` 升到每千词 9.4，句首 `When` 是人写论文最高值的两倍多。第 5 轮把同一组设置收到段首，显示分降到 75，扣掉以后的分数不变。第 5 条原则因此加了「不要每句都挂 When」。
- **立论分和论证分**（edge，同一版 Overleaf 底稿对比改写稿）：论证分 100 → 100，立论分 50 → 65。改写没有拉低这两项。
- **内容保真**：引用和公式全部保留；有 3 个数字各少了一次，都是重复出现的值被换成了回指。每轮都有子代理加进原文没有的推论或条件（6g 4 处，edge 3 处，其中一处被下一轮又加了回来），全部由主代理改回。所以审改动这一步不能省。
- **局限**：三篇都是同一位作者的稿件，同一个改写模型；分数只是检测器上的变化，不等于审稿人读起来更好。有页数上限的稿件，这条路涨得很少。
- `expiry`: 换改写模型，或者在其他作者的论文上方向不一致，就降级

### E10 — 按检测器词表改写是 reward hacking；增益集中度能识别（2026-10-20，一手）

- `claim`: 从检测器的逐词权重反推出来的换词规则能涨分，但那是换词，不是写法变化；看「变化最多的 3 个词占了多少」能把它和按写法改写分开
- `source`: 一手实验（p2g 仓库 `spike-runs/1019-calib/rewrite-anti_nso-97f9d44/`）
- `date`: 2026-10-20
- `instrument`: 文风分同 E9；p2g 服务的 `gain_top3_share`（这一轮虚词变化里，变化最多的 3 个词占的比例，不透露是哪几个词）
- `register`: 学术长文，anti_nso（S&P 投稿稿，约 10,000 词，页数放开）
- `n`: 1 篇（集中度另在 6g、edge 上各核对一次）
- `verdict`: 词表 `refuted`，已删；第 3 条新款 `untested-here`（这篇上没有分数变化，见下）
- **经过**：anti_nso 按六原则跑到第 2 轮 54（52），第 3 轮不涨。我查看了检测器的逐词贡献，发现扣分集中在三个量化词和情态词上，于是在第 5 条里写了一张「这几个词在做什么、该用哪个」的表。按它跑一轮：62（59）。
- **为什么算 reward hacking**：规则的来源是检测器计哪些词，不是作者在为读者做什么；拿来佐证的「ChatGPT 之前 140 篇」又正是检测器的训练语料，等于自证。涨的 8 分里约 6 分来自那三个词；被换进来的词超过了人写论文 95% 的上限，但它在检测器里几乎没有权重，所以分数没有察觉。
- **增益集中度**：六原则各轮 0.24–0.38（anti_nso 三轮 0.35 / 0.24 / 0.38，6g 0.35，edge 0.34）；词表那一轮 0.77。loop skill 因此把「超过 0.5 的那一轮不算涨」写进停止判据。
- **替代做法**：不给 Codex 看词表和分数，只给六原则、拟议规则和两段原文，请它评审。它独立指出了同义词替换的漏洞，建议不另立一条，在第 3 条下加一款：全称和必要性论断要交代范围和依据，重述要有作用；换同义词不算修改。按这一款在第 2 轮上再跑一轮：53（51），集中度为 null（虚词几乎没动）。大多数子代理报告「这些论断都已交代依据，不用改」。这篇的全称论断多半是内容本身需要，按写法改写的上限大约就是 54。
- **局限**：一篇稿子；0.5 这个阈值只在 4 次改写上看过。
- `expiry`: 在其他论文上，按写法改写的轮次集中度超过 0.5，就要重定阈值

## 本 skill 新增规则的验证状态

| Rule ID | 来源 | 验证状态 | 下一步 |
|---|---|---|---|
| `PROSE.AI_LEXICON` | 外部 wordbank + 学术语域裁剪 | **eval-passed (n=1, E7)**——误伤检验通过，术语全部存活 | 真实投稿 draft 上跑 lint 统计误报率 |
| `PROSE.FRACTAL_SUMMARY` | 外部 tells.md「fractal summaries」 | **eval-passed (n=1, E7)** | 在真实投稿 draft 上跑 regex，统计命中数与误报数 |
| `PROSE.INVENTED_CONCEPT_LABEL` | 外部 tells.md「invented concept labels」 | **eval-passed (n=1, E7)**——正反例都判对（删 supervision paradox、留 CPCL） | 需要一次审稿人反馈作为证据（review 里出现"检索不到这个概念"即为 supported） |
| `PROSE.RESTATEMENT_DILUTION` | 外部 tells.md「one-point dilution」 | **eval-passed (n=1, E7)** | 用删除测试在已接收论文上做对照：作者自己的 pre-GPT 论文复述率应显著低于 AI draft |
| `HEDGING_DISCIPLINE` 双向化 + Over-Correct 护栏 | academic-humanizer (E5) | **eval-passed (n=1, E7)**——护栏是 39:37 delta 的一半来源 | 真实 rebuttal/Discussion 上验证 |
| `PROSE.EXPLAINING_REGISTER`（六原则） | 一手测量 + 盲改写 (E8、E9、E10) | **supported（n=3 篇改写）**——文风 33 → 71（E8）；逐章迭代 57 → 75、48 → 75（E9）；按检测器词表改写被识别为换词并删除（E10） | 在其他作者的论文上复测；查 ③ 型对照是否保留 |
| Interleave protocol 比例阈值 | 外部 anti-ai 包 (E2) | **仍未实测**（学术长文语域） | 用户真实底稿 + 目标检测器复测 |

## 加新条目的规矩

1. **一次只动一个变量。** E4 是模板：删两句、其余一字不改。改五处再测，测出来什么都不能归因
2. **`n` 必须写。** 一次观察就写 `n: 1`，不要用"多次测试"这种说法糊过去
3. **`verdict: refuted` 要跟动作。** 推翻了就去改或退役对应的 Rule Card，并在 Card 里注明本条 entry
4. **外部结论一律 `untested-here`**，除非我们自己复现过。别把别人的实验写成我们的
5. **失效检查**：任何带检测器版本号的 entry，版本变了就重测或降级为历史记录
