# Claude Scholar

<div align="center">
  <img src="LOGO.jpeg" alt="Claude Scholar Logo" width="100%"/>
</div>

**语言**: [English](README.md) | [中文](README.zh-CN.md)

面向学术研究和软件开发的个人 Claude Code 配置仓库 - 一个完整的工作环境。

## News

- **2026-08-27 (v1.15.0)**: 新增规则 `PROSE.CAUSAL_CONNECTIVE`，以及一次**改写了规则本身说法**的评估。起点观察：`X, so Y` 是口语的因果连接，用它就把正式连接词承载的区别抹掉了——`therefore`（逻辑蕴含）· `hence`（承接刚建立的结论）· `thus`（以此方式）· `consequently`（经验后果）。以 arXiv 源码建的 pre-ChatGPT 基线：`, so` 0.18/千词、正式连接词 2.16/千词，比例近 12:1；该基线在**类别与月份窗口完全不重叠**的样本外语料上复现（0.28 对 2.18）。本地 draft 密度是基线的 15–25 倍。**然后评估推翻了原本的框架。** 样本外类别（cs.SE / stat.ML / cs.DB）到 2026 只涨 1.3 倍，而 cs.LG/CR/CL 涨 3.4 倍；同类别两个月窗差近 2 倍——漂移是**领域特异**的，单窗口数字不能当点估计。更决定性的是：从 pre-GPT / 2026 / 本地 draft 各抽 14 条被命中的真实句子，**打乱编号隐去来源后盲评**，定位精度 42/42（无一条非因果 `so`），而**逐条判别力为零并且反转**——收紧判据后 pre-GPT 应改率 64%、draft 只有 29%。任取一个 `, so`，无论谁写的，可改进的比例都差不多，差别只在**有多少个**。因此本卡是一条**因果精度**规则而非 AI 痕迹规则，只对三个可诊断子类触发改写：设计选择伪装成推论 · 因果未被证据支持 · 证明推导步骤。其余一律保留——用"能否更精确"当判据会把 pre-GPT 水准的散文一起改掉。在真实章节上实施：该规则 5 → 0，**其他规则零变化**，逗号少 4 个，正式连接词分散在三个词上而非塌到一个词。
- **2026-08-27 (v1.14.1)**: CI 从 2026-08-17 起一直是红的，三条 FAIL 只在 runner 上出现。`deprecated_by` 的可解析性用 `-e` 判断后继，而 `-e` 会跟随符号链接——有三个 skill 是指向 vendor submodule 的符号链接，不带 submodule 的 checkout 会让它们悬空，后继随之无法解析。缺的是**树里的内容**，不是**引用本身**；现改用 `-L`，两种树下答案一致，同时 workflow 拉取 submodule 让其余步骤看到真实的树。修改前后都在无 submodule 的 clone 上复现验证过。
- **2026-08-26 (v1.14.0)**: 补两套现有测试从来没问过的问题。`policy/test-lint.sh` 测的是 lint **机制**（flag、fix 发射、退出码），从不问规则有没有报在对的句子上；`validate.sh` 查 registry 不变量，从不读散文。**`policy/test-corpus.sh`** 跑 88 条带标注的语料（单句、术语近似碰撞、整段），逐 case 报漏报与误报。**precision case 是重点**——过度触发是本引擎被记录在案的失效模式，一条把每个 threat-model 句子都报出来的规则，最终会被作者直接关掉。**`policy/test-referrals.sh`** 把此前靠手工验的转诊图机器化：目的地存在（R1）、目的地确实执行接手的那条规则（R2，词级规则与 vendor submodule skill 的豁免带理由登记在案）、转诊写明**最小需要跑什么**而不只报一个 skill 名（R3，只对"贵"的目的地生效）。写这两套测试的过程中暴露出六个缺陷。**字符类按字节匹配**：在 macOS 实际走的 perl 路径上，`[→←↔]` 命中了所有破折号、弯引号和省略号——pattern 从环境变量取回时是字节串。CI 走 GNU grep，永远看不到；现在语料在两个引擎下各跑一遍，`LINT_ENGINE` 可强制指定。**`%` 注释里的散文被 lint**，作者搁置的备选写法产生了任何编辑都清不掉的 finding。**`lint_patterns` 块里的一行 YAML 注释会静默截断它**，其后所有 pattern 被禁用且无任何诊断；`validate.sh` 新增 4d 节把它变成错误。**句首大写泄漏**：大小写敏感的 pattern 漏掉了 `A lot of` / `Significantly` / `Things` / `Dramatically` / `A number of`——而句首正是口语化最常出现的位置。**`harness`** 作为本领域每个 evaluation harness 的标准名词却被当成 AI 词表命中，现改为仅动词形态触发。**`PROSE.NEGATIVE_PARALLELISM` 只匹配缩写形式**（`It's not X, it's Y`），因此在学术散文实际使用的非缩写形式上从不触发。另外发现并修掉两处只报 skill 名的转诊，与 v1.13.4 修的是同一类。
- **2026-08-26 (v1.13.4)**: 让转诊在**真实工作流**上走得通。起草（`ml-paper-writing`）是 0→1 的活动；面对已有稿件，入口是逐节调用的 `writing-anti-ai`，一切结构性发现能否到达作者，取决于它转诊是否成功。此前两方面都不成功。**覆盖**：`PROSE.RULE_OF_THREE` 的跨节情形**完全没有转诊**（边界前一天刚加进规则卡，没同步到 skill），`PROSE.SELF_UNDERMINING` 的转诊是间接的。现均已显式化。**可达性**：只报一个 skill 名字的转诊，在作者只想清理一节时读起来就是"去跑四遍全文审计"。现在两边都写明**最小需要跑什么**——`claim-architecture-review` 记录了**定点入口**（spine 已存在 → 只对该节跑 P1；不存在 → 先跑廉价的 P0，它只读 abstract/intro/标题/主题句而不读正文，再 P1；只有需要全局搬迁规划才跑 P2/P3），而 `progress.md` 让定点运行成为全量审计的预付款而非一次性投入。**归属**：arch-review 是五条规则结构半边的声明目的地，却一个 marker 都没有——现从 1 条增至 6 条，每条挂在**真正执行该检查的那个 pass** 上，判据是「**执行才认领**，仅仅相关不算」。`EXP.EXPERIMENT_ROLE` 现在于 P1 对 Results 节按 spine 判定，其重设计→弱化→删除阶梯映射到该 skill 既有的 verdict——此前它只在起草时与 self-review 触发，改稿路径上**从不运行**。P3 增加第二问：不只问 spine 是否闭环，还问它是否就是最强证据指向的那个命题——一篇论文可以围绕它自己数据并不支持的主张完美闭环。
- **2026-08-26 (v1.13.3)**: 补上 v1.13.2 留下的接缝，并加固校验器。`PROSE.RULE_OF_THREE` 新增的"同一集合不得枚举两次"没有**跨节归属**：规则把自己限定在段与节内，却从未说明一组对象在 Method 列一次、Discussion 又列一次该归谁；而 `claim-architecture-review` 的 information ledger 是**命题键**，全文零处提到列举。两边都以为对方管。现在规则用与 `PROSE.RESTATEMENT_DILUTION` 完全一致的措辞划出边界（节内归线编，跨节归结构编辑），ledger 则把**一组被列举的对象记为一个 information unit 并以该集合为键**——`{CNN channels, MLP hidden units, attention coords}` 是一个 info-key 而不是四个——于是别处第二次列举会在 `lookup-before-create` 上撞上。另外，`validate.sh` 第 1 节此前每张卡 fork 14 次 `grep`（全规则集约 1400 次 fork），在系统重载下偶发的 fork 失败与"字段缺失"无法区分，于是校验器会对**完全正确的文件**报内容错误。现改为纯 bash 模式匹配（零 fork），frontmatter 读不到单列为一种独立失败，单次运行耗时约减半。**一个会在正确输入上偶发失败的校验器，会训练使用者"重跑到绿"，那比不检查更糟。**
- **2026-08-24 (v1.13.2)**: `PROSE.RULE_OF_THREE` 从"三项并列"扩为**并列列举的密度与重复**——一段真实方法学文字暴露了两个洞。它的三个列表里只有一个是三项并列，另两个（各四项）从定义上逃逸；机械层唯一命中是 `PROSE.COMMA_OVERUSE`，那是**副作用命中**，报的是逗号不是列举，作者照它去修会改错地方。更糟的是 `writing-anti-ai` 里写着 *"Rule of three: prefer two or four items"*——这句从通用 anti-AI 语境继承（四项列表能破坏三段式指纹），但在学术散文里它**生产**更长的列举墙，且与卡自己的"三个以上用 enumerate"直接矛盾。现在四条判据：同段三项并列 ≤1 次（不变）；**同一集合不得枚举两次**——首次列举时命名，之后引用名字；内联项数按项的长短分档（短项 ≤4、多词名词短语 ≤2），于是 `expand, duplicate, reorder, or rescale` 可以留在句内而四个长名词短语不行；一段中带列表的句子 ≤2 句。反向护栏放在卡的最前面：**技术散文里列举合法且常见，修法是命名+引用，永不删项**——尤其 `PROSE.COMMA_OVERUSE` 不得靠删列表项来满足。第 2、4 条属**行为变更**。ID 保留以维持引用稳定，并在卡内注明其范围已超出名字。
- **2026-08-23 (v1.13.1)**: 把叙事主动性原则里三处**可线编**的部分接进 `writing-anti-ai`——草稿扫描时真正能改的地方。`PAPER.OUTCOME_LOGIC` 此前在那里 **0 marker**：它挂在写作时与审查时，于是流水账在 review 里被**发现**，却没有线编环节去**修**。新增 §8b 覆盖三种句级形态（`we first tried X, which did not work…`、`initially we used A but later switched to B`、`in an earlier version of this work`），配删除测试、**显式的不写 regex 指令**（`first`/`then`/`initially` 合法用法远多于违规），以及最要紧的边界：只删**实现弯路**——跑过且结果限定了主张的实验是证据，消融与划定失效边界的负面结果属于成果逻辑而非行踪。章节重排仍归 `claim-architecture-review`。`PROSE.AI_LEXICON` 的**套话开场**与**句首连接词密度**两类此前 lint 一直在抓，却在**执行它的 skill 里不可见**——读 skill 的执行者没有理由去查开篇句；现已 surface，并补上正面要求：Abstract/Introduction 首句须携带可验证的结构性事实而非趋势陈述。`PROSE.OVER_DEFENSIVE` 新增第 5 条**文档域**落点错误：Abstract 与 Introduction 中贡献落地之前不得出现 caveat。**行为变更**——此前通过的稿件可能在这条上新报违规；其处置**只移不删**，且必须先确认 Limitations 已完整承担该边界。原则 ①（围绕优势组织）②（选对战场）④（实验职责）**刻意不接**进来：三者的判定都需要全文 claim 或证据全貌，让只看得见句子的执行者去判，正是本仓库反复记录的过度执行失效模式。
- **2026-08-23 (v1.13.0)**: 叙事主动性层，来自同行对 agentic 论文写作的现场诊断。诊断本身：agent 因为怕被指"不够周全"，会**替想象中的审稿人攻击自己**——把每一个可能的弱点都标出来、到处提醒，宁可平庸也不愿犯错。用同行的六条原则逐条审计本仓库，结果是四条覆盖、两条缺失、外加**两处真实的内部矛盾**。(1) **三条新规则。** `PROSE.SELF_UNDERMINING` —— "不递刀子"的词级层，而 `PROSE.OVER_DEFENSIVE` 明确声明自己只管结构不管词级；15 条高精度 pattern 带术语豁免前瞻，`falls short of the information-theoretic limit` 与 `lags behind by two time steps` 放行，`Unfortunately, our method does not outperform…` 命中，而**中性量化陈述的不利结果放行**——这个分界就是本卡的全部意义。`EXP.EXPERIMENT_ROLE` —— 每个实验须承担四种论证职责之一（证明方法有效／归因优势来源／目标场景价值／排除最可能的竞争解释），配重设计 → 弱化 → 删除的处置阶梯；此前七条 `EXP.*` 全是格式与诚信，无一条问"这个实验是干什么用的"。`PAPER.OUTCOME_LOGIC` —— 写最终成立的逻辑而非做事的顺序，并带上其他规则都没有给的**授权**：证据撑不住原始叙事时，重定义问题、重排贡献是正确动作，不是让步。(2) **三条各自都写了完整性边界**，否则每一条都可能被读成隐藏证据的许可：管措辞不减披露；实验因"不服务于任何主张"下架，绝不因"数字不好看"；重排改的是顺序与 framing，不是报告集合。预注册与审稿人要求的实验豁免于一切阶梯。(3) **`policy/style-guide.md` 一直在与它被宣称同级的规则打架** —— 它的 canonical 段落范例以 `With the rapid development of X, Y has attracted significant attention` 开篇、以 `significantly improves` 收尾，两处分别被 `PROSE.AI_LEXICON` 与 `PROSE.INTENSIFIERS_ELIMINATION` 禁止。照着这份强制先读的 style-guide 写，下一步就被同仓库的 lint 判违规。范例已修（扫描中又发现 §3.1 第二处违规），作者的五段式骨架与声音**原样不动**；新增 **`validate.sh` 11b** 对 style-guide 自己的散文代码块跑 lint，让同级权威不能再漂移。(4) **`validate.sh` 4c** 校验 `phases:` 是否在 Phase 词汇表内——这从来没有被检查过，`writing-intro` 作为未声明值在一张卡里躺了 30+ 个 commit；Introduction 确是真实阶段，已补入词汇表与 Step→Phase 映射。另外：`claim-architecture-review` 的 claim spine 增加优势维度（并显式护栏："不支撑优势"永远不是删除授权），`ml-paper-writing` 增加**选对战场**定位程序——刻意不立规则，因为没有任何句级检查能区分"选对了战场"和"避重就轻"，其硬限是**领域默认的那个比较仍须照常报告**。
- **2026-08-17 (v1.12.0)**: 补上 **author-original 文本**的语域覆盖，来源是一篇 NDSS 投稿（20 页）的现场反馈——该稿此前已跑过 `writing-anti-ai`。逐节清理抓出约 30 处语域问题，把其中 26 条代表串喂给 `PROSE.INFORMAL_VOCABULARY` 的 9 条 lint pattern，命中 **0**。诊断不是执行失败而是**声明范围与工具能力不匹配**：卡片自称负责"词表层命中，以及未被本次 pass 改动的文本"，但工具只有 9 条单词级 regex，而 30 处里 **29 处是多词构造**。(1) **`PROSE.INFORMAL_VOCABULARY` 重建为五类分类表**——习语性状语（regex 判定）、短语动词顶替拉丁语源动词、判断性形容词、谓语位置的具象名词比喻、内部工作痕迹动词；后四类由 LLM 按各自判据 + `params` allowlist 裁决，因为**过度执行才是已记录的失效模式**（`from scratch`、`rules out`、`falls back`、`cheap unlearning` 全是术语，加长黑名单会把它们一并毁掉）。类 2–4 刻意**不做 autofix**。(2) **新规则 `PROSE.IDIOM_COLLISION`**——技术短语与常用习语同形时读者第一遍走习语义（`a fair bit` 指无偏比特、`on the order of`、`significant`）。既不是语域问题也不是准确性问题，现有规则全覆盖不到。(3) **新增 `policy/references/tex-prose-extraction.md`**，并挂到全部 59 张 `llm_*` 规则卡的 Check 段：手搓 `.tex` 扫描器产生**假的「已清零」结论**的四种实测写法——`split('%')` 在 `$95\%$` 处截断、`$` 奇数配对剔数学吞掉整段（某次实测丢了一节的 42%）、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致。(4) `PROSE.OVER_DEFENSIVE` 增加多 home caveat 的删除顺序步骤；`PROSE.REGISTER_PRESERVATION` 补充 anti-AI pass 也是触发场景——一处 `at all` 正是 agent 在「让句子更好懂」时加进去的，**「降低阅读门槛」与「降低语域」被系统性混为一谈**。该规则的 diff-only 范围**刻意不动**。
- **2026-08-15 (v1.11.0)**: 两条由真实投稿扫描驱动的 policy 加固，外加一个 CI 修复。(1) **`PROSE.NO_INTERNAL_PROVENANCE` 加固**——规则本就存在，却仍让 11 处开发痕迹进了 ACM ASIA CCS 投稿的编译产物（其中一处是印在 evaluation 正文里的本地路径）。失效原因是三个落地缺口而非规则本身：`guardrail-checklist.md` 里根本没有它（那份紧凑清单才是起草时真会被读的）、`lint.sh` 零覆盖、`severity: warn` 与它自己 Rationale 里"硬伤"的措辞错配。现在：checklist 注册、`lint.sh` 新增 **builtin P1–P5 检测器**（排除表在匹配**之前**剥离 `\includegraphics`/`\input`/引用 key/artifact `\url`/EXP 披露——排除表是"守得住的护栏"与"被关掉的护栏"的分界）、`severity: error` 且用 `params.drafting_severity` 记录起草期降级、taxonomy 3 → 7 类（数据路径、schema 列名、内部 fixture 名、修订叙事）、Requirement 改为**先讲 provenance 该去哪**再讲禁令（根因是范式冲突：同 turn 计算即写作时，写出路径感觉像在守项目奖励的规矩），并新增 `policy/scripts/extract-undefined-identifiers.sh` 支撑两阶段未定义标识符子检查。(2) **新规则 `PROSE.REGISTER_PRESERVATION`**——语域是**编辑动作**的属性而非词的属性，因此执行点是 diff。一次压缩 pass（968 → 728 词）产生 9 处语域违规，`PROSE.INFORMAL_VOCABULARY` 的五条 pattern 命中 **0**；其中两处根本不口语，只是更不精确。加宽正则在专家验收文本上实测 **precision 0.00 / recall 0.00**（35 条命中），因此本规则刻意采用 `check_kind: llm_style` 且**不发布** `lint_patterns`。规则卡先给修复规则（用稿件别处已有的措辞——9 处里 7 处可这样恢复），含 5 类 drift 分类与 10 条实测排除，并设工作流闸门：**register check 未通过前不得报告词数或压缩百分比**。`writing-anti-ai` v1.2.0 的 "Do NOT Over-Correct" 第 1 条改为**替换**方向——原六条全是删除方向，而九处违规无一是删除。(3) **CI 修复**：`grep -c` 在计数为 0 时退出码为 1，在 `bash -eo pipefail` 下导致 policy workflow 恰恰因为 `validate.sh` 变干净而失败。
- **2026-08-15 (v1.10.0)**: 反 AI 写作大修 + policy 冲突审计 + 首轮实测。新增 4 条规则：`PROSE.AI_LEXICON`（tier-1 零容忍词表 + tier-2 密度阈值 + 套话开场 + 句首连接词预算，术语豁免保证 `loss landscape`/`robust`/`optimize` 不被误伤）、`PROSE.FRACTAL_SUMMARY`（禁止逐层预告/回顾）、`PROSE.INVENTED_CONCEPT_LABEL`（自造术语须有出处或显式命名声明）、`PROSE.RESTATEMENT_DILUTION`（同一命题一节内只说一次）；`PROSE.HEDGING_DISCIPLINE` 双向化（over-claim 与 over-hedge 同轴判定，并设校准红线防反向矫正）。冲突审计修掉自相矛盾的禁用词、autofix 修复循环、命中数学存在量词的裸词误报、`SENTENCE_LENGTH` 语义错误的 lint pattern，以及短句四卡互相振荡；`validate.sh` 新增 5c 检查把"修复不得触发其他规则"变成机器不变量。`writing-anti-ai` 升至 v1.1.0：读者层与统计检测器分离（不承诺过检测）、interleave protocol、语域分流、过度矫正护栏、证据日志与回归用例；首轮实测 39/39，改造前快照 37/39。同时以 submodule 引入 [zksecurity/zk-skills](https://github.com/zksecurity/zk-skills) 的 `circom-auditor`，并修复 `skill-forced-eval.js` 漏检所有符号链接 skill 的 bug。
- **2026-02-21**: 新增首版 SoK 策略包：4 条语义规则 `SOK.*`、`security-sok-sp` profile，以及 3 个入口 skill 的 marker 集成。v1 中 SoK 仍通过 profile 激活（暂不做 schema 迁移）。
- **2026-02-19 (v1.3.0)**: 引入论文策略引擎（`policy/`）：在 `policy/rules/` 采用规则卡设计并作为唯一真相源，支持分层作用域（`core/domain/venue`）、`policy/profiles/` 配置覆盖，以及 `policy/validate.sh` + `policy/lint.sh` 的可执行校验流程。同步强化图表工作流策略（Figure 1 必须存在；非实验图默认走 AutoFigure-Edit）。
- **2026-02-16 (v1.2.1)**: 新增全局出图规则：任何生成图（AutoFigure-Edit 概念图、旧版生图链路、Python 实验图）都不添加图内标题；标题信息统一放在论文 caption/正文中。
- **2026-02-16**: 强化 `paper-figure-generator` 执行优先级：默认先走 `AutoFigure-Edit + OpenRouter`，仅在默认链路失败后才回退到旧版 Gemini/OpenAI 流程；新增旧插件缓存提示（`GOOGLE_API_KEY` / `OPENAI_API_KEY`）排障说明。
- **2026-02-15**: 迁移 `paper-figure-generator` 至 AutoFigure-Edit — 从方法文本生成可编辑 SVG 矢量图；替代 Gemini/OpenAI 光栅生成；支持风格迁移；使用 OpenRouter + Roboflow（免费 SAM3 API）
- **2026-02-13**: 新增 `paper-figure-generator` 技能；项目打包为 Claude Code 插件（`.claude-plugin/plugin.json`）；新增 `.env.example`；深度整合至 ml-paper-writing、results-analysis、post-acceptance、using-claude-scholar 工作流；共 34 个技能
- **2026-02-11**: 大版本更新，新增 10 个 skills（research-ideation、results-analysis、citation-verification、review-response、paper-self-review、post-acceptance、daily-coding、frontend-design、ui-ux-pro-max、web-design-reviewer）、7 个 agents、8 个研究工作流命令、2 条新规则（security、experiment-reproducibility）；重构 CLAUDE.md；涉及 89 个文件
- **2026-01-26**: 所有 Hooks 重写为跨平台 Node.js 版本；README 完全重写；扩展 ML 论文写作知识库；合并 PR #1（跨平台支持）

## 简介

Claude Scholar 是一个面向 Claude Code CLI 的个人配置系统，提供丰富的技能、命令、代理和钩子，针对以下场景优化：
- **学术研究** - 完整的研究生命周期：想法生成 → 实验 → 结果分析 → 论文写作 → 审稿回复 → 会议准备
- **软件开发** - Git 工作流、代码审查、测试驱动开发、ML 项目架构
- **插件开发** - Skill、Command、Agent、Hook 开发指南与质量评估
- **项目管理** - 规划文档、代码规范、跨平台钩子驱动的自动化工作流

## 快速导航

| 主题 | 说明 |
|------|------|
| 🚀 [快速开始](#快速开始) | 快速上手指南 |
| 📚 [核心工作流](#核心工作流) | 论文写作、代码组织、技能进化 |
| 🛠️ [功能亮点](#功能亮点) | 技能、命令、代理概览 |
| 📖 [安装指南](#安装选项) | 完整、最小化或选择性安装 |
| 🔧 [项目规则](#项目规则) | 代码规则 + 论文策略引擎 |

## 核心工作流

### 主要工作流

完整的学术研究生命周期 - 从想法到发表的 7 个阶段。

#### 1. 研究构思

系统化的研究启动，包含想法生成和文献综述：

**工具**: `research-ideation` skill + `literature-reviewer` agent

**流程**:
- **5W1H 头脑风暴**: What, Why, Who, When, Where, How → 结构化思维框架
- **文献综述**: arXiv + Semantic Scholar 集成 → 自动化论文搜索和分类
- **Gap 分析**: 5 种类型（文献、方法论、应用、跨学科、时间）→ 识别研究机会
- **研究问题**: SMART 原则 → 制定具体、可衡量的问题

**命令**: `/research-init "topic"` → 启动完整的研究启动工作流

#### 2. ML 项目开发

可维护的 ML 项目结构，用于实验代码：

**工具**: `architecture-design` skill + `code-reviewer` agent + `git-workflow` skill

**流程**:
- **结构**: Factory & Registry 模式 → 配置驱动模型（仅 `cfg` 参数）→ 由 `rules/coding-style.md` 强制执行
- **代码风格**: 200-400 行文件 → 需要类型提示 → 配置使用 `@dataclass(frozen=True)` → 最多 3 层嵌套
- **调试** (`bug-detective`): Python/Bash/JS 的错误模式匹配 → 堆栈跟踪分析 → 反模式识别
- **Git**: Conventional Commits (`feat/scope: message`) → 分支策略（master/develop/feature）→ 使用 `--no-ff` 合并

**命令**: `/plan`, `/commit`, `/code-review`, `/tdd`

#### 3. 实验分析

实验结果的统计分析和可视化：

**工具**: `results-analysis` skill + `data-analyst` agent

**流程**:
- **数据处理**: 自动化清理和预处理实验日志
- **统计检验**: t-test, ANOVA, Wilcoxon signed-rank → 验证显著性
- **可视化**: matplotlib/seaborn 集成 → 发表级图表（折线图、柱状图、热图）
- **消融实验**: 系统化组件分析 → 理解每个部分的贡献

**命令**: `/analyze-results <experiment_dir>` → 生成带有图表和统计数据的分析报告

#### 4. 论文写作

从模板到最终草稿的系统化论文写作：

**工具**: `ml-paper-writing` skill + `paper-miner` agent + `latex-conference-template-organizer` skill

**流程**:
- **模板准备**: 下载会议 .zip → 提取主文件 → 删除示例内容 → 输出适合 Overleaf 的干净结构
- **引文验证** (`citation-verification`): 多层验证（格式 → API → 信息 → 内容）→ 防止幻觉引用
- **系统化写作**: 叙事框架 → 5 句式摘要公式 → 分节起草与反馈循环
- **去 AI 化处理** (`writing-anti-ai`): 移除夸大象征、宣传语言、模糊归因 → 添加人性化声音和节奏 → 双语支持（中英文）

**会议**: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, Science, Cell, PNAS

#### 5. 论文自审

提交前的质量保证：

**工具**: `paper-self-review` skill

**流程**:
- **结构检查**: 逻辑流畅性、章节平衡、叙事连贯性
- **逻辑验证**: 论证合理性、主张-证据对齐、假设清晰性
- **引文审计**: 引用准确性、适当归属、引文完整性
- **图表质量**: 视觉清晰度、标题完整性、色彩无障碍性
- **写作润色**: 语法、清晰度、简洁性、学术语气
- **合规性**: 页数限制、格式要求、伦理披露

**多项检查清单** → 系统化质量评估（含图表标题与 LaTeX 数学公式规范）

#### 6. 论文提交与 Rebuttal

论文提交和审稿意见回复：

**工具**: `review-response` skill + `rebuttal-writer` agent

**提交流程**:
- **提交前检查**: 会议特定检查清单（NeurIPS 16 项、ICML 更广泛影响、ICLR LLM 披露）
- **格式检查**: 页数限制、匿名化、补充材料
- **最终审查**: 校对、检查引用、验证图表

**Rebuttal 流程**:
- **审稿意见分析**: 解析并分类评论（主要/次要/错字/误解）
- **回复策略**: 接受/辩护/澄清/实验 → 针对每种评论类型的定制方法
- **Rebuttal 写作**: 结构化回复，包含证据和推理
- **语气管理**: 专业、尊重、基于证据的语言

**命令**: `/rebuttal <review_file>` → 生成完整的 rebuttal 文档和实验计划

#### 7. 录用后处理

会议准备和研究推广：

**工具**: `post-acceptance` skill

**流程**:
- **演讲**: 幻灯片创建指导（15/20/30 分钟格式）→ 视觉设计原则 → 叙事结构
- **海报**: 学术海报模板（A0/A1 尺寸）→ 布局优化 → 视觉层次
- **推广**: 社交媒体内容（Twitter/X, LinkedIn）→ 博客文章 → 新闻稿 → 研究摘要

**命令**: `/presentation`, `/poster`, `/promote` → 自动化内容生成

**覆盖范围**: 90% 的学术研究生命周期（从想法到发表）

### 支撑工作流

这些工作流在后台运行，增强主要工作流。

#### 自动化执行工作流

跨平台钩子（Node.js）自动化工作流执行：

```
会话开始 → 技能评估 → 会话结束 → 会话停止
```

- **skill-forced-eval** (`skill-forced-eval.js`): 在每次用户提示之前 → 动态扫描所有可用技能（本地 + 插件）→ 强制评估每个技能 → 要求实现前激活 → 确保不遗漏相关技能
- **session-start** (`session-start.js`): 会话开始时 → 显示 Git 状态、待办事项、可用命令、包管理器 → 一目了然地展示项目上下文
- **session-summary** (`session-summary.js`): 会话结束时 → 生成全面的工作日志 → 总结所做的所有更改 → 附带 orchestrator 状态与最近事件摘要
- **stop-summary** (`stop-summary.js`): 会话停止时 → 快速状态检查 → 检测临时文件 → 显示可操作的清理建议

**跨平台**: 所有钩子使用 Node.js（非 shell 脚本），确保 Windows/macOS/Linux 兼容性。

#### 知识提取工作流

两个专门的挖掘代理持续提取知识以改进技能：

- **paper-miner** (agent): 分析研究论文（PDF/DOCX/arXiv 链接）→ 提取写作模式、结构见解、会议要求、审稿意见回复策略 → 使用分类条目更新 `ml-paper-writing/references/knowledge/`（structure.md、writing-techniques.md、submission-guides.md、review-response.md）
- **kaggle-miner** (agent): 研究获胜的 Kaggle 竞赛解决方案 → 提取竞赛简介、前排方案详细技术分析、代码模板、最佳实践 → 更新 `kaggle-learner` skill 的知识库（`references/knowledge/[domain]/` 目录，按 NLP/CV/Time Series/Tabular/Multimodal 分类）

**知识反馈循环**: 每篇分析的论文或解决方案都会丰富知识库，创建一个随您研究进化的自我改进系统。

#### 技能进化系统

维护和改进技能的 3 步持续改进循环：

```
skill-development → skill-quality-reviewer → skill-improver
```

1. **开发** (`skill-development`): 创建具有正确 YAML frontmatter 的技能 → 清晰的描述和触发短语 → 渐进式披露（精简的 SKILL.md，详细信息在 `references/`）
2. **审查** (`skill-quality-reviewer`): 4 维质量评估 → 描述质量（25%）、内容组织（30%）、写作风格（20%）、结构完整性（25%）→ 生成优先修复的改进计划
3. **改进** (`skill-improver`): 合并建议更改 → 更新文档 → 根据反馈迭代 → 自动读取并应用改进计划

## 文件结构

```
claude-scholar/
├── AGENTS.md            # Codex 行为参考（保留在仓库中；不再复制）
├── .codex/              # Codex 专用文件
│   └── INSTALL.md               # Codex 安装指南
│
├── hooks/               # 跨平台 JavaScript 钩子（仅 Claude Code）
│   ├── session-start.js         # 会话开始 - 显示 Git 状态、待办事项、命令
│   ├── skill-forced-eval.js     # 每次提示前强制技能评估
│   ├── session-summary.js       # 会话结束 - 生成带有建议的工作日志
│   ├── stop-summary.js          # 会话停止 - 快速状态检查、临时文件检测
│   └── security-guard.js        # 文件操作的安全验证
│
├── skills/              # 35 个专业技能（领域知识 + 工作流）
│   ├── ml-paper-writing/        # 完整论文写作：NeurIPS, ICML, ICLR, ACL, AAAI, COLM
│   │   └── references/
│   │       └── knowledge/        # 从成功论文中提取的模式
│   │       ├── structure.md           # 论文组织模式
│   │       ├── writing-techniques.md  # 句子模板、过渡
│   │       ├── submission-guides.md   # 会议要求（页数限制等）
│   │       └── review-response.md     # 审稿意见回复策略
│   │
│   ├── research-ideation/        # 研究启动：5W1H、文献综述、Gap 分析
│   │   └── references/
│   │       ├── 5w1h-framework.md           # 系统化思维工具
│   │       ├── gap-analysis-guide.md       # 5 种研究 Gap 类型
│   │       ├── literature-search-strategies.md
│   │       ├── research-question-formulation.md
│   │       ├── method-selection-guide.md
│   │       └── research-planning.md
│   │
│   ├── results-analysis/         # 实验分析：统计、可视化、消融
│   │   └── references/
│   │       ├── statistical-methods.md      # t-test, ANOVA, Wilcoxon
│   │       ├── visualization-best-practices.md  # matplotlib/seaborn
│   │       ├── results-writing-guide.md    # 结果章节写作
│   │       └── common-pitfalls.md          # 常见分析错误
│   │
│   ├── review-response/          # 系统化 rebuttal 写作
│   │   └── references/
│   │       ├── review-classification.md    # 主要/次要/错字/误解
│   │       ├── response-strategies.md      # 接受/辩护/澄清/实验
│   │       ├── rebuttal-templates.md       # 结构化回复模板
│   │       └── tone-guidelines.md          # 专业语言
│   │
│   ├── paper-self-review/        # 多项质量检查清单
│   ├── post-acceptance/          # 会议准备
│   │   └── references/
│   │       ├── presentation-templates/     # 幻灯片创建（15/20/30 分钟）
│   │       ├── poster-templates/           # 学术海报设计
│   │       ├── promotion-examples/         # 社交媒体内容
│   │       └── design-guidelines.md        # 视觉设计原则
│   │
│   ├── citation-verification/    # 多层引文验证
│   ├── writing-anti-ai/         # 移除 AI 模式：象征主义、宣传语言
│   │   └── references/
│   │       ├── patterns-english.md    # 要移除的英文 AI 模式
│   │       └── patterns-chinese.md     # 要移除的中文 AI 模式
│   │
│   ├── architecture-design/     # ML 项目模式：Factory、Registry、配置驱动
│   ├── git-workflow/            # Git 纪律：Conventional Commits、分支
│   ├── bug-detective/           # 调试：Python、Bash、JS/TS 错误模式
│   ├── code-review-excellence/  # 代码审查：安全性、性能、可维护性
│   ├── skill-development/       # 技能创建：YAML、渐进式披露
│   ├── skill-quality-reviewer/  # 技能评估：4 维评分
│   ├── skill-improver/          # 技能进化：合并改进
│   ├── kaggle-learner/          # 从 Kaggle 获胜解决方案中学习
│   ├── doc-coauthoring/         # 文档协作工作流
│   ├── latex-conference-template-organizer  # Overleaf 模板清理
│   └── ... （10+ 更多技能）
│
├── commands/            # 50+ 斜杠命令（快速工作流执行）
│   ├── research-init.md         # 启动研究启动工作流
│   ├── analyze-results.md       # 分析实验结果
│   ├── rebuttal.md              # 生成系统化 rebuttal 文档
│   ├── presentation.md          # 创建会议演讲大纲
│   ├── poster.md                # 生成学术海报设计方案
│   ├── promote.md               # 生成推广内容
│   ├── plan.md                  # 带代理委托的实施方案规划
│   ├── commit.md                # Conventional Commits：feat/fix/docs/refactor
│   ├── code-review.md           # 质量和安全审查工作流
│   ├── tdd.md                   # 测试驱动开发：Red-Green-Refactor
│   ├── build-fix.md             # 自动修复构建错误
│   ├── verify.md                # 运行验证循环
│   ├── checkpoint.md            # 保存验证状态
│   ├── refactor-clean.md        # 移除死代码
│   ├── learn.md                 # 从代码中提取模式
│   └── sc/                      # SuperClaude 命令套件（20+ 命令）
│       ├── sc-agent.md           # 代理管理
│       ├── sc-estimate.md       # 开发时间估算
│       ├── sc-improve.md         # 代码改进
│       └── ...
│
├── agents/              # 14 个专业代理（专注任务委托）
│   ├── literature-reviewer.md   # 文献搜索和趋势分析
│   ├── data-analyst.md          # 自动化数据分析和可视化
│   ├── rebuttal-writer.md       # 系统化 rebuttal 写作
│   ├── paper-miner.md           # 提取论文知识：结构、技巧
│   ├── architect.md             # 系统设计：架构决策
│   ├── code-reviewer.md         # 审查代码：质量、安全、最佳实践
│   ├── tdd-guide.md             # 指导 TDD：测试优先开发
│   ├── kaggle-miner.md          # 从 Kaggle 提取工程实践
│   ├── build-error-resolver.md  # 修复构建错误：分析和解决
│   ├── refactor-cleaner.md      # 移除死代码：检测和清理
│   ├── bug-analyzer.md          # 深度代码执行流分析和根因调查
│   ├── dev-planner.md           # 实施规划和任务拆解
│   ├── ui-sketcher.md           # UI 蓝图设计和交互规范
│   └── story-generator.md       # 用户故事和需求生成
│
├── rules/               # 全局指导原则（始终遵循的约束）
│   ├── coding-style.md          # ML 项目标准：文件大小、不可变性、类型
│   ├── agents.md                # 代理编排：何时委托、并行执行
│   ├── security.md              # 密钥管理、敏感文件保护
│   └── experiment-reproducibility.md  # 随机种子、配置记录、检查点
│
├── policy/              # 论文策略引擎（规则卡 + 校验 + lint）
│   ├── rules/                    # 论文写作规则卡（单一真相源）
│   ├── profiles/                 # 领域/会议覆盖配置（severity/params）
│   ├── validate.sh               # 规则卡结构与集成校验
│   ├── lint.sh                   # 可机器执行的规则检查
│   └── README.md                 # 策略引擎设计说明
│
├── scripts/
│   ├── install-codex.sh         # Codex 安装器（macOS/Linux，符号链接）
│   ├── install-codex-windows.ps1 # Codex 安装器（Windows，junction）
│   └── lib/                     # 共享脚本工具
│
├── CLAUDE.md            # 全局配置：项目概述、偏好设置、规则
│
└── README.md            # 本文件 - 概述、安装、功能
```

## 功能亮点

### 技能（29 个）

**写作与学术：**
- `ml-paper-writing` - 顶级会议/期刊的完整论文写作指导
- `writing-anti-ai` - 移除 AI 写作模式（双语支持）
- `doc-coauthoring` - 结构化文档协作工作流
- `latex-conference-template-organizer` - LaTeX 模板管理
- `daily-paper-generator` - 自动化每日论文生成，用于研究追踪

**研究工作流：**
- `research-ideation` - 研究启动：5W1H 头脑风暴、文献综述、Gap 分析
- `results-analysis` - 实验分析：统计检验、可视化、消融实验
- `review-response` - 系统化 rebuttal 写作，语气管理
- `paper-self-review` - 多项质量检查清单（含图表与 LaTeX 数学公式规范）
- `post-acceptance` - 会议准备：演讲、海报、推广
- `citation-verification` - 多层引文验证，防止幻觉引用
- `paper-figure-generator` - 学术论文概念图生成（系统总览、Pipeline、架构图等，基于 AutoFigure-Edit，生成可编辑 SVG）

**开发：**
- `daily-coding` - 日常编码检查清单（极简模式，自动触发）
- `git-workflow` - Git 最佳实践（Conventional Commits、分支）
- `code-review-excellence` - 代码审查指南
- `bug-detective` - Python、Bash、JS/TS 调试
- `architecture-design` - ML 项目设计模式
- `verification-loop` - 测试和验证

**安全审计：**
- `circom-auditor` - Circom / ZK 电路审计：soundness、completeness、隐私、约束缺陷（17 agent 委派工作流，vendored 自 [zk-skills](https://github.com/zksecurity/zk-skills)）

**插件开发：**
- `skill-development` - 技能创建指南
- `skill-improver` - 技能改进工具
- `skill-quality-reviewer` - 质量评估
- `command-development` - 斜杠命令创建
- `agent-identifier` - 代理配置
- `hook-development` - 钩子开发指南
- `mcp-integration` - MCP 服务器集成

**工具：**
- `uv-package-manager` - 现代 Python 包管理
- `planning-with-files` - 基于 Markdown 的规划
- `kaggle-learner` - 从 Kaggle 解决方案中学习

### 命令（50+）

**研究命令：**
| 命令 | 用途 |
|------|------|
| `/research-init` | 启动研究启动工作流（5W1H、文献综述、Gap 分析） |
| `/analyze-results` | 分析实验结果（统计检验、可视化、消融实验） |
| `/rebuttal` | 从审稿意见生成系统化 rebuttal 文档 |
| `/presentation` | 创建会议演讲大纲 |
| `/poster` | 生成学术海报设计方案 |
| `/promote` | 生成推广内容（Twitter、LinkedIn、博客） |

**开发命令：**
| 命令 | 用途 |
|------|------|
| `/plan` | 创建实施计划 |
| `/commit` | 使用 Conventional Commits 提交 |
| `/code-review` | 执行代码审查 |
| `/tdd` | 测试驱动开发工作流 |
| `/build-fix` | 修复构建错误 |
| `/verify` | 验证更改 |
| `/checkpoint` | 创建检查点 |
| `/refactor-clean` | 重构和清理 |
| `/learn` | 提取可重用模式 |
| `/sc` | SuperClaude 命令套件（20+ 命令） |

### 代理（14 个专业）

**研究代理：**
- **literature-reviewer** - 文献搜索、分类和趋势分析
- **data-analyst** - 自动化数据分析和可视化
- **rebuttal-writer** - 系统化 rebuttal 写作，语气优化
- **paper-miner** - 从成功论文中提取写作知识

**开发代理：**
- **architect** - 系统架构设计
- **build-error-resolver** - 修复构建错误
- **code-reviewer** - 审查代码质量
- **refactor-cleaner** - 移除死代码
- **tdd-guide** - 指导 TDD 工作流
- **kaggle-miner** - 提取 Kaggle 工程实践
- **bug-analyzer** - 深度代码执行流分析和根因调查
- **dev-planner** - 实施规划和任务拆解

**设计与内容代理：**
- **ui-sketcher** - UI 蓝图设计和交互规范
- **story-generator** - 用户故事和需求生成

## 快速开始

### 多运行时支持

Claude Scholar 支持两个运行时环境：

| | Claude Code | Codex |
|---|------------|-------|
| **技能** | 35 个（完整） | 27 个通用 + 6 个参考 |
| **钩子** | 5 个自动化 | 不适用（using-claude-scholar 技能替代） |
| **命令** | 50+ 斜杠命令 | 不适用（直接使用技能） |
| **代理** | 14 个专业 | 14 个（通过 `spawn_agent`） |
| **安装** | 克隆 / 插件 | 仅符号链接（原生技能发现） |

### 安装选项

#### Claude Code 安装

选择适合您需求的安装方式：

##### 选项 1：插件安装（推荐）

通过 Claude Code 插件管理器安装：

```bash
# 第一步：添加 marketplace
claude plugin marketplace add OniReimu/claude-scholar

# 第二步：安装插件
claude plugin install claude-scholar@claude-scholar
```

**优势**：自动组件发现、版本跟踪、通过 `claude plugin update` 便捷更新。

**包含**：所有 35 个技能、50+ 命令、14 个代理、5 个钩子和项目规则。

##### 选项 2：完整安装（Git Clone）

通过克隆到 `~/.claude` 进行完整设置：

```bash
# 克隆仓库（--recursive 拉取 vendored skills：
# scientific-figure-making、fireworks-tech-graph、circom-auditor）
git clone --recursive https://github.com/OniReimu/claude-scholar.git ~/.claude

# 已经克隆但没拉 submodule？
git -C ~/.claude submodule update --init --recursive

# 重启 Claude Code CLI
```

**包含**：所有 35 个技能、50+ 命令、14 个代理、5 个钩子和项目规则。

##### 选项 3：最小化安装

仅核心钩子和基本技能（加载更快，复杂度更低）：

```bash
# 克隆仓库
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar

# 仅复制钩子和核心技能
mkdir -p ~/.claude/hooks ~/.claude/skills
cp /tmp/claude-scholar/hooks/*.js ~/.claude/hooks/
cp -r /tmp/claude-scholar/skills/ml-paper-writing ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/research-ideation ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/results-analysis ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/review-response ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/writing-anti-ai ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/git-workflow ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/bug-detective ~/.claude/skills/

# 清理
rm -rf /tmp/claude-scholar
```

**包含**：5 个钩子、7 个核心技能（完整研究工作流 + 基本开发）。

##### 选项 4：选择性安装

选择和选择特定组件：

```bash
# 克隆仓库
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar
cd /tmp/claude-scholar

# 复制您需要的内容，例如：
# - 仅钩子
cp hooks/*.js ~/.claude/hooks/

# - 特定技能
cp -r skills/latex-conference-template-organizer ~/.claude/skills/
cp -r skills/architecture-design ~/.claude/skills/

# - 特定代理
cp agents/paper-miner.md ~/.claude/agents/

# - 项目规则
cp rules/coding-style.md ~/.claude/rules/
cp rules/agents.md ~/.claude/rules/
```

**推荐用于**：想要自定义配置的高级用户。

#### Codex 安装

```bash
# 克隆仓库
git clone https://github.com/OniReimu/claude-scholar.git ~/claude-scholar

# 运行安装脚本（创建符号链接，迁移旧版 AGENTS.md）
chmod +x ~/claude-scholar/scripts/install-codex.sh
~/claude-scholar/scripts/install-codex.sh
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/OniReimu/claude-scholar.git $HOME\claude-scholar
& "$HOME\claude-scholar\scripts\install-codex-windows.ps1"
```

**安装内容：**
- 创建符号链接：`~/.agents/skills/claude-scholar` → `skills/`
- 检测并迁移旧版 `~/.codex/AGENTS.md`
- 通过 `git pull` 更新，无需重新安装

详细 Codex 安装指南请参阅 [.codex/INSTALL.md](.codex/INSTALL.md)。

### 系统要求

- Claude Code CLI 或 Codex CLI (v0.91+)
- Git
- （可选）Node.js（用于钩子）
- （可选）uv、Python（用于 Python 开发）

### 首次运行

安装后，钩子提供自动化工作流辅助：

1. **每次提示**触发 `skill-forced-eval` → 确保考虑适用技能
2. **会话开始**时使用 `session-start` → 显示项目上下文
3. **会话结束**时使用 `session-summary` → 生成带有建议的工作日志，并附带 orchestrator 状态/事件摘要
4. **会话停止**时使用 `stop-summary` → 提供状态检查

## 项目规则

### 论文策略引擎

在 `policy/` 中定义：
- `policy/rules/` 是论文写作约束（图表、LaTeX、引文、实验、投稿合规）的唯一真相源。
- 规则卡采用 frontmatter 元数据（`id`、`layer`、`artifacts`、`phases`、`check_kind`、`enforcement`）+ 必要正文段落（`Requirement`、`Rationale`、`Check`、`Examples`）。
- 分层模型：`core`（全局必守）、`domain`（领域特定）、`venue`（会议/期刊特定）；覆盖配置在 `policy/profiles/*.md`。
- v1 的 SoK 规则通过 profile 激活（如 `policy/profiles/security-sok-sp.md`），包含语义规则 `SOK.TAXONOMY_REQUIRED`、`SOK.METHODOLOGY_REPORTING`、`SOK.BIG_TABLE_REQUIRED`、`SOK.RESEARCH_AGENDA_REQUIRED`。
- 当前限制：`policy/lint.sh --profile` 仅加载单个扁平 profile 文件（暂不支持 inheritance/composition）。
- 校验与执行流程：
  - `bash policy/validate.sh`：结构与集成校验
  - `bash policy/lint.sh`：可机器执行的规则检查
- skills/commands 通过 `<!-- policy:RULE_ID -->` marker 关联规则。

### 代码风格

由 `rules/coding-style.md` 强制执行：
- **文件大小**：最大 200-400 行
- **不可变性**：配置使用 `@dataclass(frozen=True)`
- **类型提示**：所有函数都需要
- **模式**：所有模块使用 Factory & Registry
- **配置驱动**：模型仅接受 `cfg` 参数

### 代理编排

在 `rules/agents.md` 中定义：
- 可用的代理类型和用途
- 并行任务执行
- 多视角分析

### 安全规则

在 `rules/security.md` 中定义：
- 密钥管理（环境变量、`.env` 文件）
- 敏感文件保护（禁止提交 token、密钥、凭证）
- 通过钩子进行提交前安全检查

### 实验可复现性

在 `rules/experiment-reproducibility.md` 中定义：
- 随机种子管理，确保可复现性
- 配置记录（Hydra 自动保存）
- 环境记录和检查点管理

## 贡献

这是个人配置，但欢迎您：
- Fork 并适应您自己的研究
- 通过 issue 提交错误
- 通过 issue 建议改进

## 许可证

MIT 许可证

## 致谢

使用 Claude Code CLI 构建，并由开源社区增强。

### 参考资料

本项目受到社区优秀工作的启发和构建：

- **[everything-claude-code](https://github.com/anthropics/everything-claude-code)** - Claude Code CLI 的综合资源
- **[AI-research-SKILLs](https://github.com/zechenzhangAGI/AI-research-SKILLs)** - 研究导向的技能和配置
- **[zk-skills](https://github.com/zksecurity/zk-skills)**（MIT，zkSecurity）- ZK 电路安全技能；`circom-auditor` 通过 `vendor/zk-skills` submodule 引入

这些项目为 Claude Scholar 的研究导向功能提供了宝贵的见解和基础。

---

**面向数据科学、AI 研究和学术写作。**

仓库：[https://github.com/OniReimu/claude-scholar](https://github.com/OniReimu/claude-scholar)
