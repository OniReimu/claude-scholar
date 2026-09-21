# Claude Scholar

<div align="center">
  <img src="LOGO.jpeg" alt="Claude Scholar Logo" width="100%"/>
</div>

**语言**: [English](README.md) | [中文](README.zh-CN.md)

面向学术研究和软件开发的个人 Claude Code 配置仓库 - 一个完整的工作环境。

## News

- **2026-09-21 (v1.32.3)**: `PROSE.RHYTHM_VARIANCE` 把方差当约束而不是目标。凑到 sd ≥ 10 最省力的办法是塞一句 6–8 词短话，这现在被点名为本卡最常见的违规：短句只能是可核对的主张或显式锚定的转折，且要过删除测试；拉宽分布只能重新切分已有信息。
- **2026-09-21 (v1.32.2)**: `PROSE.FRACTAL_SUMMARY` 的回翻距离改以冷进本节的审稿人为参照，而不是刚读完上一节的人；顶层 section 唯一的路线图只要说清各部分做什么，就保留。`PROSE.RESTATEMENT_DILUTION` 不再把展开当成复述。
- **2026-09-19 (v1.32.1)**: skill 校验终于在 CI 里跑起来了——`validate-skills.sh` 一直存在，却没有任何 workflow 调用它。替换版第一次跑就抓到一个 frontmatter 不是合法 YAML 的 `SKILL.md`，以及一位贡献者的 home 目录被提交在三个文件里。
- **2026-09-19 (v1.32.0)**: `research-workspace-layout` —— 把 [research-workspace 契约](https://github.com/DELONG-L/Research-Workflow-Skills) 落地到已有仓库，外加把 Overleaf 挂成 submodule 的全部踩坑。它把拓扑当作**闭集**：会长出十一个并列的 cache 目录，不是因为有人决定要十一个，而是没有任何地方规定第十二个必须去哪。
- **2026-09-14 (v1.31.0)**: `peer-review` —— 补上 Claude Scholar 唯一没有 skill 的那种评审：为某个 venue 审别人的稿子。它自身不带任何评审逻辑，在**打开稿件之前**就转交给单独安装的 [More Than Peer Review](https://github.com/DELONG-L/More-Than-Peer-Review-Skill)，所以本侧不会有任何已成形的批评越界过去。
- **2026-09-11 (v1.30.0)**: `PROSE.CAUSAL_CONNECTIVE` 拿到了它自己的论证所要求的仪器。规则卡早已证明效应完全在**密度**上（pre-GPT 语料每千词 0.18–0.28，实测草稿 3.76），而 lint 只数五个正式连接词、**从来没数过 `, so`**；现在两者都以每千词速率连同基线一起打印，扫描顺序也写死：先读密度，再逐例判断。
- **2026-09-01 (v1.28.0)**: polish 循环的出口从数量口径换成判决。`PAPER.REVISION_CLOSURE` 让自审以四选一的全文结论收尾（**STOP_REVISING**、**ONE_BOUNDED_ROUND**、**REOPEN_SUBSTANTIVE_REVISION**、**UNASSESSED**），因为「零发现就停」把本仓库唯一测量过的代价定价为零：每一轮修补都会把新的统计特征沉进稿子里。
- **2026-08-28 (v1.27.0)**: 连接词单一化在两个项目里连续出现后拿到结构性修复。三条清标点规则把关系推进词汇层，`CAUSAL_CONNECTIVE` 随即交出最安全的那一个，每轮一两处、每处都局部站得住——26 个正式连接词里 22 个落成 `therefore` 而逐例检查全绿。现在 lint 打印一张只报不判的分布表，补救阶梯也把「删除」排到了第一位。
- **2026-08-28 (v1.26.0)**: `PROSE.OVER_DEFENSIVE` 增加**句子层**，管免责式否定谓语——它原有五条都是位置与计数判断，单句全能通过，而十六句加起来就是一篇每千词 1.7 次在填免责声明的稿子。补救是一个动作而不是删除：把「我们不做 X」翻成「我们做的是 Y」。接线时暴露出引擎缺陷：lint 只在 `check_kind: regex` 时跑 pattern，于是带定位器的判断型卡片校验全绿却从未执行。
- **2026-08-28 (v1.25.1)**: 倒序系动词对比 `is not A, but B` 在 v1.25 的 A 档表里，却没有任何 pattern 覆盖——原规则锚在 `, not ` 这个字节序列上，而它只存在于正序形式。第六条 pattern 补上了；写语料用例时又发现 `not only … but also` 的未覆盖是**刻意**的（它按频次管理，单次不算违规），现在记为 `coverage_note` 而不是一个意外。
- **2026-08-27 (v1.25.0)**: `PROSE.NEGATION_CONTRAST` 分两档——零容忍与仅提示——起因是整篇稿子报出 **15** 处对比结构而实际存在 **38** 处。按各自 Requirement 列举的形式逐条探查每个 `lint_script` 规则，在另外三张卡上发现同样的收窄，而它们都在执行者从不打开的 Check 段里**声明过**；于是新增 `coverage_note` 字段把缺口写在 lint 会打印的位置，并由 `validate.sh` 双向强制。
- **2026-08-27 (v1.24.0)**: `PROSE.SEMICOLON_RESTRICTION` 补第二条补救路径，因为第一条在制造另一条规则的违规、而那条规则的修法正好把它撤销——39 个可测分号里，「拆成两句」有 19 次留下不足十词的第二分句，正是 `PROSE.THEATRICAL_SPLIT` 点名要合并回去的形状。补救改为两档、句法改写先于拆分，且第一档须同时满足四个条件，仍是少数情形（39 例中 32 例照拆）。
- **2026-08-27 (v1.23.0)**: 重锚获得距离维度：重述几节之前定义的构件是正常写作，不是 `PROSE.FRACTAL_SUMMARY` 要抓的自相似冗余，而这张卡此前完全没有距离概念，把紧贴标题上方两行的段落和三节之外的同类判成一样。修复先落在 `claim-architecture-review` P2，否则信息账本会把每一次跨节重锚都压平；且**不设任何数值阈值**——要问的是读者会不会需要回头翻，而本仓库每一次给判断加阈值，最后量到的都是仪器本身。
- **2026-08-27 (v1.22.0)**: 三处修正，全部来自认真读流水线自己产出的「干净」文本。`PROSE.SENTENCE_LENGTH` 在悄悄漏判，因为任何带点的 token 都被当成句子边界，而小数恰恰出现在这条规则要抓的那类句子里；新规则 `PROSE.SEMICOLON_RESTRICTION` 做成 builtin，因为纯 pattern 会在行内公式上误击；`PROSE.EM_DASH_RESTRICTION` 的补救本身是错的。两张卡现在共用一条判准：换标点不算修复，结构必须变。
- **2026-08-27 (v1.21.0)**: 新套件 `policy/test-pipeline.sh` —— 第一个测试规则**合起来**做了什么的套件，用植入缺陷的夹具打分，三个断言动词加上「输出里每个数字都必须在输入中已存在」。它同时断言**未编辑的草稿必须失败**，而这个阴性对照当场就证明了自己的价值：两条被标注为捕捉顺序陷阱的断言，在只做行级编辑时照样通过——标签会朝你的意图漂移，而不是朝它实际测的东西。
- **2026-08-27 (v1.20.0)**: `PROSE.SEMANTIC_IDLING` 增加**改写契约**，来自对一份规格说明自带的十条 gold 改写的盲判：**10 条里 6 条仍然违规**，其中两条正是它自己正确诊断为循环、然后又循环地改写掉的段落。所以只有当还有命题幸存时才输出改写——压缩能提纯有内容的段落，对没内容的只是缩短，而简短的空洞读起来像结论——75–85% 的比例记为观察值，绝不作为目标。
- **2026-08-27 (v1.19.1)**: `PROSE.SEMANTIC_IDLING` 验收评测——十段外部提供的空转文字，与十段未见过的 pre-GPT 已发表段落盲混，使判者无法推断基率：**10/10 命中，0/10 误报**。随后两处修正：Future Work 套话明确不豁免；form B 记录为在标准机理上过度触发，于是判准改写成一个问题——解释项有没有引入一个可独立测量的量？
- **2026-08-27 (v1.19.0)**: 新规则 `PROSE.SEMANTIC_IDLING` —— 每个句子都必须给出一个可证伪的命题。「空转」查下来其实是三种现象，其中两种已有归属，所以这张卡只收从所有其他卡缝里漏掉的两种：什么都没点名的长篇元叙述句，以及 `because` 从句复述自己后件的循环归因。它**禁止度量代理**，依据是本仓库自己的证据：藏在阈值后面的判断，量到的是仪器——结构重复信号自判 16 倍，换独立盲判后塌到 1.22 倍。
- **2026-08-27 (v1.18.0)**: `PROSE.ADHOC_COMPOUND_MODIFIER` 改为三态判决——**flag** / **hint** / **clear**——因为二元判决在同一组段落上同时朝两个方向出错。放行项现在也会报告：只报违规会让作者无从分辨某个词是被检查后放行、还是压根没被看到；段落长度经测量后被**否决**，它不是 tell。
- **2026-08-27 (v1.17.0)**: 打磨 `PROSE.ADHOC_COMPOUND_MODIFIER`，其中一项的分量超过其余：**`-based` 是一种构式而非生造**，把它算进去会让两个时代的区分度从 16 倍稀释到 4 倍，于是它移出默认后缀集。卡片同时写明**那个 16 倍不可作为效应量引用**；把一个复合词判为既有术语，现在必须一并给出它的先前来源——一个读者可以去核的断言。
- **2026-08-27 (v1.16.1)**: 把 `PROSE.ADHOC_COMPOUND_MODIFIER` 的局限写进卡片而不是让它心照不宣：机械的那一半谁跑都得同一个数，但「这个复合词是不是本领域既有术语」是 LLM 判断，受知识截止限制、在冷门子领域更弱。卡片带着一个自查出的误判作为证据，并把方向定为保守——拿不准就不要标记，因为在真实领域术语上误报一次，整条规则就会被关掉。
- **2026-08-27 (v1.16.0)**: 新规则 `PROSE.ADHOC_COMPOUND_MODIFIER` —— 一整天测量里唯一经得起推敲的信号。**判准是频次而非构式**：只出现一次的复合词在 2019–2021 语料是每千词 0.16，2025–2026 是 0.48；而一份 pre-GPT 区块链稿件因反复使用把这类词推到基线的 7 倍，按用量打分会把整个领域判死。它可信的原因是**不需要判断**——同日测的另外两个候选需要，其中一个在独立盲判下从 16 倍塌到 1.22 倍。
- **2026-08-27 (v1.15.1)**: 补上 `writing-anti-ai` 在表里声明却从不执行的五条 `doc` 规则，更要紧的是让这一类缺口可被机器发现——`lint_script` 规则无论 skill 正文怎么写都有正则兜底，`doc` 规则没有。写这个检查时翻出更糟的事：`set -eo pipefail` 会让一次合法地匹配不到内容的 `grep` 中途终止整轮，而截断的运行打印出的 `FAIL:` **比完整运行更少**，于是 CI 一直把提前退出读成了改善。
- **2026-08-27 (v1.15.0)**: 新规则 `PROSE.CAUSAL_CONNECTIVE`，以及一次改写了规则说法的评测。42 个被标记的句子盲判后，定位器完美（42/42），而**逐例区分力为零、继而反转**——pre-GPT 的实例被判「值得改」的比例是 64%，当代草稿是 29%。既然只有密度有差异，这条规则就以「因果精确性规则」而非「AI tell」发布，只标记三类可诊断的情形，其余一律保留。
- **2026-08-27 (v1.14.1)**: CI 从 2026-08-17 起一直是红的，三个 FAIL 只在 runner 上出现：`deprecated_by` 可解析性用 `-e` 测后继项，而 `-e` 会跟随符号链接，偏偏有三个 skill 是指向 vendor submodule 的符号链接。缺的是内容不是引用——改用 `-L` 后两种树上答案一致，workflow 也会 checkout submodule。
- **2026-08-26 (v1.14.0)**: 补两套现有测试从来没问过的问题——`policy/test-corpus.sh` 跑 88 个带标注的夹具，逐例报告漏检与误报；`policy/test-referrals.sh` 把此前靠手工核对的转介图交给机器检查。写它们时翻出六个缺陷，包括字节级字符类让 `[→←↔]` 在 macOS 解析到的 perl 路径上匹配了每一个 em dash、一条 YAML 注释静默截断整个 `lint_patterns` 块、以及被注释掉的文字仍被 lint。
- **2026-08-26 (v1.13.4)**: 让转介路径适配人们真正在用的工作流——在已有稿件上入口是 `writing-anti-ai`，所有结构性的东西只有在这个 skill 成功转介时才会到达作者，而它此前做不到。两侧现在都写明最小操作而非只给一个 skill 名；`claim-architecture-review` 的 marker 从 1 个增到 6 个，依据是**一个 skill 执行某条规则时才携带它的 marker**，而不是仅仅相关。
- **2026-08-26 (v1.13.3)**: 补上 v1.13.2 留下的跨节接缝——规则把自己限定在段落与小节，而信息账本以命题为键、完全没提枚举，于是两边都以为对方管了。另外 `validate.sh` 每轮 fork 约 1400 次 `grep`，偶发的 fork 失败与字段缺失无法区分：**一个在正确输入上偶尔失败的校验器，教会用户反复重跑直到变绿，这比不检查更糟。**
- **2026-08-24 (v1.13.2)**: `PROSE.RULE_OF_THREE` 从三元组扩到**枚举密度与重复**，起因是一个真实方法段里三个列表只有一个是三元组，而唯一的机械命中报的是逗号、不是枚举。`writing-anti-ai` 当时还在指示「优先用二或四项」——那是从通用反 AI 建议继承来的，可在学术文字里它**制造**更长的枚举墙；反向护栏现在写在卡片开头：修法是命名后引用，绝不是砍掉条目。
- **2026-08-23 (v1.13.1)**: 把叙事主动性原则里三处可做行编辑的部分接进 `writing-anti-ai`，让扫稿真能改得动——`PAPER.OUTCOME_LOGIC` 此前只接在撰写期和评审期，于是时序泄漏被*发现*却没有任何步骤去*修*。最要紧的边界随之落地：只删实现层面的弯路，因为一个跑过且其结果为主张划界的实验是证据；需要全文语境的原则被**刻意**不接进来，因为让行编辑去判它们正是有记录的过度执行失效模式。
- **2026-08-23 (v1.13.0)**: 叙事主动性层，来自同行的现场诊断：一个害怕被指摘不够周全的 agent，会替想象中的审稿人先行攻击这篇论文。三条新规则（`PROSE.SELF_UNDERMINING`、`EXP.EXPERIMENT_ROLE`、`PAPER.OUTCOME_LOGIC`）各自都带一条**完整性边界**，以免任何一条被读成掩盖证据的许可——措辞受管而披露绝不削减，实验因为不服务任何主张而被删，绝不因为数字不好看。这次审计还发现 `policy/style-guide.md` 在违反它被宣称与之平级的那些规则，于是 `validate.sh` 现在会 lint 风格指南自己的文字。
- **2026-08-17 (v1.12.0)**: 补上 **author-original 文本**的语域覆盖，来自一份已经跑过 `writing-anti-ai` 却仍有约 30 处语域缺陷的 NDSS 投稿——把其中 26 处喂给 `PROSE.INFORMAL_VOCABULARY` 的九条 pattern，命中 **0**，因为 30 处里有 29 处是多词结构，而那张卡只发了单词级正则。它被重建为五类分类法，配 LLM 判断与允许列表，同时新增 `PROSE.IDIOM_COLLISION`，以及一份记录了手写 `.tex` 扫描器产生虚假「干净」结论的四种实测方式的参考文档。
- **2026-08-15 (v1.11.0)**: 两条来自真实投稿扫描的加固。`PROSE.NO_INTERNAL_PROVENANCE` 早已存在，却仍让十一处开发期产物进了已编译的 ACM ASIA CCS 投稿——问题是三个漏洞而非缺一条规则：不在 agent 真正会读的清单里、lint 零覆盖、`severity: warn` 与它自己的理由自相矛盾。另一条是新的 `PROSE.REGISTER_PRESERVATION`，它在**差分**上执行，因为语域是编辑的属性而不是词的属性：一次压缩引入九处违规，而既有 pattern 一处都没抓到。
- **2026-08-15 (v1.10.0)**: 反 AI 写作大改、policy 冲突审计，以及第一次有证据支撑的 eval（39/39，改版前快照 37/39，两处差异恰好落在新能力上）。四条新规则之外，还修掉一条自相矛盾的禁令、一个 autofix 修复环路，以及一条把「每文件句子数」当成「每句词数」的 `SENTENCE_LENGTH` pattern；`validate.sh` 现在把**任何 autofix 输出都不得触发另一条规则**做成机器校验的不变量。分别自 [zksecurity/zk-skills](https://github.com/zksecurity/zk-skills) 与 [AIScientists-Dev/academic-humanizer](https://github.com/AIScientists-Dev/academic-humanizer) 审阅后借入。
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

### 技能（49 个）

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
- `lineage` - 可选开启的实验脉络页面：哪些线在跑、哪些悄悄停了、哪些结果论文根本没读
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
- 创建符号链接：`~/.agents/skills/claude-scholar` 与 `~/.codex/skills/claude-scholar` → `skills/`
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
