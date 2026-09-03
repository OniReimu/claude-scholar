# Claude Scholar

<div align="center">
  <img src="LOGO.jpeg" alt="Claude Scholar Logo" width="100%"/>
</div>

**语言**: [English](README.md) | [中文](README.zh-CN.md)

面向学术研究和软件开发的个人 Claude Code 配置仓库 - 一个完整的工作环境。

## News

- **2026-09-01 (v1.28.0)**: polish 循环的出口从数量口径换成判决。`PAPER.REVISION_CLOSURE` 要求 self-review 跑完必须对全稿出一条四档取一的判决——**STOP_REVISING**（不存在足以重启修订的实质根因；剩余 findings 进方向性建议或投稿准备轴，与“停”共存）/ **ONE_BOUNDED_ROUND**（一个局部实质问题，判决必须同时给出根因、节段范围、限定一轮）/ **REOPEN_SUBSTANTIVE_REVISION**（claim 不成立、实验支撑有缺口、结构性论证断裂——这三类改措辞收不了）/ **UNASSESSED**（缺节、占位文本、正文截断；对半份稿子说 STOP_REVISING 是伪造判决）。旧出口是「全部 pass？」，而数量口径在本仓库自己的证据面前站不住：`PROSE.CAUSAL_CONNECTIVE` 记录了连接词单一化是**跨 pass 累积**出来的——没有任何一条单规则在任何一遍上失效，逐处判都全绿，26 个正式连接词里 22 个是 `therefore`。推论是：**pass 次数本身就是风险变量**——每多跑一轮修补都会把新的统计特征攒进稿子，而「零 findings 才停」把这个代价定义成了零。逐条护栏封不住它（护栏是逐处的，累积是跨处的），所以 closure 放在所有单条规则之上，只裁「还要不要再跑一轮」，这也是它 `conflicts_with` 为空的原因。三类判决理由明文禁止：**findings 数量**（“还有 N 处违规”是 lint 的输出，不是稿件质量的读数）、**抽象完美标准**（“还能更好”对任何一份稿子都成立，因此对任何一份都不提供信息）、**接收概率猜测**（在预测一个本卡观察不到的量）。**不设任何数值阈值**——与 `PROSE.SEMANTIC_IDLING` 拒绝嵌入相似度与压缩率目标同构，且这里多一层：降低 findings 数最省力的方式是让规则更难触发，而不是让稿子更好。防循环条款堵住明显的洞：**同一根因不得第二次触发限定轮**——一轮限定改写没能解决它，就说明它不是局部问题，要么按新证据判 REOPEN，要么作为已知限制留在稿上。
- **2026-08-28 (v1.27.0)**: 连接词单一化在连续两个项目复现后获得结构性修复。机制是系统性的：三条标点清理规则（`EM_DASH_RESTRICTION`、`SEMICOLON_RESTRICTION`、`MIDSENTENCE_COLON`）的修法都把标点隐含的关系——"所以""即""因此"——赶到词汇层，而 `CAUSAL_CONNECTIVE` 递上菜单里最安全的一项：`therefore` 排第一、定义最宽，卡里"说不出是哪一种 = 没想清楚"的判据在它身上永不触发，因为 `therefore` 永远说得出。既有护栏（"不要批量替换"）守的是单次 pass；实际发生的是**跨 pass 累积**——每遍清理引入一两个，逐处判都成立，最终 26 个正式连接词 22 个是 `therefore` 而每处检查全绿。补三个缺口：**lint 在 `CAUSAL_CONNECTIVE` 下打印 report-only 计数表**（`therefore N · thus N · hence N · consequently N · accordingly N`）——刻意不判违规、刻意不设占比阈值，全为逻辑蕴含的稿子本该 `therefore` 占多数；表的存在是让那句一行判据——**这真的全是同一种因果吗？**——对着真实数字问，而不是对着印象。**修法阶梯重排**：先删（因果已由上一句承担或与 `However`/`Because` 双重标记——实测 10 处 8 处）、再从属化、最后按语义换型，绝不为凑分布硬塞 `hence`。**缺声明的两张标点卡**补上与 `CAUSAL_CONNECTIVE` 的双向 `conflicts_with`（分号卡 v1.24 已有），并写明：清理一批标点之后，必须复测连接词分布。
- **2026-08-28 (v1.26.0)**: `PROSE.OVER_DEFENSIVE` 增加**句子层**——免责式否定谓语。既有五条判据全是位置与次数：一句摆放得当的 *"We do not select an optimal policy"* 五条全过，而十六句叠起来通篇在报备（实测 1.7/千词）。卡自己的排除清单还在保护这个形状：免责句长得就像 scope 陈述，而 scope 陈述是豁免的。先分诊，只有 A 类在射程内：**A** 作者/本文产物主语 + 拒绝主张谓语；**B** 事实性否定（定义、发现、数值结果）——实测 33 处里 17 处属 B，不划界会被一起铲掉。修法一步而非删除：把「我们不做 X」翻成「我们做的是 Y」——`PROSE.SELF_UNDERMINING` 三步阶梯的第三步落到句子层，信息不减且常常更多（*policy selection remains with the venue* 说出了谁来选）。三类必须保留：发现本身是否定命题（翻正面=over-claim）、带理由的方法学取舍（翻正面要凭空编出用了什么）、唯一列出被排除项的地方。禁止机械全翻——实测 12 处改写用了六种结构，统一化只是把报备指纹换成模板指纹。随卡发一条定位 pattern（作者主语 + 拒绝谓语），只定位不裁决：同一形状同时承载 A 类免责、保留类取舍与 B 类事实，主语表劈不开它们。接线时暴露一个引擎缺陷：**lint 只在 `check_kind: regex` 时运行 pattern**——判断卡带上正则定位器后，解析通过、验证通过、从不执行。门改为按 pattern 存在与否（9c 已保证凡带 pattern 者皆有意）。与 `PROSE.NEGATION_CONTRAST` 在 neither/nor 与 bound 主语句上的交叉命中，如实标注为定位层成本，未压掉。
- **2026-08-28 (v1.25.1)**: 反序系动词形 `is not A, but B` 在 v1.25 的档 A 表里列了，却没有任何一张卡的 pattern 覆盖它。原始规则锚在字节序列 `, not ` 上——它只存在于正序里；反序形的逗号在 `but` 前面，而 `PROSE.NEGATIVE_PARALLELISM` 的 pattern 要求双代词或 `not just`。由用户追问"v1.25 之前为什么很多 is xxx not xxx 没抓到"暴露。新增第六条 pattern 覆盖之，lookahead 把 `not just/only/merely/simply` 让给排比卡；让步性 but 从句（*is not complete, but the gap is small*）会被定位、由判断层放行，与档 B 同一契约。写 corpus case 时又撞出下一条：`not only … but also` 被本卡的排除条款声称归 NEGATIVE_PARALLELISM 管，而那张卡的 pattern 同样不含它——查证后是刻意的：该形式是频次管理（每篇 ≤2 次），单次命中不是违规。这个"刻意"现按 9d 补成排比卡的 `coverage_note`，并由一条 corpus case 钉死沉默。
- **2026-08-27 (v1.25.0)**: `PROSE.NEGATION_CONTRAST` 分两档，并且让"口径不全"这件事在所有存在它的地方变得可见。卡文列了三种句式却只有一条 pattern，于是一份全稿报出 **15** 处对比构造而实际有 **38** 处——漏得最多的 `rather than`（17 处）恰恰是卡文用散文警告过"不要把 `, not Y` 反射性改成这个"的那一种。卡**声明过**这个省略（"合法用法太多，不做硬 regex"），孤立看是站得住的取舍，但在分档可用之后就是错的：**档 A**（系动词对比 `X is A, not B` / `neither … nor`，是 `PROSE.NEGATIVE_PARALLELISM` 的 `It is not X, it is Y` 近亲）零容忍，**档 B**（`rather than` / `instead of` 挂在动词上）是提示，因为那里排除项常常就是主张本身——*"we mark X as unavailable rather than guessing its timing"* 是方法学取舍不是修辞。档 B 只作提示就解掉了精确度顾虑，于是五种形式现在全部由机械层定位。反向护栏写进卡里：**同句内没有正面对项的单纯否定谓语不在范围内**——*"that difference is not an effect estimate"*——实测那份稿子 16 处命中里有 5 处属此类，铲掉它们会让作者关掉整条规则。具体修复之外还有一个一般性的。把每条 `lint_script` 规则拿它自己 Requirement 列举的形式去探，发现 `PROSE.SUPERFICIAL_ING_SUFFIX`、`PROSE.ABSTRACT_AGENCY`、`PROSE.TRAILING_AFTERTHOUGHT` 同样窄——而三条**都声明过**，声明在 Check 段里，而按 lint 输出干活的人从不打开那一段。它们的 pattern 未动（理由成立且没有测量反驳）；改为新增 `coverage_note` 字段把缺口写在起作用的地方，由 lint 在规则头下直接打印，并由 `validate.sh` **9d** 双向强制：散文里声明了缺口却没写 note、或写了 note 而卡文从不解释，都判失败。
- **2026-08-27 (v1.24.0)**: `PROSE.SEMICOLON_RESTRICTION` 补第二级修法——因为第一级会制造另一条规则的违规，而那条规则的修法恰好是把它撤销。全稿去分号实测（47 处，39 处可测）：只给"拆成两句"这一条路，拆出的第二子句词数中位数 **10 词**，**19/39** 短于 10 词、**8/39** 短于 8 词；最差一例 *"The point-estimate rule passes. Interval equivalence does not."*（4 词 + 4 词）正是 `PROSE.THEATRICAL_SPLIT` 点名的违规型，**而那张卡的规定修法就是用 `but`/`yet`/`, and` 合并回去**。两张卡互相指着对方：这边逼你拆，那边叫你接，而本卡的措辞（"不是把分号换成逗号"）把出口堵死了。措辞混淆了两个不同的动作——`;` → **裸逗号**是 comma splice，仍然禁止；`;` → **逗号 + 连接词**引入了一个承载语义的词，是句法改变而非标点替换。修法现为两级：**① 句法改变**（从属化 / 关系从句 / 逗号 + 连接词——与 `PROSE.CAUSAL_CONNECTIVE` 对同一句法现象在另一种标点下的既有阶梯对齐）与 **② 断句**。用 ① 需四条同时成立：第二子句 < 10 词、关系是对比/让步/互补且连接词不当胶水、断句会留下回声碎片、合并后逗号 ≤ 3——最后一条是实测逼出来的，一处两半各带三项列表的句子合并后达 5 个逗号，把分号违规换成了 `PROSE.COMMA_OVERUSE` 违规。并列的规格罗列仍走断句；39 处里 **32 处**保持 ②，所以 ① 是少数情形而非新默认。反向护栏：连接词按语义分布，不得统一——全篇一律 `while` 只是把分号指纹换成 `while` 指纹并压低句长方差。不发 `fix_patterns`，理由与 `PROSE.CAUSAL_CONNECTIVE` 的 `autofix: none` 相同：连接词的选择取决于语义关系，机械替换会改错意思，那比留着分号更糟。
- **2026-08-27 (v1.23.0)**: 重锚获得距离维度。一节要用到几节之前定义的构件（RQ 集合、命名术语、框架阶段）时把它重新说一遍，是正常人类写法，不是 `PROSE.FRACTAL_SUMMARY` 要抓的自相似冗余。而这张卡此前**完全没有距离概念**：紧贴在所预告标题上方两行的段落，和隔了三节真正需要重新锚定的段落，判定完全相同。修正**先落在 `claim-architecture-review` P2**，因为危害在那里才是系统性的——ledger 只看得见「一个命题两个 home」，其设计目标就是把这种情况压成一个，于是它会清掉每一处跨节重锚，而线编只是偶然撞上。P2 现在保留 **re-anchor home**，在 `other-homes` 里标 `(re-anchor)` 并写出 canonical home 的位置——一个没有说明的幸存者，读起来和一个漏掉的重复没有区别。两个条件缺一不可：**自足**（不持有「标签→内容」绑定也读得懂——*"The study asks three questions. First, what does a reviewer's visible action contain?"* 合格；*"RQ1 establishes which public events can be interpreted"* 不合格，它预设了重锚本该还清的那笔债，**只点标签在任何距离下都不合格**）＋**距离足以让读者本来要回翻**。刻意**不设数字阈值**：写成数字会把问题变成「数中间隔了几节」，而真正该问的是「读者要不要翻回去看」，且本仓库试过的每一个「给判断套阈值」最后测的都是仪器自己。单位取节不取页——页码随排版而变。距离只解除冗余合并，**不豁免 `PROSE.SEMANTIC_IDLING` 形态 A**：什么都没断言的回顾隔多远都是空转。卡里原有的 Survey/SoK taxonomy 导航段豁免，事后看正是这条规则被硬编码到单一文体的特例。未新增 pattern（自足性与距离都不是正则能判的），语料补上 2×2 四格作精确度 case，锁死五条既有 pattern 在四格上全部保持沉默。
- **2026-08-27 (v1.22.0)**: 三处修正，全部来自认真读流水线自己产出的"干净"文本。**`PROSE.SENTENCE_LENGTH` 一直在静默漏检**：它用 `[^.!?\s]+` 数词，任何带点的 token 都被当成句子边界，`$\beta=0.9$` 把一个 55 词的句子劈成 30 和 24 两段，两段都不超阈值。小数恰恰住在这条规则最该抓的那些句子里，所以漏检是系统性的。词类改为允许"后面不跟空白"的点，并包进原子组——同样的嵌套量词形状此前已在 `PROSE.COMMA_OVERUSE` 上造成灾难性回溯。缩写仍然测不到，记为 `@XFAIL`：`i.e.` 与句号逐字节相同，没有正则能区分。**新规则 `PROSE.SEMICOLON_RESTRICTION`** 禁止正文段落中的分号——两个独立子句拆成两句，而不是换个标点。实现为 builtin，因为纯 pattern 会在 `p(y \mid x; \theta)` 上误报，实测参考语料中 7–16% 的分号位于行内数学内部；列表项、`\;` 细空格宏与 verbatim 环境同样先剥离，且剥离数量会被报出来。卡里明写这是**作者选定的风格约束，不是被验证过的痕迹**：密度差（作者本人 2.62 → 5.60/千词；arXiv 1.32 → 1.80）如实记录，但两侧语料不可比且未做盲评。**`PROSE.EM_DASH_RESTRICTION` 的修法本身是错的**——它把"逗号插入语"列为合法替代，于是 `--- a 12.5x reduction` 变成 `, a 12.5x reduction`，这条规则本要去掉的尾巴原样留下。两张卡现在共用同一判据：替换标点不算修法，结构必须变。新规则当场命中本仓库自己语料中的四处，其中一处正是当初写来当 `PROSE.CAUSAL_CONNECTIVE` **正确改法范例**的句子；那处与两个 Pass 示例已一并更正。流水线 fixture 里那段"必须逐字节存活"的陷阱段自带一个分号和一个 55 词句子，与新规则直接矛盾——现已清理成除被测项外每个维度都干净，参考输出重新录制。
- **2026-08-27 (v1.21.0)**: 新套件 `policy/test-pipeline.sh` —— 第一次测规则**之间**的协作。既有三套各测一条规则或一处机制，没有一套能回答 `claim-architecture-review` → `writing-anti-ai` 是否删对、留对、且顺序没颠倒。一个 fixture = 埋了缺陷的 draft + 埋了误报陷阱 + **运行前封存**的答案；agent 那一半手动（`PROMPT.md`），打分那一半确定性——三个断言词（`GONE` / `KEPT` / `VERBATIM`）外加"输出里每个数字都必须在输入里出现过"的伪造检查，编造数字一律失败，与断言无关。种子 fixture 是 378 词的 section，含五处结构缺陷、五处句层缺陷、三处必须原样存活的陷阱，得分 22/22——其中 `because` 从句引入独立可测量的那段陷阱**逐字节存活**。CI 跑不了 agent，因此给录制的参考输出打分，使断言不会随规则演进而腐烂；同时断言**未加工的原稿必须失败**——只通过一切的打分器什么都没报告。这条反向控制当场就赚回了自己：有两条断言的标签声称能抓顺序颠倒（句级命中藏在本该先被结构层删掉的段落里），喂进只做词级编辑的输出后发现它们恰恰在那种情况下照样通过，真正的守卫是段落自己的主题短语。**标签会朝着你的本意漂移，而不是朝着它实际测的东西。**
- **2026-08-27 (v1.20.0)**: `PROSE.SEMANTIC_IDLING` 增加 **Rewrite 契约**，来源是把出题方自己给的 gold rewrite 送回规则里测。十段"轱辘话"对应的十条 gold rewrite 与十句真实句子盲混判定：**10 条里 6 条仍然违规**。四条仍是形态 A（`Our framework consistently outperforms baseline methods on standard benchmarks` 依然没有 baseline、benchmark、幅度），两条仍是形态 B（`Minimizing intermediate computation time reduces overall end-to-end inference latency`——两个量是同一个量）；而这两条恰恰是出题方**自己诊断对了**是回环的那两段，诊断对了、改写仍是回环。失败集中在无命题存活的段落，通过集中在有命题存活的段落。因此：**只有命题存活才输出改写**，存活的必须 100% 保留；一句都没存活时输出删除或转诊——**压缩对有内容的段落是提纯，对没内容的段落是把空话变短**，而短的空话读起来更像结论。**压缩率被否决为指标**：75–85% 是删完填充后观察到的结果而非应瞄准的数，当指标会制造为凑比例而删内容的压力，这与 Check 拒绝阈值是同一个理由。另修正：**verdict 逐段不逐句**——提取逐句进行，但孤立单句会系统性高估，真实已发表散文在单句粒度误报 3/10、段落粒度 0/30，牺牲者正是下一句就兑现的 topic sentence。出题方的十个诊断标签**未采纳**：它们塌成四个机制，其中两个属于 `PROSE.RESTATEMENT_DILUTION` 与 `PROSE.ELEGANT_VARIATION`，照抄等于给四件事起十个名并捅穿三张卡的边界。
- **2026-08-27 (v1.19.1)**: `PROSE.SEMANTIC_IDLING` 验收评测——十段构造的"轱辘话"，与十段未用过的、来自某作者 pre-GPT 已发表论文的真实段落盲混（判官无从推断配比）：**10/10 全中，0/10 误报**。分档是有内容的而非一刀切：七段 `escalate`（无一句存活，整段升级到 `claim-architecture-review` P1）对三段 `flag-B`，而这三段正是首句确实带命题、回环出现在下游的那三段。由此改两处。**Future Work / Conclusion 套话明确不豁免**——判官不得不自己推出这一点，而 `paving the way for progressive improvements in subsequent research endeavors` 与任何论文的未来工作段可以互换，按定义就是形态 A。另记入一条 limitation：形态 B **会在标准机制上过判**，把 `because the representations remain invariant across distribution shifts` 判成回环，而 invariance → OOD reliability 是真实因果。判据因此改写成一个提问——解释项有没有引入独立可测的量？未被测量的机制主张是含糊，不是空转。
- **2026-08-27 (v1.19.0)**: 新规则 `PROSE.SEMANTIC_IDLING` —— 每句必须新增可证伪的内容。需求是"轱辘话"（最被感知为 AI 味的写法），但查证结果是：它不是一个现象而是三个，而其中最大的一块已经有主。同义反复归 `PROSE.RESTATEMENT_DILUTION`（命题层）+ `PROSE.ELEGANT_VARIATION`（术语层）+ `claim-architecture-review` P2（跨节）。真正从所有卡中间漏下去的是**长**的元叙述句（`PROSE.FRACTAL_SUMMARY` 只抓节首节尾的预告，`PROSE.ANNOUNCEMENT_SENTENCE` 只抓**短**标签句，`PROSE.FILLER_PHRASES` 只抓在册短语）和**因果回环**（`because` 之后的解释项是被解释项的换词重述）。二者共用同一个测试——这句断言了什么可能为假的内容——所以是一张卡的两种形态 A 与 B。复述是**一个命题说了两遍**，空转是**一个命题都没有**，因此删除测试不迁移：空转句删掉后信息同样零损失，但修法是补事实而不是删重复。本卡**禁止指标代理**——不做嵌入余弦、不做 filler-token 占比、不做命题密度阈值——依据是本仓库自己的证据：给判断套阈值，测的是仪器自己（结构性重复信号自评 16x，fresh 模型盲评后塌到 1.22x）。anti-ai 与 `claim-architecture-review` 在这里是真联动：一段之内**多数句子**空转时整段升级到 P1（该 pass 现已带上本卡 marker，判 `merge`/`delete`），因为逐句"具体化"一个本身没有内容的段落，产出的是更好听的空话。盲评 26 段（fresh 模型，无标注）：某作者 pre-GPT 已发表论文 **20/20 全部放行**，构造违规 **4/4 全中**，豁免陷阱 **2/2 放行**（下一句即兑现的 topic sentence + ethics 声明）。更有信息量的不是召回率，而是三次让位——判官把短标签句让给 `ANNOUNCEMENT_SENTENCE`、把节预告让给 `FRACTAL_SUMMARY`、把重叠句对让给 `RESTATEMENT_DILUTION`，没有据为己有。
- **2026-08-27 (v1.18.0)**: `PROSE.ADHOC_COMPOUND_MODIFIER` 改为三值判定，因为二值在中间地带实测必然失准：对一组目标段落，二值规则同时**报重** `gradient-norm-dependent`（表意清晰但笨重）和**漏掉** `pre-training/fine-tuning`（标准但可优化）——一次错两个方向。三档为 **flag**（造词，读者须停下解码；给两个不同类型的具体改法）· **hint**（合法且清晰，只是笨重；一句提示，作者可以不理）· **clear**（领域既有术语，**必须写出领域先验出处**）。**已放行的也要报**：只报违规时，作者分不清 `read/write` 是"查过合格"还是"根本没看到"——沉默不等于干净，这个教训本仓库今天学了三次。斜杠并列（`load-balancing/routing module` 是"一个模块做两件事"还是"两个模块之一"？）并入本卡而非另立新规，因为测量不支持：arXiv 两组 0.52 → 1.04/千词，而一位本地作者的 pre-GPT 稿件密度**高于**其当前 draft。标准对偶 clear，只有真正有歧义的才 flag。段落长度经查证后**否决**：两组中位数同为 57 词，均值 74 对 71，CV 0.86 对 0.80。该需求真正指向的东西——一个该并进邻段的两句话段落——是 `claim-architecture-review` 既有的 `merge`/`split` verdict，因此 `writing-anti-ai` 改为转诊过去，并写明**段长本身不是痕迹**。
- **2026-08-27 (v1.17.0)**: 依需求规格打磨 `PROSE.ADHOC_COMPOUND_MODIFIER`，其中一条的价值远超其余。**`-based` 是构式不是造词**：`X-based` 就是 "based on X" 的压缩写法，可自由组合，两个年代都在大量使用——2019–2021 语料的 25 个 hapax 里 **22 个是 `-based`**，2025–2026 的 83 个里 43 个是。含它时两组差 4.05x，去掉后 16.27x，因此移出默认后缀集，`LINT_ADHOC_INCLUDE_BASED=1` 可折回（`concatenation-based` 说明 `-based` 仍可能被造得生涩）。**16.27x 不可作效应量引用**——pre-GPT 侧非 `-based` 的 hapax 总共只有 3 个，卡里写明了。另外三项机械改进：只在**前置定语位置**报（`a model-agnostic estimator` 报，`the estimator is model-agnostic` 不报），少报 8% 而分离度不变；复合词后接缩略语定义、或各段首字母全大写的，视为显式命名行为直接放行；左项本身为复合结构（`out-of-distribution-driven`）**打风险标记而非当过滤器**，因为它只覆盖一成命中。判定侧：认定某词为既有术语时**必须举出领域先验出处**——一个读者可以核查的主张，而不是只能接受的断言；改写指引改为**按意图分流**（图省事 → 动词化；拟引入概念 → 显式命名或换常规搭配），不再是一串优先级。频次放宽到 2 次是**测过而非假定**：精度不变（两侧均约 50%），召回 +33%，因此记为可采纳但排在后缀分层之后——真正解决误报的是后者。
- **2026-08-27 (v1.16.1)**: 把 `PROSE.ADHOC_COMPOUND_MODIFIER` 的局限**写进卡里**，而不是留作默认。本卡的机械部分（统计只出现一次的复合词）换谁跑都是同一个数；另一半——判定被标出的复合词是不是本领域既有术语——**由 LLM 依据训练知识判断**：不可复现、有知识截止、冷门子领域识别率低。卡中现已写明，并保留一处自身误判作为证据：`brokerage-oriented` 被列为造词，而 *brokerage* 是社会网络分析的既有概念。因此判定方向定为保守——**不确定时不报**，因为误报一个真实领域术语会让整条规则被作者关掉。两条机械替代方案连同其状态一并记录：以 pre-GPT 语料建通用词表**已实测不可行**（56 篇仅 89 个类型，`agent-based` / `model-agnostic` / `sharpness-aware` 均不在）；查论文自身参考文献有吸引力但**未完成测量，不作任何结论**。
- **2026-08-27 (v1.16.0)**: 新增规则 `PROSE.ADHOC_COMPOUND_MODIFIER`——今天一整天的测量里**唯一经得住检验的信号**。连字符复合修饰语（`X-based` · `X-aware` · `X-driven`）本身是标准技术英语；不标准的是**临时造一个、只用一次**：读者要现场解码 `community-shift-aware`，解码完这个词再不出现，成本付了收益为零。40 篇 arXiv 源码实测，**恰好只出现一次**的复合词从 2019–2021 的 0.16/千词升到 2025–2026 的 0.48/千词，**3.1 倍**。**判据是频次不是构造**：一份本地 pre-GPT 区块链稿件的出现率是 arXiv 基线的 7 倍，但几乎全部来自 `blockchain-based` / `sharding-based` / `PBFT-based` 的反复使用——按总量判会把整个领域误伤，按一次性判则不会，这条反例正是本卡判据的来源。由于判据是**全文频次**，逐行正则无法表达，因此以 `lint.sh` builtin 形式实现而非 `lint_patterns`。**它可信的原因是它不需要任何判断。** 同日检验的另两个候选都需要：`, so` 因果连接词密度与结构性重复（命题复述 / 集合重列）在同一批语料上分别只有 1.00x 与 1.22x——而结构层那个在"规则作者兼判官"时一度显示 16x，换独立盲判后塌到 1.22x。hapax 计数是纯机械的，换谁跑都是同一个数。**它仍然不是 AI 检测器**：分布严重重叠，个案无判别力。已知误报类保留在语料里而非压制：恰好只出现一次的领域既有术语（`sharpness-aware minimisation`）会被 builtin 报出，由语义层清除。
- **2026-08-27 (v1.15.1)**: 补上 `writing-anti-ai` 表里声明却从不执行的五条 `doc` 类规则——`ELEGANT_VARIATION` · `FORMATTING_RESTRAINT` · `ANAPHORA_ABUSE` · `GERUND_FRAGMENT_LITANY` · `SHORT_PUNCHY_FRAGMENTS`——更要紧的是**把这类缺口变成机器可查的**。`enforcement: lint_script` 的规则有正则兜底，正文写不写都跑得到；`doc` 类没有任何兜底，只在表里点名就等于**声称了并不存在的覆盖**。`validate.sh` 新增 9b 节检查它，按 skill **显式 opt-in**（`<!-- policy-table:checklist -->`）——因为 Policy Rules 表有两种，`using-claude-scholar` 为 Codex 索引全部 101 条却一条都不执行。写这个检查时暴露出更严重的问题：`set -eo pipefail` 下，一个合法地匹配不到东西的 `grep` 会让运行**中途终止**，而被截断的运行打印的 `FAIL:` 行**比完整运行更少**——CI 门禁于是把提前退出读成了"变干净了"。现在 validate.sh 用文件里定义的 section 数自校完成度，CI 在 summary 缺失时直接失败，而不是去数崩溃前打印了什么。五条里 `ELEGANT_VARIATION` 最值得写出来：它是另外两条修法所依赖的约束——`RULE_OF_THREE` 要求给枚举集合命名后全文引用同一个名字，`INFORMAL_VOCABULARY` 要求用稿件已有的措辞替换。
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
