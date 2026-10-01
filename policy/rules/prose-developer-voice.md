---
id: PROSE.DEVELOPER_VOICE
slug: prose-developer-voice
severity: warn
locked: false
layer: core
artifacts: [text, figure, table]
phases: [writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: lint_script
params: {}
conflicts_with: [PROSE.NO_INTERNAL_PROVENANCE, PAPER.OUTCOME_LOGIC, EXP.TAKEAWAY_BOX]
constraint_type: guardrail
autofix: none
lint_targets: "**/*.tex"
coverage_note: "the patterns locate engineering slang (dump the scores, spin up, kick off) and a tool or process as the acting subject (the pipeline calls, our harness feeds, the scoring script discards). Script names as nouns and hardcode are deliberately left out: on 745 real manuscript files they hit artifact-availability statements and security objects (hardcoded secrets) far more often than developer voice. Data-processing narrative written in ordinary verbs (we parse the outputs, deduplicate, and drop ...) and captions written as procedure are invisible to them and are the judgment layer's to catch. Every hit is a candidate: in a systems paper the pipeline may be the research object."
lint_patterns:
  - pattern: "\\bdump(?:s|ed|ing)? (?:the|all|each|every|its|their|our|them|it|results?|scores?|outputs?|logs?|predictions?)\\b"
    mode: match
  - pattern: "\\b(?:spun up|spin(?:s|ning)? up|kick(?:s|ed|ing)? off|monkey-?patch(?:es|ed|ing)?)\\b"
    mode: match
  - pattern: "\\b(?:[Tt]he|[Oo]ur|[Tt]his|[Ee]ach) (?:[a-z]+(?:-[a-z]+)? )?(?:pipeline|harness|script|wrapper|workflow|codebase|notebook|launcher)s? (?:then |first |also |simply |automatically |only )?(?:calls?|invokes?|dumps?|writes?|reads?|loads?|emits?|feeds?|pipes?|triggers?|spawns?|launch(?:es)?|runs?|parses?|caches?|logs?|stores?|saves?|exports?|discards?|drops?|filters?|removes?|computes?|aggregates?|merges?|plots?|grades?|scores?)\\b"
    mode: match
---

## Requirement

论文描述**研究对象**：现象、量、方法（作为数学或算法对象）、样本与总体、发现。它不记录**项目开发过程**：哪个工具调用了哪个工具、脚本做了什么、文件写到了哪里、流程分几个阶段。每一句的主语和谓语都要落在研究对象上。

### 判据：重实现测试

**一个读者只拿着这篇论文、用完全不同的代码重新实现，这句话对新的实现是否仍然成立？**

- 成立 → 这句在描述研究对象（方法的定义、实验协议、样本构成、发现）。
- 只对**我们这份代码、这次运行**成立 → developer voice，改写。

"The pipeline then calls GPT-4o and dumps the scores to disk" 对重实现者不成立：新的实现里没有这条 pipeline，也不往磁盘写分数。这句话里对重实现者有用的只有一件事：每个回答由 GPT-4o 打分。那就只写这件事。

区分两种"做"：**实验操作**是对研究对象做的事（fine-tune 三个 epoch、对每个样本采样五次、在 held-out split 上评测），属于科学协议，保留；**机械操作**是代码做的事（调用、写盘、解析、缓存、重跑、按阶段传递），不进正文。

### 七类 developer voice

| 类 | 形态 | 例 |
|---|---|---|
| D1 工具当主语 | pipeline / harness / script / workflow / wrapper 作为施动者 | "Our harness feeds each prompt to the model three times." |
| D2 工程动词 | dump, spin up, kick off, hardcode, monkey-patch, wire into, plug in, cache, re-run the job | "We hardcode the temperature to 0.7." |
| D3 流程结构当科学结构 | 内部阶段名、workflow 名、"stage 2 of our pipeline"、"the filter step" | "In the PREP stage, malformed rows are dropped." |
| D4 代码标识符代替概念 | 变量名、函数名、flag、config key 出现在句子里做名词 | "`agg_mean` exceeds 0.4 for all models." |
| D5 数据处理流水账 | 逐个操作讲数据怎么被处理，而不是讲最终样本是什么 | "We parse the logs, deduplicate by hash, and drop rows with NaN." |
| D6 caption 写成操作步骤 | 图例讲图是怎么算出来、怎么画出来的 | "We compute X for each run, average over seeds, and plot it." |
| D7 运行与调试叙事 | 跑了几次、哪次崩了、修了什么 bug 之后结果变了 | "After fixing a bug in the extractor, accuracy rose to 71%." |

D4 与 `PROSE.NO_INTERNAL_PROVENANCE` 第 5 类（schema 标识符）重叠，D7 与它的第 7 类（修订叙事）及 `PAPER.OUTCOME_LOGIC` 重叠。分工：**那两张卡判"这个 token / 这段历史能不能出现"，本卡负责"删掉之后这句话怎么用对象语言重写"**。一处文本可能两边都报，各报各的。

### 转写：把句子交还给研究对象

按顺序做四步：

1. **找对象。** 这句话里真正被讨论的是什么：一个量、一个样本集、方法的一步、一个发现？
2. **让对象做主语。** 工具和流程退出主语位置。
3. **把机械动词换成它建立的关系。** 机械操作总是在建立某种科学关系，把那个关系写出来：

   | 机械操作 | 它建立的关系 | 写成 |
   |---|---|---|
   | compute / calculate X | X 的定义 | "X is defined as …" / 直接给公式 |
   | filter / drop / discard rows | 纳入标准 + 它的效应 | "Responses without a parsed answer are excluded (k of N, x%)." |
   | aggregate / average / group by | 统计量 | "the mean over five seeds" |
   | call a model to grade / judge | 测量工具与测量对象 | "Each response is graded by GPT-4o against the reference answer." |
   | sample / split / shuffle with seed | 抽样设计 | "a stratified 80/20 split (seed 0)" |
   | run / re-run N times | 重复与方差 | "results are averaged over N independent runs" |
   | plot / visualise | 图里显示的量 | "Figure 3 shows X against Y." |

4. **对象层什么都不剩 → 删掉。** 纯机械的句子（"the script writes a JSON per model"）对读者没有内容。复现需要的细节写成协议事实放进实验设置或附录，代码本身交给 artifact（`PROSE.NO_INTERNAL_PROVENANCE` 的 provenance 归属表）。

代码标识符换成概念名或符号，并在首次使用处定义（`PAPER.DEFINE_BEFORE_USE`）。

### Caption（D6）

图例说明**图里是什么**：显示的量、编码方式（颜色、阴影、误差棒代表什么）、样本。它不说图是怎么算出来的，也不说发现了什么：发现进正文（`EXP.TAKEAWAY_BOX`）。"We compute X for each run, average over seeds, and plot it" 改成 "Mean X over five seeds; shaded bands are 95% CIs."

### 例外（必须放行）

| 情形 | 为什么合法 |
|---|---|
| 系统论文里系统本身就是研究对象 | 重实现测试下组件的设计就是贡献，"the scheduler dispatches requests to idle workers" 在描述对象 |
| 算法的过程性描述 | "The algorithm first computes …, then …" 描述的是方法，重实现者照样要这么做 |
| 实验协议 | fine-tune 设置、采样次数、评测 split、超参、随机种子，都是对研究对象的操作 |
| 计算资源与软件版本 | `REPRO.COMPUTE_RESOURCES_DOCUMENTED` 要求的复现事实："Experiments use eight A100 GPUs." |
| 引用使用的外部工具 | "We use the LM Evaluation Harness \cite{…}" 是协议的一部分 |
| artifact 可得性声明 | "Code is available at \url{…}" 是要求，不是泄漏 |
| 方法名 | 论文提出的方法有名字是正常的，它就是研究对象 |

## Rationale

这条病的来源和 `PROSE.NO_INTERNAL_PROVENANCE` 相同，而且更难察觉。Agent 往往在同一个 session 里先跑实验、再写论文：写 Results 时，工作上下文里活着的是 pipeline、脚本、阶段、文件，于是它照着自己做过的事去写。每个词都准确，每个数字都对，句子却在报告"我们的代码做了什么"，而不是"研究对象是什么样"。

`NO_INTERNAL_PROVENANCE` 抓的是有形状的痕迹：路径、文件名、snake_case。developer voice 可以一个这样的 token 都没有，"The pipeline then calls the judge and stores its verdict" 在 token 层面完全干净，所以前者的扫描器对它是盲的。

代价有三层。读者要先把机械过程翻译回科学对象才能读懂主张，这是作者该做的事。审稿人读到这类句子，会把论文归入"工程报告"：实验像是跑出来的，而不是设计出来的。更深的一层是，机械叙事会掩盖科学判断：缺失答案的样本被"drop"掉了，可那是一个纳入标准，它影响多少样本、会不会有偏，都应该写明，而不是藏在一个动词里。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。

- **Regex 定位（部分覆盖）**：`policy/lint.sh --rule PROSE.DEVELOPER_VOICE`。抓工程俚语（dump the scores、spin up、kick off）和工具或流程当施动主语（the pipeline calls、our harness feeds、the scoring script discards）。**命中是候选，不是判决**：系统论文里 pipeline 可能就是研究对象。
- **LLM 逐句检查（主要手段）**：范围是 Method、Experimental Setup、Results、Discussion、所有 caption、表格注释、附录。
  1. 对每一句做重实现测试。
  2. 不通过的，标出类别（D1–D7），按四步转写给出改写；对象层什么都不剩的，标"删除"并说明复现细节该去哪里。
  3. D5 特别检查：每一个被 drop / filter / exclude 的步骤，改写后是否给出了纳入标准和受影响的样本量。
  4. 每个 caption 单独过一遍 D6。
- **regex 覆盖不到的部分**：用普通动词写成的数据处理流水账（"we parse the outputs, deduplicate, and drop …"）和操作步骤式的 caption，regex 都抓不到，只能逐句读。`hardcode` 和作名词的 "training / evaluation scripts" **刻意不进 regex**：在 745 个真实稿件文件上，它们大多命中 artifact 可得性声明（"the repository contains the evaluation scripts"）和安全论文里的研究对象（hardcoded secrets），误报远多于 developer voice。判断层遇到它们照常判。

## Examples

### Pass

```latex
Each response is graded by GPT-4o against the reference answer
(prompt in Appendix~\ref{app:judge}). Responses without a parsable
final answer are excluded (312 of 6{,}000, 5.2\%). The exclusion rate
does not differ between conditions ($p=0.41$).

\caption{Mean refusal rate over five seeds; shaded bands are 95\% CIs.}
```

### Fail

```latex
% D1 + D2：工具做主语，工程动词
The pipeline then calls GPT-4o to grade each response and dumps the
scores to disk.
% D5：数据处理流水账，纳入标准和它的效应都藏在动词里
We parse the outputs, deduplicate by hash, and drop rows where the
extractor returned NaN.
% D4：代码标识符做名词
\texttt{agg\_mean} exceeds 0.4 for all models.
% D6：caption 写成操作步骤
\caption{We compute the refusal rate for each run, average over seeds,
and plot it with seaborn.}
```

## Conflicts

- `PROSE.NO_INTERNAL_PROVENANCE`：那张卡判 token 能不能出现，本卡判句子的主语与谓语是否落在研究对象上，并给出改写。两边可以同时命中同一句。
- `PAPER.OUTCOME_LOGIC`：D7 的运行与调试叙事同时是过程流水账。那张卡决定这段时间顺序该不该出现；本卡给出删除后的对象层写法。
- `EXP.TAKEAWAY_BOX`：caption 不写发现。D6 把操作步骤移出 caption 之后，不要用发现填补空位。
