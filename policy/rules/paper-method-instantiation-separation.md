---
id: PAPER.METHOD_INSTANTIATION_SEPARATION
slug: paper-method-instantiation-separation
severity: warn
locked: false
layer: core
artifacts: [text, equation]
phases: [writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [PAPER.METHOD_NOVELTY_WEIGHT]
constraint_type: guardrail
autofix: none
---

## Requirement

Method 讲**机制**：方法里的量用符号表示，设计选择写明理由。**实例化**放在 Setup。本卡说的 Setup 指 **Experiments 章节的第一个子章节**（仓库模板里是 `\subsection{Experimental Settings}`，实现细节写在其中的 `Implementation Details.` 段），不是独立的一节。Experiments 从第二个子章节起报告实验现象与结果，下文的 Results 指这些子章节。实例化包括：实验实际用的取值（$\lambda=0.1$、$k=5$、学习率、batch size、epoch、层数）、作为 backbone 的具体模型、prompt 模板、硬件与运行时间。穷尽列表放附录，代码交给 artifact。

Method 里出现的是符号，必要时说明它在权衡什么（"$\lambda$ sets how far utility may be traded for forgetting"）。Setup 给出它的取值；如果做过敏感性分析，也写取值范围和结论出处。

**判据：改变这个值，改变的是方法是什么，还是只改变了这次实验怎么跑？** 后者是实例化，搬到 Setup。

**留在 Method 的例外：**

1. **方法定义的一部分**：对称损失里的 1/2、定义方法结构的常数（"a two-layer projection" 本身就是设计）。
2. **分析推导出来的值**：$k=\lceil\log n\rceil$ 来自 Theorem 2，推导属于方法。符号式定义（$k=2f+1$）同理。
3. **取值本身就是贡献或被主张的性质**："works with a single sample per prompt" 是主张，不是调参。
4. **问题或威胁模型的参数**：作为符号留在 Method（隐私预算 $\varepsilon$）；实验中取的具体值（$\varepsilon \in \{1, 4, 8\}$）进 Setup。

**实验内容不属于 Method**：结果数字与消融结论搬到 Experiments 后面的结果子章节，数据集统计与 baseline 描述搬到 Setup（`EXP.ABLATION_IN_RESULTS`）。

**搬家不是删除。** 每个从 Method 移出的值都要在 Setup、附录或 artifact 里有落点，复现信息一条不少。`paper-self-review` 的复现检查只问"是否有落点"，不问"Method 是否写全"。

## Rationale

Method 写的是一个对任何合理实例都成立的方法，Setup 写的是这一次实验选了哪个实例。两者混在一起，读者分不清哪些是方法本身、哪些是这次的设置：$\lambda=0.1$ 写在目标函数旁边，读起来像方法依赖这个值。审稿人会追问"为什么是 0.1"，而这个问题本应由 Setup 里的敏感性分析回答。

这也是 agent 写 Method 最常见的失误来源。agent 往往一边看着 config 文件一边写，数值就在手边，写进去还会被"可复现性"的要求奖励。一个只问"Method 是否足以复现"的审查环节会把这种倾向放大，改两三轮之后，机制被实例化细节淹没，创新点反而看不出来了（`PAPER.METHOD_NOVELTY_WEIGHT`）。

## Check

- **画像**：`uv run skills/claim-architecture-review/scripts/method_profile.py <paper> --show` 列出 Method 节里右侧是纯数值的赋值（`k = 2f+1` 这类符号式定义不计），以及 Setup 节的对应计数。有范文画像（`architecture-review/exemplars.md`）时，与范文的区间对照。
- **LLM 逐条判**：Method 里每一个具体取值、具体模型名、实现选择，按判据归类为"方法定义 / 推导值 / 主张 / 问题参数 / 实例化"。实例化的给出目标位置（Setup 的哪一段，或附录），并确认目标位置存在。
- **反向检查**：Setup 里是否缺了 Method 用到的符号的取值？搬家后每个符号都要有值。
- **实验内容泄漏**：Method 里是否出现结果数字、消融结论、数据集或 baseline 描述？
- **结构执行**：`claim-architecture-review` 的 M-pass（P1 跑到 Method 节时）给每段标 `method_role`，`instantiation` 和 `evaluation-leak` 段在 relocation-map 里给出搬家计划。

## Examples

### Pass

```latex
% Method：符号 + 它在权衡什么
The forgetting penalty is scaled by $\lambda$, which sets how far the
optimiser may trade utility on $\mathcal{D}_r$ for forgetting on
$\mathcal{D}_f$.

% Setup：取值 + 敏感性出处
We use $\lambda = 0.1$ for all models. Appendix~\ref{app:sens} shows that
forgetting quality varies by less than 1.2 points for
$\lambda \in [0.05, 0.5]$.
```

### Fail

```latex
% Method 里写死了实验取值、backbone 和优化器设置
We set $\lambda = 0.1$ and fine-tune Llama-3-8B with AdamW
(learning rate $2\times10^{-5}$, batch size 32) for 3 epochs on the
forget set, which reduces forget accuracy to 4.1\%.
% 最后半句是结果，属于 Results
```

## Conflicts

- `PAPER.METHOD_NOVELTY_WEIGHT`：本卡把实例化移出 Method，那张卡决定留下的篇幅怎么分配。两张卡一起执行，Method 剩下的是机制和设计理由。
