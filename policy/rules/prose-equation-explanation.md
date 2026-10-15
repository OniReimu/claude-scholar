---
id: PROSE.EQUATION_EXPLANATION
slug: prose-equation-explanation
severity: warn
locked: false
layer: domain
artifacts: [text, equation]
phases: [writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

关键公式（非辅助性公式）要放进方法的数据流里讲，而不是孤立地给出符号对照表。三步：

1. **引入：这个公式在方法里干什么。** 公式前一句话说明它计算的量，以及方法为什么在这一步需要它。"is defined as follows" 只说了"下面有个定义"，不算引入。
2. **公式本身。**
3. **解释新出现的符号在数据流里的位置。** 对每个**首次出现**的符号，交代两件事：
   - **它从哪来**：输入数据、上游哪一步的输出、或哪个公式（`\eqref`）算出来的；
   - **它在这里起什么作用**：它加权、约束、聚合、控制的是什么，改变它会改变什么。

   前文已解释过的符号不再重复解释（`PROSE.RESTATEMENT_DILUTION`）。**不要求"每个符号一条 where 子句"**：那种格式产出的正是干巴巴的对应关系。

**粘贴测试（判据）**：把这条 where 子句原样贴到另一篇论文里同形状的公式下面，它是否仍然成立？"where $x$ is the input and $y$ is the label" 贴哪都成立，所以它没有告诉读者任何关于**本方法**的信息，判为不合格。合格的解释离开本文就不成立，因为它说的是本文数据流里的具体位置。

**公式链：每一步都要写出推理关系。** 两个相邻的展示公式之间，文字必须说明从上一式到下一式做了什么操作：代入哪个式子、用了哪个引理或不等式、因为哪个量非负、对什么求和或求导。只有 "Then we have" / "Thus" / "We obtain" 不算推理关系。"一句话配一行公式"、每句话只是把下一个公式念成文字的写法，判为违规：读者看见每一行，却看不见行与行之间为什么成立。

notation table 是查找表，不是解释。符号即使在表里定义过，在正文首次使用时仍要交代它在数据流里的位置。

## Rationale

公式在方法里是数据流的一个节点：输入从上游来，输出往下游去。读者要读懂的是这个节点在流程里的位置，不是符号和名字的对应关系。

"每个符号一条 where 子句"这个格式要求本身就在生产干巴巴的对应关系。它让作者（以及 agent）满足于给每个符号配一个名词短语，于是每个公式单独看都"定义完整"，连起来却读不出数据怎么流动，读者只能自己从符号名里重建方法。审稿人的判断恰恰来自这一层：数据流讲清了，方法的逻辑和创新点就清楚；讲不清，公式写得再全也读起来"糊成一团"。

公式链的问题相同。把推导写成一串等式、每行配一句话，看起来步骤齐全，可是最关键的那一步（为什么可以这样代入、为什么这个不等式成立）恰好被跳过了。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **画像（可选）**：`uv run skills/claim-architecture-review/scripts/method_profile.py <paper> --show` 给出 Method 节的展示公式数、"一句话配一行公式"的步数、只有 then/thus 的过渡数，并列出全部 where 子句原文，方便逐条做粘贴测试。这些都是代理量，判断仍要逐条读。
- **LLM 检查**：
  1. 关键公式前的引入句：是否说明了这个量在方法里的作用？只说"defined as follows"的不合格。
  2. 每个首次出现的符号：是否交代了它从哪来、在这里起什么作用？逐条做粘贴测试。
  3. 已解释过的符号是否被重复解释？
  4. 相邻展示公式之间：是否写出了推理关系？只有 then / thus / we obtain 的，指出缺了哪一步操作。
  5. 正文首次使用的符号是否只靠 notation table 解释？

## Examples

### Pass

```latex
To decide how much each retained sample should resist forgetting, the
objective weights the forgetting penalty by the sample's influence score
$s_i$ from Eq.~\eqref{eq:influence}:
\begin{equation}
  \mathcal{L} = \mathcal{L}_{\text{task}}(\mathcal{D}_r)
              + \lambda \sum_{i \in \mathcal{D}_f} s_i \, \ell_{\text{forget}}(x_i)
\end{equation}
where the task loss is evaluated only on the retained set $\mathcal{D}_r$,
so that utility is anchored to data the model keeps, and $\lambda$ sets how
far the optimiser may trade that utility for forgetting.
Substituting the first-order approximation of $s_i$ from
Eq.~\eqref{eq:influence-approx} makes the second term linear in the
gradient of $\ell_{\text{forget}}$, which gives the closed-form update
\begin{equation}
  \theta' = \theta - \eta \lambda \sum_{i \in \mathcal{D}_f} s_i \nabla_\theta \ell_{\text{forget}}(x_i).
\end{equation}
```

### Fail

```latex
% 引入句不说作用；where 子句贴到任何论文的同形公式下都成立；
% 两个公式之间只有 "Then"，没有推理关系
The loss is defined as follows:
\begin{equation}
  \mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{forget}}
\end{equation}
where $\mathcal{L}_{\text{task}}$ is the task loss,
$\mathcal{L}_{\text{forget}}$ is the forgetting loss, and $\lambda$ is a
hyperparameter. Then we have
\begin{equation}
  \theta^* = \arg\min_\theta \mathcal{L}.
\end{equation}
```
