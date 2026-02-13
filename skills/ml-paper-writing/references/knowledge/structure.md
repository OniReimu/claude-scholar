# Paper Structure Patterns

This file contains actionable patterns for organizing ML conference papers, extracted from successful publications.

---

## Introduction Patterns

### Pattern: Contribution Statement Structure
**Source:** "Attention Is All You Need", NeurIPS (2017)
**Context:** Introducing the main contribution

**Pattern:**
1. Start with broader context or problem
2. Narrow down to specific limitation
3. Present your approach as solution
4. State clear contribution upfront

**Example Template:**
```markdown
[Context/Problem]: Existing approaches struggle with [limitation] due to [reason].

[Our Approach]: We propose [method name], which [key innovation].

[Contribution]: This achieves [result] and enables [capability].
```

**Application:** Use this pattern when introducing your main contribution in the first or second paragraph of the introduction.

---

### Pattern: Bulleted Contribution List
**Source:** "BERT: Pre-training of Deep Bidirectional Transformers", NAACL (2019)
**Context:** Summarizing contributions for clarity

**Pattern:**
- Place near end of Introduction (after Related Work)
- Use 2-4 bullets
- Each bullet: 1-2 lines max (in two-column format)
- Start with strong verbs ("We propose", "We demonstrate", "We show")

**Example Template:**
```markdown
Our contributions are three-fold:
- We propose [method], which achieves [result].
- We demonstrate that [technique] improves [metric].
- We show that [approach] enables [new capability].
```

**Application:** Use this when you need to clearly delineate multiple contributions for reviewers.

---

### Pattern: Related Work Organization
**Source:** "Attention Is All You Need", NeurIPS (2017)
**Context:** Structuring literature review

**Pattern:**
- Organize methodologically, not chronologically
- Group papers by approach/assumption
- Contrast your approach with each group
- Use "One line of work uses X whereas we use Y because..."

**Example Template:**
```markdown
[Approach Category]: Several approaches use [assumption A] [refs].
[Contrast]: We adopt [assumption B] because it allows [benefit].

[Alternative Category]: Other methods focus on [aspect C] [refs].
[Positioning]: We build on this by adding [our innovation].
```

**Application:** Use this to position your work relative to existing literature without paper-by-paper reviews.

---

## Methods Section Patterns

### Pattern: Algorithm Presentation
**Source:** "Adam: A Method for Stochastic Optimization", ICLR (2015)
**Context:** Describing algorithms clearly

**Pattern:**
1. High-level overview first
2. Mathematical formulation
3. Algorithm pseudocode (if complex)
4. Implementation details

**Example Template:**
```markdown
[Overview]: We formulate [problem] as optimization. Let [objective] be our goal.

[Method]: Our approach optimizes [objective] using [technique].
Specifically, we [algorithm description].

[Algorithm]: The full procedure is shown in Algorithm 1.

[Implementation]: In practice, we [practical details].
```

**Application:** Use this when presenting novel algorithms or optimization methods.

---

### Pattern: Component Breakdown
**Source:** "BERT: Pre-training of Deep Bidirectional Transformers", NAACL (2019)
**Context:** Describing multi-component systems

**Pattern:**
- Present model architecture first
- Break down into key components
- Explain each component's role
- Show how components interact

**Example Template:**
```markdown
[Architecture]: Our model consists of [N components]: [list].

[Component 1]: The [component] module [function].
[Component 2]: The [component] layer [operation].

[Integration]: These components are stacked sequentially, with [connection pattern].
```

**Application:** Use this when describing complex architectures with multiple interacting parts.

---

## Results Section Patterns

### Pattern: Quantitative Opening
**Source:** "BERT: Pre-training of Deep Bidirectional Transformers", NAACL (2019)
**Context:** Presenting main findings

**Pattern:**
- Start with strongest quantitative result
- Use exact numbers and metrics
- Include comparison to baselines
- State statistical significance

**Example Template:**
```markdown
[Main Result]: Our method achieves [score] on [dataset], improving
over the previous best of [baseline] by [margin] (p<0.001).

[Comparison]: Compared to baselines:
- [Method A]: [score]
- [Method B]: [score]
- Ours: [score]

[Significance]: Results are averaged over N runs; standard deviations shown in parentheses.
```

**Application:** Use this to open your Results section with your strongest finding.

---

### Pattern: Table Integration
**Source:** "Attention Is All All You Need", NeurIPS (2017)
**Context:** Presenting results in tables

**Pattern:**
- Bold best results in each column
- Include direction indicators (↑↓)
- Provide table caption that stands alone
- Reference table in text before presenting

**Example Template:**
```markdown
Table 1 shows our method's performance. Our model (bold) outperforms
all baselines across datasets.

[Table content]

As shown in Table 1, we achieve state-of-the-art on [datasets].
```

**Application:** Use this when presenting comparative results in table format.

---

## Discussion Section Patterns

### Pattern: Limitations First
**Source:** "Attention Is All You Need", NeurIPS (2017)
**Context:** Acknowledging limitations proactively

**Pattern:**
- State limitations clearly in first paragraph
- Explain why limitations don't undermine core claims
- Distinguish between limitations and future work

**Example Template:**
```markdown
[Limitation Statement]: Our approach has [limitation]. Specifically,
[constraint].

[Mitigation]: Despite this, our core findings about [main contribution] remain
valid because [reason].

[Future Work]: Addressing this limitation is an important direction for
future research.
```

**Application:** Use this to acknowledge limitations honestly while maintaining paper strength.

---

### Pattern: Broader Impact Framing
**Source:** "Language Models are Few-Shot Learners", GPT-3 Paper (2020)
**Context:** Discussing wider implications

**Pattern:**
- Start with direct implications
- Expand to related domains
- Consider societal impact (if appropriate)
- End with forward-looking statement

**Example Template:**
```markdown
[Direct Impact]: Our findings suggest that [implication for domain].

[Broader Implications]: Beyond [specific domain], this approach could
enable [application in other areas].

[Future Outlook]: As [trend] continues, methods like ours will become
increasingly important for [reason].
```

**Application:** Use this when writing the final paragraphs of Discussion or Conclusion.

---

## Transition Patterns

### Pattern: Section Transitions
**Source:** "Attention Is All You Need", NeurIPS (2017)
**Context:** Moving between sections

**Pattern:**
- Introduction → Methods: "We now describe our approach."
- Methods → Results: "We evaluate our method on [tasks]."
- Results → Discussion: "These results suggest that [insight]."

**Example Template:**
```markdown
[Transition to Methods]: Having established [motivation], we present
our method.

[Transition to Results]: To validate our approach, we conduct experiments
on [datasets].

[Transition to Discussion]: The experimental results reveal several insights
about [phenomenon], which we discuss next.
```

**Application:** Use these to create smooth transitions between major sections.

---

## Notes

- **Consistency**: Maintain consistent terminology throughout the paper
- **Flow**: Each section should logically lead to the next
- **Clarity**: Make structure explicit with signposting
- **Audience**: Write for tired reviewers - make their job easy



## Kaiming He's Paper Structure Patterns

> Source: Analyzed 19 representative papers by Kaiming He

### Abstract Structure Patterns

Common opening patterns used by Kaiming He in abstracts:

**Pattern 1: Direct contribution statement**
```
We introduce [method name], a [key feature] framework for [task].
We show that [method] achieves [result] on [dataset].
```

**Pattern 2: Problem-solution**
```
[Problem] is difficult for [task]. We present [solution]
that addresses this by [key mechanism].
```

**Example** (from ResNet):
```
Deeper neural networks are more difficult to train. We present a
residual learning framework to ease the training of networks that
are substantially deeper than those used previously.
```

### Introduction Structure Patterns

**Three-part introduction:**
1. **Problem statement** (2-3 paragraphs) - Describe challenges and existing methods
2. **Method overview** (1-2 paragraphs) - Concise introduction of the solution
3. **Main contributions** (1 paragraph) - Bulleted list, 1-2 lines each

**Contribution list pattern:**
```markdown
- We propose [method], which addresses [problem]
- We demonstrate [method] achieves [performance improvement] on [dataset]
- We show that [principle] is effective
```

### Methods Section Structure

Kaiming He's methods sections typically contain:

1. **Notation definitions** - Clearly define all variables and symbols
2. **Problem formalization** - Mathematical formulation
3. **Method description** - Step-by-step algorithm explanation
4. **Implementation details** - Network architecture, training setup

**Common sentence patterns:**
- "Let us consider [variable] as [definition]"
- "Formally, we define [formula]"
- "We hypothesize that [hypothesis]"
- "To the extreme, [extreme case]"

### Experiments Section Structure

1. **Experimental setup** - Datasets, evaluation metrics, implementation details
2. **Main results** - Core performance comparisons
3. **Ablation studies** - Component analysis
4. **Visualization analysis** - Figures and charts

**Result description patterns:**
- "Table X shows that [result]"
- "Fig. Y illustrates that [observation]"
- "Our method achieves [metric] on [task]"
- "This represents a [X]% improvement over baseline"

### Related Work Organization

Kaiming He favors **thematic organization** over chronological ordering:

**Good organization:**
- "One line of work uses [Method A] [refs], whereas we use [Method B]"
- "[Method A] [refs] assumes [assumption], but we show [counter-evidence]"

**Avoid:**
- "X et al. introduced [method]. Y et al. improved [method]"

---

## Paper Organization Standards

### Standard Section Order

Recommended section organization for security/AI venue papers:

```
1. Introduction
2. Related Works
3. Preliminary (optional; may be merged with Related Works if brief)
4. System Model
5. <Your Method / Your Design>
6. Experiments
   6.1 Experiment Settings
   6.2 Results
7. Discussion (Security Analysis for security papers; Limitations also here)
8. Conclusion / Future Work
```

**Key notes:**

- **Preliminary** is optional. If the content is brief, merge it with Related Works. When merged, use this recommended structure: Definitions first, then Existing Works, then a Comparison Table between your method and prior work
- **Related Works must include a comparison table** (see templates in `references/latex-style-guide.md`)
- **Experiments** must be split into Settings and Results subsections; do not mix them together
- **Discussion** is where Security Analysis and Limitations belong; do not place Limitations in the Conclusion

### Section Organization Anti-patterns

- **Do not over-subdivide**: No more than 8 top-level sections (excluding References)
- **Do not mention "draft" or "xxx-style" in the paper body**:
  - Forbidden: "This manuscript provides a complete pre-experimental draft for TIFS-style review"
  - Forbidden: "This is an xxx-style paper"
  - Forbidden: "This draft presents..."
  - Correct approach: State the content directly without meta-descriptions
- **Do not include an overly long Paper Organization at the end of the Introduction**:
  - 2-3 sentences maximum: "The rest of this paper is organized as follows. Section 2... Section 3..."
  - No need to write a description for every single section
