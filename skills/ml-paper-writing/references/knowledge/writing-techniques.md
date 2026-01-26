# Writing Techniques and Patterns

This file contains actionable sentence patterns, transition phrases, and writing techniques extracted from successful ML conference papers.

---

## Transition Phrases

### Literature Review Transitions
**Source:** Various NeurIPS/ICML papers

**Introducing Problems:**
- "However, these methods suffer from [limitation]."
- "Despite recent progress, [challenge] remains unsolved."
- "While existing approaches address [aspect], they struggle with [issue]."

**Presenting Solutions:**
- "To address this, we propose..."
- "We overcome this limitation by..."
- "Our key insight is that..."

**Connecting to Related Work:**
- "Building on [prior work], we extend..."
- "Unlike approaches that [method], we instead..."
- "Following the success of [paper], we apply..."

### Methods Section Transitions
**Source:** "BERT: Pre-training of Deep Bidirectional Transformers", NAACL (2019)

**Describing Components:**
- "Our model consists of two main components: [A] and [B]."
- "We divide our approach into [N] stages: [list]."

**Explaining Rationale:**
- "We choose this architecture because..."
- "This formulation allows us to..."
- "Motivated by [intuition], we design..."

### Results Section Transitions
**Source:** "Attention Is All You Need", NeurIPS (2017)

**Presenting Findings:**
- "Our method achieves [result], outperforming baselines by [margin]."
- "As shown in Table 1, our approach..."
- "Figure 2 demonstrates that..."

**Analyzing Results:**
- "These results suggest that [insight]."
- "Notably, we observe that..."
- "This improvement indicates that..."

### Discussion Transitions
**Source:** "Language Models are Few-Shot Learners", GPT-3 (2020)

**Interpreting Findings:**
- "These findings reveal that..."
- "This performance gap suggests that..."
- "The strong correlation between...indicates..."

**Connecting to Broader Context:**
- "Beyond the specific task, our results imply..."
- "This has important implications for..."

**Acknowledging Limitations:**
- "It is important to note that our study is limited to..."
- "While these results are promising, several questions remain..."

---

## Sentence Patterns

### Claim Presentation
**Source:** "Attention Is All You Need", NeurIPS (2017)

**Strong Claims:**
- "We show that [approach] achieves [result]."
- "We demonstrate that [method] outperforms..."
- "We prove that [technique] converges to..."

**Nuanced Claims:**
- "Our results suggest that [factor] contributes to..."
- "We observe that [phenomenon] emerges when..."
- "Experiments indicate that [approach] is particularly effective for..."

### Technical Description
**Source:** "Adam: A Method for Stochastic Optimization", ICLR (2015)

**Algorithm Description:**
- "Formally, we optimize [objective] using [method]."
- "The update rule for [parameter] is given by..."
- "We modify the standard [approach] by..."

**Implementation Details:**
- "In practice, we implement [feature] as..."
- "For computational efficiency, we approximate..."
- "We initialize [parameters] using..."

### Results Presentation
**Source:** "BERT: Pre-training of Deep Bidirectional Transformers", NAACL (2019)

**Quantitative Results:**
- "Our model achieves [score] (±[std]), improving over..."
- "On [dataset], we obtain [result], compared to..."
- "We observe a [percentage]% improvement over baselines."

**Statistical Reporting:**
- "Results are averaged over N runs with different seeds."
- "Standard deviations are shown in parentheses."
- "The improvement is statistically significant (p<0.01)."

---

## Clarity Techniques

### Active Voice Usage
**Source:** Various well-written papers

**Passive (avoid):**
- "The model was trained using..."
- "Experiments were conducted on..."

**Active (prefer):**
- "We trained the model using..."
- "We conducted experiments on..."

**Guideline:** Use active voice for actions you performed. Use passive for general facts or when the actor is unclear.

### Specificity Over Generality
**Source:** "Attention Is All You Need", NeurIPS (2017)

**Vague (avoid):**
- "This approach improves performance."
- "The method learns good representations."

**Specific (prefer):**
- "This approach improves accuracy by 15%."
- "The method learns representations that transfer to downstream tasks."

**Guideline:** Be quantitative whenever possible. Use specific numbers and metrics.

### Signposting
**Source:** "BERT: Pre-training of Deep Bidirectional Transformers", NAACL (2019)

**Section Openings:**
- "We now describe our model architecture."
- "We evaluate on three tasks: [list]."
- "The results suggest three key insights:"

**Internal Structure:**
- "First, we [action]. Next, we [action]. Finally, we [action]."
- "Our approach has three stages: [A], [B], and [C]."

**Guideline:** Use explicit signposting to help tired reviewers follow your paper.

---

## Common Phrase Templates

### Opening Abstract
**Good Examples:**
- "We introduce [method], a novel approach for [task]."
- "We present [method], which achieves [result] by [mechanism]."
- "We propose [framework] to address [challenge]."

**Avoid:**
- "In this paper, we study..." (generic)
- "Large language models have..." (overused opening)

### Introducing Related Work
**Good Examples:**
- "Recent work has shown promise in [area] [refs]."
- "Several approaches have been proposed for [task] [refs]."
- "The standard approach to [problem] is [method] [refs]."

### Describing Experiments
**Good Examples:**
- "We evaluate on [datasets], comparing against [baselines]."
- "We conduct ablation studies to validate [component]."
- "To verify [claim], we experiment with [variations]."

### Presenting Results
**Good Examples:**
- "Table 1 shows that our method outperforms all baselines."
- "As shown in Figure 3, performance improves as [factor] increases."
- "Our method achieves state-of-the-art on [task/metric]."

### Discussing Limitations
**Good Examples:**
- "Our approach has limitations: [constraint]."
- "We note that our method is currently restricted to [condition]."
- "A key limitation is [issue], which we leave for future work."

---

## Writing Principles

### From Top Papers

**Clarity First:**
- "Make it easy for reviewers to understand your contribution."
- "Use concrete examples and specific language."
- "Avoid vague or ambiguous statements."

**Rigorous Presentation:**
- "Provide enough detail for reproduction."
- "Include error bars and statistical tests."
- "Show negative results when relevant."

**Storytelling:**
- "Your paper tells a story: problem → approach → solution → impact."
- "Make the narrative clear in the introduction."
- "Each section should advance the story."

**Honesty:**
- "Acknowledge limitations explicitly."
- "Don't overclaim results."
- "Trust reviewers to appreciate honesty."

---

## Notes

- **Adapt patterns**: These templates can and should be adapted to your specific context
- **Venue matters**: Some venues prefer certain styles (check venue-specific guides)
- **Consistency**: Use consistent terminology throughout
- **Tone**: Maintain professional, objective tone
- **Length**: Keep transitions concise; don't over-explain

**Attribution:** All patterns extracted from analyzed papers with source citations for traceability.
