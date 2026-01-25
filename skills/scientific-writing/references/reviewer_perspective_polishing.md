# Reviewer Perspective: Scientific Paper Polishing Guide

## Overview

This guide provides a comprehensive approach to polishing scientific papers from a reviewer's perspective. Use this checklist to identify and fix common issues that lead to rejections or negative reviews.

## The Reviewer's Mindset

When reviewers read your paper, they are asking:
1. Is this contribution novel and significant?
2. Are the methods sound and reproducible?
3. Do the results support the conclusions?
4. Is the writing clear and accessible?
5. Would this work benefit the field?

**Polish from this perspective**: Anticipate and address reviewer concerns before submission.

## Pre-Submission Checklist

### Content Review

**Novelty and Significance**:
- [ ] Clear statement of what is new
- [ ] Comparison with prior work (what's different?)
- [ ] Explanation of why this matters
- [ ] Position in the broader field

**Technical Soundness**:
- [ ] Methods described in sufficient detail
- [ ] Experimental design appropriate
- [ ] Statistical analysis correct
- [ ] Results support all claims
- [ ] Limitations acknowledged

**Clarity and Accessibility**:
- [ ] Abstract captures key contributions
- [ ] Introduction sets up the problem clearly
- [ ] Methods are reproducible
- [ ] Results are presented logically
- [ ] Discussion interprets appropriately

### Structure Review

**Title**:
- [ ] Descriptive and concise
- [ ] Contains key keywords
- [ ] Avoids hype or overstatement
- [ ] Accurately reflects content

**Abstract**:
- [ ] Problem stated clearly
- [ ] Approach described briefly
- [ ] Key results summarized (with numbers)
- [ ] Implications stated
- [ ] Within word limit
- [ ] No undefined abbreviations

**Introduction**:
- [ ] Starts with broad motivation
- [ ] Narrows to specific problem
- [ ] Reviews relevant literature
- [ ] Identifies knowledge gap
- [ ] States contributions clearly
- [ ] Provides paper roadmap

**Methods**:
- [ ] Sufficient for reproduction
- [ ] Justification for design choices
- [ ] Statistical methods explained
- [ ] Ethical considerations included
- [ ] Data/software availability stated

**Results**:
- [ ] Logical flow from primary to secondary outcomes
- [ ] Figures/tables referenced in text
- [ ] Statistical significance reported
- [ ] Effect sizes included
- [ ] No interpretation in this section

**Discussion**:
- [ ] Interprets results without overstatement
- [ ] Compares with prior work
- [ ] Acknowledges limitations
- [ ] Suggests implications
- [ ] Proposes future directions

**References**:
- [ ] All cited works included
- [ ] All references cited in text
- [ ] Formatted consistently
- [ ] Recent literature included
- [ ] Key works not missing

## Common Issues and Fixes

### Issue 1: Unclear Contribution Statement

**Problem**: Reviewers can't quickly identify what's new.

**Weak examples**:
- "We study X problem." (Doesn't say what's new)
- "Our approach improves performance." (Doesn't say how or why)
- "We present a novel method." (Doesn't explain novelty)

**Strong examples**:
- "We introduce a new attention mechanism that reduces computational complexity from O(n²) to O(n log n) while maintaining 98% of model accuracy."
- "Unlike prior work that focuses on X, our approach addresses Y, achieving 15% improvement on Z benchmark."
- "We demonstrate, for the first time, that phenomenon X can be explained by mechanism Y, challenging the prevailing theory Z."

**Fix**: Add a clear contributions paragraph in Introduction:
```
This paper makes three key contributions:
(1) We propose [novel method/approach], which [key innovation].
(2) We demonstrate that [key finding] through [experiment type].
(3) We show that our approach achieves [quantitative improvement]
    compared to [baseline methods] on [benchmark datasets].
```

### Issue 2: Weak Related Work

**Problem**: Related work reads like a laundry list or misses key papers.

**Weak approach**:
- Chronological summary: "Smith (2020) did X. Jones (2021) did Y. Brown (2022) did Z."
- Missing connections to your work
- Not organizing by theme

**Strong approach**:
- Organize by theme or approach
- Contrast with your work clearly
- Highlight what's missing from prior work
```
Recent approaches to X can be categorized into three types:

Methods based on [A-type] [Smith 2020, Jones 2021] achieve
strong performance but require [limitation]. [B-type] methods
[Brown 2022] address this limitation but introduce [new problem].

Our approach differs from prior work in two key ways. First,
unlike [A-type] methods, we [difference]. Second, compared to
[B-type] methods, we [difference]. To our knowledge, this is
the first work to [novel aspect].
```

### Issue 3: Missing Experimental Details

**Problem**: Reviewers can't assess validity or reproduce results.

**Required information**:
- Dataset description (size, source, characteristics)
- Train/validation/test splits
- Hyperparameter values and selection method
- Evaluation metrics and why chosen
- Baseline methods and implementation details
- Statistical tests used
- Computational resources (GPU hours, etc.)

**Fix**: Create dedicated "Experimental Setup" subsection:
```
4.1 Experimental Setup

Datasets. We evaluate on three datasets: [name] ([citation]),
which contains [description]; [name], with [characteristics];
and [name], described in [citation]. For each dataset, we use
the standard train/validation/test split of [ratio].

Implementation Details. Our model is implemented in [framework]
and trained on [hardware]. We use [optimizer] with learning rate
[value], batch size [value], and train for [epochs] epochs.
Hyperparameters are selected via [method].

Baselines. We compare against [methods], implemented as
described in [citations] or using [official code].

Evaluation Metrics. We use [metrics] to evaluate [aspect],
following prior work [citation]. Statistical significance is
assessed using [test] with [threshold].
```

### Issue 4: Overclaiming Results

**Problem**: Conclusions not fully supported by data.

**Weak language**:
- "Our method significantly outperforms all baselines."
- "This approach will revolutionize the field."
- "We prove that X is superior to Y."

**Strong language**:
- "Our method outperforms baselines on 4 out of 5 datasets, with
  statistically significant improvements (p < 0.01)."
- "These results suggest that X may be promising for [application]."
- "We demonstrate that X outperforms Y under [specific conditions]."

**Fix**: Quantify claims and qualify appropriately:
- Use numbers and statistical tests
- Specify conditions under which claims hold
- Acknowledge when results are preliminary
- Avoid absolute language ("always", "never", "prove")

### Issue 5: Poor Figure Design

**Problem**: Figures unclear, hard to interpret, or missing context.

**Common issues**:
- Labels too small or missing
- Inconsistent colors or notation
- No explanation in caption
- Figure not referenced in text
- Too much information in one figure

**Fix checklist**:
- [ ] All axes labeled with units
- [ ] Legend explains all symbols/colors
- [ ] Caption makes figure self-explanatory
- [ ] Resolution sufficient for publication
- [ ] Colorblind-friendly palette
- [ ] Consistent style across figures
- [ ] Referred to in text (with context)

**Example good caption**:
```
Figure 1: Overview of our proposed architecture. The model consists
of three components: (a) encoder network that extracts features
from input X, (b) attention module that computes weights for each
feature, and (c) decoder that reconstructs output Y. Arrows indicate
information flow. Blue boxes show learnable parameters. See Section
3 for detailed descriptions.
```

### Issue 6: Awkward Phrasing and Grammar

**Problem**: Distracts from content, creates poor impression.

**Common issues**:
- Subject-verb disagreement
- Run-on sentences
- Inconsistent terminology
- Missing articles (a, an, the)
- Unclear antecedents ("this", "that")

**Fix**: Read paper aloud and mark awkward passages. Then revise:
- Break long sentences (>30 words) into shorter ones
- Add transitions between ideas
- Ensure pronouns have clear antecedents
- Use consistent terminology
- Run grammar checker (but don't rely solely on it)

**Example revision**:
- *Before*: "The method which we propose achieves better results
  than the existing methods which are based on the traditional
  approach that has been used for many years."
- *After*: "Our proposed method outperforms traditional approaches
  that have dominated the field for decades. Specifically, we
  achieve X% improvement on Y benchmark."

### Issue 7: Missing or Incomplete Limitations

**Problem**: Reviewers suspect authors are hiding weaknesses.

**What to include**:
- Scope of applicable conditions
- Known failure modes
- Computational costs
- Data requirements
- Assumptions that may not always hold
- Potential negative societal impacts

**Example**:
```
Limitations. Our approach has several limitations. First, the
computational cost scales as O(n²), making it impractical for
very large datasets (n > 1M). Second, our method requires [resource],
which may not be available in all settings. Third, while our
experiments focused on [domain], it's unclear whether the results
generalize to [other domains]. Finally, the method assumes [assumption],
which may not hold in all real-world scenarios.

Future work could address these limitations by [direction].
```

### Issue 8: Inconsistent Notation

**Problem**: Confusing or inconsistent mathematical notation.

**Check**:
- All variables defined at first use
- Same symbol used consistently
- Fonts used correctly (vectors bold, matrices uppercase, etc.)
- Subscripts/superscripts consistent
- Physical units included

**Create notation table** if using many symbols:
```
Notation:
x  - Input feature vector (d-dimensional)
y  - Output label
θ  - Model parameters
L  - Loss function
η  - Learning rate
```

## Polishing Workflow

### Stage 1: Self-Review (1-2 weeks before submission)

1. Print full paper and read with fresh eyes
2. Check against submission checklist
3. Verify all figures and tables
4. Confirm all references cited and included
5. Run spell-check and grammar-check

### Stage 2: Peer Review (1 week before)

1. Send to colleague for feedback
2. Ask specific questions:
   - Were contributions clear?
   - Was anything confusing?
   - Did you notice any errors?
   - What would strengthen the paper?

3. Revise based on feedback

### Stage 3: Final Polish (2-3 days before)

1. Check formatting requirements (page limit, font, etc.)
2. Verify all links work
3. Ensure all supplementary materials included
4. Final proofread
5. Generate submission PDF and verify

## Quick Fixes for Common Problems

### Problem: Paper feels disorganized

**Quick fix**: Add "roadmap" sentence at end of Introduction:
```
The remainder of this paper is organized as follows:
Section 2 reviews related work. Section 3 describes our
proposed approach. Section 4 presents experimental results.
Section 5 discusses implications and limitations.
```

### Problem: Reviewers won't understand significance

**Quick fix**: Add "Significance Statement" (for broad-impact journals):
```
Significance. This work is significant because it [explains
advance]. Unlike prior approaches that [limitation], our method
[advantage]. These findings could enable [applications] and
advance understanding of [field].
```

### Problem: Abstract doesn't capture contributions

**Quick fix**: Rewrite abstract with this structure:
```
[Problem] is a critical challenge in [field]. Existing approaches
[limitation]. Here we show that [key advance]. Our approach [method]
achieves [quantitative result], demonstrating [implication].
These findings represent a step toward [broader goal].
```

### Problem: Writing feels dry or monotonous

**Quick fix**: Vary sentence structure and add transitions:
- Mix simple and complex sentences
- Use transition words (however, moreover, in contrast)
- Start some sentences with prepositional phrases
- Occasional rhetorical questions (sparingly)

## Reviewer Pet Peeves to Avoid

1. **Wall of text**: Break up long paragraphs
2. **Missing baseline comparison**: Always compare to SOTA
3. **Overstating claims**: Qualify appropriately
4. **Typos in title/abstract**: Creates bad first impression
5. **Undefined abbreviations**: Define at first use
6. **Incomplete figures**: Make self-explanatory
7. **Missing acknowledgments**: Thank funding sources, colleagues
8. **Inconsistent formatting**: Follow template precisely
9. **Unclear contribution**: State explicitly and prominently
10. **No limitations discussion**: Acknowledge weaknesses

## Final Checklist Before Submission

**Content**:
- [ ] Novel contribution clearly stated
- [ ] Related work comprehensive
- [ ] Methods reproducible
- [ ] Results support all claims
- [ ] Limitations acknowledged
- [ ] Significance explained

**Writing**:
- [ ] No grammar or spelling errors
- [ ] Consistent terminology throughout
- [ ] Clear, logical structure
- [ ] Transitions between sections
- [ ] Accessible to target audience

**Formatting**:
- [ ] Within page/word limits
- [ ] Follows journal/conference template
- [ ] All figures and tables included
- [ ] References formatted correctly
- [ ] Supplementary materials complete

**Ethics**:
- [ ] Anonymity preserved (for double-blind)
- [ ] Competing interests declared
- [ ] Funding acknowledged
- [ ] Ethical approvals included
- [ ] Data availability stated

## Post-Submission: Preparing for Reviews

**Anticipate likely reviewer comments**:
1. What will reviewers question most?
2. What experiments might they request?
3. What clarifications might be needed?

**Prepare additional materials**:
- Extended results
- Additional ablations
- Alternative visualizations
- Clarified method descriptions
- Code for reproducibility

This preparation enables faster, more effective revisions.
