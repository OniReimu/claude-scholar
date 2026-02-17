---
name: paper-self-review
description: This skill should be used when the user asks to "review paper quality", "check paper completeness", "validate paper structure", "self-review before submission", or mentions systematic paper quality checking. Provides comprehensive quality assurance checklist for academic papers.
version: 0.1.0
---

# Paper Self-Review

A systematic paper quality checking tool that helps researchers conduct comprehensive self-review before submission.

## Core Features

### 1. Structure Review

Check whether all sections of the paper are complete and conform to academic standards:
- Does the Abstract include problem, method, results, and contributions?
- Does the Introduction clearly articulate research motivation and background?
- Is the Method detailed enough to be reproducible?
- Do the Results sufficiently support the conclusions?
- Does the Discussion address limitations and future work?

### 2. Logic Consistency Check

Verify the logical coherence of the paper:
- Do research questions match the methodology?
- Does the experimental design support the research hypotheses?
- Are result interpretations reasonable?
- Are conclusions supported by evidence?

### 3. Citation Completeness

Check the completeness and accuracy of citations:
- Are all citations present in the references?
- Is the reference format consistent?
- Are key related works cited?
- Do citations accurately reflect the original content?

### 4. Figure/Table Quality

Evaluate the quality and effectiveness of figures and tables:
- Do all figures/tables have clear captions + labels (no in-figure title text)?
- Do figures/tables support the text narrative?
- Are figures/tables clear and readable?
- Do formats comply with journal/conference requirements?

### 5. Writing Clarity

Check writing clarity and readability:
- Is the language concise and clear?
- Is technical terminology used appropriately?
- Are sentence structures clear?
- Is paragraph organization logical?

### 6. LaTeX Math Conformance

Check whether math notation follows project rules:
- Are display equations written with `\begin{equation}...\end{equation}`?
- Is raw `$$...$$` or `\[...\]` avoided for display equations?
- Are inline equations written with `$...$` where appropriate?
- Are variable-like tokens longer than 3 letters wrapped with `\text{}` in math mode?

## Quality Checklist

Use this checklist for systematic paper self-review:

```
Paper Quality Checklist:
- [ ] Abstract includes problem, method, results, contributions
- [ ] Introduction clearly states research motivation
- [ ] Method is reproducible
- [ ] Results support conclusions
- [ ] Discussion addresses limitations
- [ ] All figures/tables have captions + labels (no in-figure title text)
- [ ] Display equations use `equation`; no `$$...$$` or `\[...\]`
- [ ] In math mode, variable-like tokens >3 letters use `\text{}`
- [ ] Citations are complete and accurate
- [ ] Cross-references use correct prefix: Fig.~\ref, Table~\ref, Section~\ref, Eq.~\eqref, Algorithm~\ref
- [ ] Conclusion is a single dense paragraph (no subsections)
- [ ] Figure source font ≥ 24pt (readable after LaTeX scaling)
- [ ] Each Python plot = 1 file → 1 figure (no subplots); composite via LaTeX \subfigure
- [ ] Experiment results subsections each end with \fbox Takeaway box
```

## When to Use

Use this skill in the following scenarios:

- **Pre-submission check** - Final review before submitting to a journal or conference
- **After first draft** - Systematic review after completing the first draft
- **Before advisor review** - Self-check before requesting advisor feedback to improve quality
- **Post-revision verification** - After revising based on reviewer comments, verify all issues are addressed
- **Collaborator review** - Quality check before sending to collaborators

## Review Process

Follow these steps for systematic paper review:

### Step 1: Structure Review
Start with the overall structure, checking if all sections are complete and logically coherent.

### Step 2: Content Review
Dive into each section, checking content accuracy and completeness.

### Step 3: Citation Check
Verify the completeness and accuracy of all citations.

### Step 4: Figure/Table Review
Check the quality and captions of all figures and tables.

### Step 5: Writing Quality
Review language expression and writing clarity.

### Step 6: Math Conformance Check
Verify equation environment and variable naming style consistency.

### Step 7: Final Checklist
Use the quality checklist for final verification.

## Best Practices

### Review Timing
- **Spaced review** - Wait 1-2 days after completing the draft before reviewing to maintain objectivity
- **Multiple rounds** - Conduct multiple review rounds, focusing on different aspects each time
- **Print review** - Print a hard copy for review; issues are easier to spot on paper

### Review Techniques
- **Reverse reading** - Read from conclusion backwards to check logical coherence
- **Read aloud** - Reading the paper aloud helps identify language issues
- **Reviewer perspective** - Assume you are a reviewer and read critically

### Common Issues
- Abstract too brief or too verbose
- Introduction lacks clear research question statement
- Method lacks sufficient detail for reproduction
- Results lack statistical significance tests
- Discussion doesn't address research limitations
- Figures/tables lack clear captions/labels, or contain in-figure title text
- Display equations use `$$...$$` or `\[...\]` instead of `equation`
- Long variable-like tokens are not wrapped with `\text{}`
- Inconsistent citation formatting

## Summary

The Paper Self-Review skill provides a systematic paper quality checking process, helping researchers identify and resolve issues before submission, improving paper quality and acceptance rates.
