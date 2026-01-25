---
name: scientific-writing
description: "This skill should be used when the user asks to \"write a paper\", \"polish my manuscript\", \"review this scientific text\", \"improve my academic writing\", \"help with submission\", \"remove AI writing patterns\", \"humanize this text\", or mentions scientific writing tasks. Covers research papers, conference submissions (NeurIPS, ICML, ICLR, KDD), and journal submissions (Nature, Science, Cell). Provides IMRAD structure, citation styles (APA/IEEE/Vancouver), figures/tables, anti-AI writing guidance, and reviewer-perspective polishing. Integrates with humanizer, humanizer-zh, and stop-slop skills for comprehensive AI-pattern removal. Includes real academic writing patterns from 378 published papers."
version: 0.6.0
---

# Scientific Writing

## Overview

Scientific writing is a process for communicating research with precision and clarity. Write manuscripts using IMRAD structure, citations (APA/AMA/Vancouver), figures/tables, and reporting guidelines (CONSORT/STROBE/PRISMA). Apply this skill for research papers and journal submissions.

**Critical Principle: Always write in full paragraphs with flowing prose. Never submit bullet points in the final manuscript.** Use a two-stage process: first create section outlines with key points, then convert those outlines into complete paragraphs.

## When to Use This Skill

This skill should be used when:
- Writing or revising any section of a scientific manuscript
- Structuring a research paper using IMRAD or CS conference formats
- Preparing submissions for top CS conferences (NeurIPS, ICML, ICLR, KDD, CVPR, AAAI)
- Writing for high-impact journals (Nature, Science, Cell, PNAS)
- Formatting citations and references
- Creating or improving figures, tables, and data visualizations
- Applying study-specific reporting guidelines
- Drafting abstracts that meet journal/conference requirements
- Polishing manuscripts from reviewer perspective
- Improving writing clarity, conciseness, and precision
- Addressing reviewer comments and revising manuscripts

## Visual Enhancement Requirement

**Every scientific paper MUST include at least 1-2 AI-generated figures.**

Use the **scientific-schematics** skill to generate publication-quality diagrams. For comprehensive guidance, refer to the scientific-schematics skill documentation.

**When to add schematics:**
- Study design and methodology flowcharts
- Conceptual framework diagrams
- Experimental workflow illustrations
- Data analysis pipeline diagrams
- Any complex concept that benefits from visualization

## Core Capabilities

### 1. Manuscript Structure and Organization

**IMRAD Format**: Introduction, Methods, Results, And Discussion structure.

For detailed guidance, refer to `references/imrad_structure.md`.

### 2. Citation and Reference Management

**Major Citation Styles:**
- **Vancouver**: Numbered citations, biomedical standard
- **APA**: Author-date citations, social sciences
- **IEEE**: Numbered square brackets, engineering and computer science

For comprehensive style guides, refer to `references/citation_styles.md`.

### 3. Figures and Tables

**Design Principles:**
- Make each table/figure self-explanatory with complete captions
- Use consistent formatting and terminology
- Label all axes, columns, and rows with units
- Follow the "one table/figure per 1000 words" guideline

For detailed best practices, refer to `references/figures_tables.md`.

### 4. Reporting Guidelines

**Key Guidelines:**
- **CONSORT**: Randomized controlled trials
- **STROBE**: Observational studies
- **PRISMA**: Systematic reviews and meta-analyses

For comprehensive guideline details, refer to `references/reporting_guidelines.md`.

### 5. Writing Principles

**Natural Academic Prose**: Avoid AI-like writing patterns.

For comprehensive guidance on producing natural, human-like text, refer to `references/anti_ai_writing.md`.

**Key principles:**
- Vary sentence length and structure naturally
- Use simple, common transitions
- Avoid overuse of "furthermore," "moreover," "additionally"
- Maintain author voice and natural variation

### 6. Real Academic Writing Patterns

**Based on analysis of 378 published ArXiv papers** (top CS venues: NeurIPS, ICML, ICLR, KDD)

#### Title Structure Patterns (by frequency)

**Pattern A: NAME: Description (64%)** - Most recommended
```
MethodName: Detailed Technical Description

Examples:
• CORE: Contrastive Masked Feature Reconstruction on Graphs
• ECHO: Toward Contextual Seq2Seq Paradigms in Large EEG Models
```

**Pattern B: X via Y (10%)** - Problem-Method correspondence
```
Goal/Problem via Method

Examples:
• Zero-shot Graph Anomaly Detection via Invariant Learning
• Scaling VLA Training via Reinforcement Learning
```

**Pattern C: Verbing Noun for Purpose (7%)** - Action-oriented
```
Verb-ing Noun for Purpose

Examples:
• Enhancing RAG with Recursive Evaluation for Multi-Hop QA
• Improving LLM Efficiency through Trajectory Reduction
```

#### Key Connectors

| Connector | Frequency | Purpose | Example |
|-----------|-----------|---------|---------|
| **for** | 43% | Purpose/application | "for Time Series Forecasting" |
| **via** | 10% | Method/approach | "via Reinforcement Learning" |
| **with** | 20% | Accompanying feature | "with Adaptive Mechanisms" |
| **and** | 18% | Parallel concepts | "Fast and Accurate" |

**Avoid**:
- ❌ "through" (3%) - too formal, "via" is preferred
- ❌ "furthermore", "moreover", "additionally" - AI generation markers

#### Technical Verbs (-ing forms)

Most frequently used technical verbs:
```
Learning     (62) - learning processes
Forecasting  (58) - prediction
Reasoning    (39) - reasoning
Decoding     (15) - decoding
Training     (11) - training
Modeling     (10) - modeling
Scaling       (9) - scaling
Enhancing     (8) - enhancement
```

**Usage principles**:
- ✅ Use -ing for action-oriented: "Learning", "Forecasting"
- ❌ Avoid static descriptions: "This is a learning approach" → "Learning Approach"

#### Technical Adjectives

Most frequently used technical adjectives:
```
Efficient    (15) - efficient
Multi-Agent  (12) - multi-agent
Dynamic      (11) - dynamic
General      (11) - general
Unified      (11) - unified
Adaptive     (10) - adaptive
Semantic      (9) - semantic
Scalable      (7) - scalable
Zero-shot     (6) - zero-shot
```

**Usage principles**:
- ✅ Technical adjectives: "Efficient", "Scalable", "Unified"
- ❌ Subjective adjectives: "important", "significant", "crucial", "pivotal"

#### Title Length

- **Average**: 9.1 words
- **Median**: 9 words
- **Recommended range**: 7-11 words

#### Specialized Patterns

**Framework Pattern** (21 papers):
```
MethodName: A [Adjective] Framework for [Application]

Examples:
• MiniOneRec: An Open-Source Framework for Scaling Recommendation
• Wave2Word: A Multimodal Framework for EEG-Text Alignment
```

**Approach Pattern** (3 papers):
```
MethodName: A [Feature] Approach to [Problem]

Examples:
• MFRS: A Multi-Frequency Approach to Scalable Forecasting
```

### 7. Field-Specific Language

Adapt language, terminology, and conventions to match the specific scientific discipline.

For detailed field-specific guidance, refer to `references/field_specific_language.md`.

## Writing Process: Two-Stage Approach

**CRITICAL: Always write in full paragraphs, never submit bullet points in scientific papers.**

**Stage 1: Create Section Outlines with Key Points**

1. Use the research-lookup skill to gather relevant literature
2. Create a structured outline with bullet points marking main arguments, key studies, data points
3. These bullet points serve as scaffolding—they are NOT the final manuscript

**Stage 2: Convert Key Points to Full Paragraphs**

1. Transform bullet points into complete sentences
2. Add transitions between sentences and ideas
3. Integrate citations naturally within sentences
4. Expand with context and explanation
5. Ensure logical flow from one sentence to the next

**Common Mistakes to Avoid:**
- Never leave bullet points in the final manuscript
- Never submit lists where paragraphs should be
- Don't use numbered or bulleted lists in Results or Discussion sections (except for specific cases like inclusion criteria)
- Do ensure every section flows as connected prose

## Targeted Polishing Workflow

**IMPORTANT: Before polishing or writing, always ask about the target venue**

1. **Ask for the target venue first**:
   - "Which journal or conference are you targeting?"
   - "What is the submission deadline?"

2. **Load the relevant reference** based on their response:
   - CS conferences → Load `references/cs_conferences.md`
   - High-impact journals → Load `references/nature_submissions.md`
   - Field-specific guidelines → Load appropriate reference

3. **Apply venue-specific requirements**:
   - Page/word limits
   - Format requirements
   - Structure requirements
   - Submission-specific content

4. **Use anti-AI writing guidelines**:
   - Always apply principles from `references/anti_ai_writing.md`
   - Vary sentence structure and length
   - Maintain natural academic prose

5. **Check logical flow**:
   - Load `references/logic_analysis.md` for detailed guidance
   - Analyze paragraph-level logical connections
   - Identify logical gaps and inconsistencies
   - Check transitions between paragraphs

## Submission-Specific Guidance

### Computer Science Conferences

For detailed guidance on CS conference submissions, refer to `references/cs_conferences.md`.

**Top CS Conferences:**
- **NeurIPS**: ML, AI, Neuroscience | 8 pages + refs
- **ICML**: ML theory and applications | 8 pages + refs
- **ICLR**: Deep learning, representations | 8 pages + refs
- **KDD**: Data mining | 10 pages + refs
- **CVPR**: Computer vision | 8 pages + refs
- **AAAI**: General AI | 7-8 pages + refs

**CS-Specific Requirements:**
- Double-blind anonymity
- LaTeX templates required
- Algorithm pseudocode for complex methods
- Ablation studies demonstrating component contributions
- Comparison with state-of-the-art baselines

### High-Impact Journals

For detailed guidance on Nature and other high-impact journals, refer to `references/nature_submissions.md`.

**Top Journals:**
- **Nature**: Multidisciplinary | 5-6 pages | ~8% acceptance
- **Science**: Multidisciplinary | 4500 words | ~7% acceptance
- **Cell**: Biology | Variable | ~10% acceptance
- **PNAS**: Multidisciplinary | 6 pages | ~20% acceptance

**High-Impact Journal Requirements:**
- Significance statement explaining broad impact
- Introduction accessible to non-specialists
- Methods in supplementary material allowed
- Clear statements of novelty and conceptual advance

### Reviewer-Perspective Polishing

For comprehensive guidance on polishing from a reviewer's perspective, refer to `references/reviewer_perspective_polishing.md`.

**Pre-Submission Checklist:**

**Content:**
- [ ] Novel contribution clearly stated
- [ ] Related work comprehensive
- [ ] Methods reproducible
- [ ] Results support all claims
- [ ] Limitations acknowledged

**Writing:**
- [ ] No grammar or spelling errors
- [ ] Consistent terminology throughout
- [ ] Clear, logical structure
- [ ] Transitions between sections

## Common Pitfalls to Avoid

**Top Rejection Reasons:**
1. Inappropriate or insufficiently described statistics
2. Over-interpretation of results or unsupported conclusions
3. Poorly described methods affecting reproducibility
4. Small, biased, or inappropriate samples
5. Poor writing quality or difficult-to-follow text
6. Inadequate literature review or context
7. Figures and tables that are unclear or poorly designed

**Writing Quality Issues:**
- Mixing tenses inappropriately (use past tense for methods/results, present for established facts)
- Excessive jargon or undefined acronyms
- Paragraph breaks that disrupt logical flow
- Missing transitions between sections

## References

**General Writing:**
- `references/imrad_structure.md`: Detailed guide to IMRAD format
- `references/citation_styles.md`: Complete citation style guides
- `references/figures_tables.md`: Best practices for data visualizations
- `references/reporting_guidelines.md`: Study-specific reporting standards
- `references/writing_principles.md`: Core principles of scientific communication

**Natural Writing:**
- `references/anti_ai_writing.md`: Guidelines to avoid AI-like patterns

**AI Pattern Removal for Academic Writing:**

⚠️ **IMPORTANT**: Academic writing requires objectivity. Do NOT use skills that "inject personality" or "add soul."

**Recommended workflow for papers:**
1. **stop-slop** (English) - Remove mechanical patterns while maintaining objectivity
2. **humanizer-zh (pattern detection only)** - Use the 24-pattern checklist, skip "soul injection" section
3. **scientific-writing** - Ensure compliance with academic standards

**What to remove from academic text:**
- Filler phrases and throat-clearing openers
- Formulaic structures (rule of three, binary contrasts)
- Mechanical sentence patterns (all same length)
- Overused AI vocabulary ("furthermore", "moreover", "crucial", "pivotal")
- Excessive hedging ("potentially possibly", "could be argued that")

**What to maintain in academic text:**
- Objective third-person voice (no "I think", "I believe")
- Professional, formal tone
- Clear, precise technical language
- Proper academic conventions and terminology

**Usage:** When polishing manuscripts:
1. Apply stop-slop for English or humanizer-zh patterns for Chinese
2. Focus on removing mechanical/AI patterns only
3. Do NOT inject personality, opinions, or "soul"
4. Maintain academic rigor and objectivity throughout
5. Use scientific-writing principles for structure and flow

**Logic Analysis:**
- `references/logic_analysis.md`: Paragraph and section-level logic checking

**Submission Guidance:**
- `references/cs_conferences.md`: Top CS conference submission guide
- `references/nature_submissions.md`: High-impact journal submission guidelines
- `references/field_specific_language.md`: Field-specific terminology and conventions

**Polishing:**
- `references/reviewer_perspective_polishing.md`: Comprehensive polishing guide

Load these references as needed when working on specific aspects of scientific writing.
