---
name: paper-writer
description: This skill should be used when the user asks to "write a paper", "improve my academic writing", "learn from papers", or asks about paper structure, writing techniques, or submission requirements for conferences/journals. Provides extracted knowledge from research papers across structure, writing techniques, submission guides, and review response strategies.
version: 0.1.0
---

# Paper Writer

Extract and apply knowledge from research papers to improve academic writing. This skill provides access to patterns and techniques from published papers across top venues (NeurIPS, ICML, ICLR, Nature, Science, Cell, PNAS).

## Overview

Good academic writing follows specific patterns and conventions. This skill captures those patterns from real papers and makes them accessible for your writing.

## When to Use

Use this skill when:
- Writing a research paper and need structure guidance
- Looking for effective transition phrases
- Preparing a submission for a specific venue
- Writing a rebuttal to reviewer comments
- Studying papers to improve writing

## Knowledge Categories

| Category | Focus | Reference File |
|----------|-------|---------------|
| **Structure** | IMRaD organization, section patterns | `references/knowledge/structure.md` |
| **Writing Techniques** | Sentence patterns, transitions, clarity | `references/knowledge/writing-techniques.md` |
| **Submission Guides** | Venue-specific requirements (NeurIPS, Nature, etc.) | `references/knowledge/submission-guides.md` |
| **Review Response** | Rebuttal strategies, addressing comments | `references/knowledge/review-response.md` |

## Quick Reference

**To learn from a paper:**
1. Provide the paper file or link
2. Ensure it's indexed by document-mcp (in watched folder)
3. The paper-miner agent will extract and categorize knowledge

**To browse existing knowledge:**
- Read the relevant category file in `references/knowledge/`
- Each file contains: patterns, examples, and techniques

## Self-Evolving

This skill automatically updates its knowledge base when the paper-miner agent processes new papers.

## Additional Resources

### Knowledge Files
- **`references/knowledge/structure.md`** - Paper organization patterns (IMRaD: Introduction, Methods, Results, Discussion)
- **`references/knowledge/writing-techniques.md`** - Writing techniques and phrases (transitions, sentence structures, clarity)
- **`references/knowledge/submission-guides.md`** - Venue requirements (NeurIPS, ICML, ICLR, KDD, Nature, Science, Cell, PNAS)
- **`references/knowledge/review-response.md`** - Rebuttal strategies (technical questions, writing issues, additional experiments)

These knowledge files are automatically updated by the paper-miner agent when processing research papers.
