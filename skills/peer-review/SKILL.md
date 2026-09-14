---
name: peer-review
description: This skill should be used when the user is acting as a reviewer and asks to "review this submission", "write my review for NeurIPS/S&P", "draft comments to the authors", "assess this manuscript", "fill in the reviewer form", or "should I recommend accept or reject". Routes reviewer-side work to the separately installed more-than-peer-review skill, which owns the entire review workflow. Do not use for author-side self-review (paper-self-review), responding to received reviewer comments (review-response), literature review (knows-literature), or code review (code-review-excellence).
version: 0.1.0
tags: [Research, Academic, Review, Routing]
---

# Peer Review Handoff

This is a **routing skill**. It classifies the request and hands off. The reviewer-side
workflow itself lives in [More Than Peer Review](https://github.com/DELONG-L/More-Than-Peer-Review-Skill),
installed separately.

**Do not duplicate or partially reimplement that workflow here.** Claude Scholar's own
review skills are all author-side; none of them is a substitute for a reviewer
assignment, and reaching for the nearest one produces a submission checklist where a
review was asked for.

## Route the request correctly

Use this handoff when the user is evaluating someone else's work **for a venue** —
journal, conference, workshop, editor, program committee, or any other evaluation
process they have been asked to serve in. Typical requests: manuscript assessment,
comments to authors, confidential comments to the editor or AC, recommendation
calibration, reviewer-form drafting, meta-review.

Everything nearby is author-side and stays here:

| The user is… | Skill |
|---|---|
| checking their **own** draft before submission | `paper-self-review` |
| restructuring their own draft (paragraph necessity, claim spine) | `claim-architecture-review` |
| responding to reviews they **received** | `review-response` |
| surveying a literature, not deciding one submission | `knows-literature` |
| reviewing source code | `code-review-excellence` |
| auditing a Circom / ZK circuit | `circom-auditor` |

When the role is genuinely ambiguous **and** the answer changes the deliverable, ask
one question: *reviewing this for a venue, or checking your own paper?* Do not guess —
the two produce different documents from the same PDF.

## Handoff contract

**Before opening, extracting, rendering, or substantively reading the manuscript**,
locate the installed skill named `more-than-peer-review` and invoke it for the whole
task:

| Harness | Invocation |
|---|---|
| Claude Code | `/more-than-peer-review` |
| Codex | `$more-than-peer-review` |

Once invoked, **its current `SKILL.md` and references are authoritative** for the
review workflow. That sentence is deliberately not a list of components: enumerating
another skill's internals here creates a second copy that drifts, which is the exact
failure this handoff exists to avoid.

Three things not to do:

- **Do not read the manuscript first and hand over a summary.** Give the external skill
  the user's original request and the named file, so it starts from the source.
- **Do not carry conclusions across.** No candidate criticisms, no draft review text, no
  recommendation guesses formed on this side.
- **Do not bundle Claude Scholar skills into the review.** `paper-self-review`,
  `writing-anti-ai`, literature survey and rebuttal guidance are not part of a
  reviewer assignment. Use one only if the user separately asks for a distinct
  deliverable after the review is finished.

## When the external skill is not installed

**Stop before reading the manuscript.** Say that reviewer-side peer review is an
external dependency, and give the install below. Do not silently fall back to
`paper-self-review` — an author-side checklist answers a question nobody asked, and it
answers it confidently.

One clone, symlinked into both harnesses, so there is a single copy to update:

```bash
# 1. Clone once, anywhere you keep it
git clone https://github.com/DELONG-L/More-Than-Peer-Review-Skill.git
SKILL_SRC="$(pwd)/More-Than-Peer-Review-Skill/more-than-peer-review"

# 2. Link it into both personal skill directories
mkdir -p ~/.claude/skills ~/.codex/skills
ln -s "$SKILL_SRC" ~/.claude/skills/more-than-peer-review
ln -s "$SKILL_SRC" ~/.codex/skills/more-than-peer-review
```

Claude Code supports symlinks in the personal skills directory
([docs](https://code.claude.com/docs/en/skills#where-skills-live)); Codex discovers
`~/.codex/skills/` natively.

Updating is then one command against that clone:

```bash
git -C /path/to/More-Than-Peer-Review-Skill pull --ff-only origin main
```

Creating the top-level `~/.claude/skills/` directory for the first time needs a Claude
Code restart before `/more-than-peer-review` resolves. Codex picks it up on the next
task.

## Ownership boundary

Claude Scholar owns **request classification and the handoff, nothing else**. More Than
Peer Review owns manuscript intake, substantive reviewer reasoning, recommendation
mapping, comments to authors and editors, and its own final pass over the review prose.

Keeping that boundary sharp is what makes upstream improvements free: a change to the
review method, severity scale, or venue calibration lands in the external skill and
reaches every session without anything being synchronised here. The moment this file
starts describing *how* to review, that stops being true.
