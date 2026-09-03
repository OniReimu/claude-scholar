---
name: peer-review
description: Use this skill when the user is acting as an assigned reviewer or authorized co-reviewer and asks to review a confidential manuscript, prepare comments to authors or editors, assess a submission, or recommend accept, revision, or rejection. Routes reviewer-side work to the separately installed more-than-peer-review skill. Do not use for author-side self-review, literature review, code review, or responding to reviewer comments.
---

# Peer Review Handoff

This is a routing skill. The canonical reviewer-side workflow is maintained in the
separate [More Than Peer Review Skill](https://github.com/DELONG-L/More-Than-Peer-Review-Skill).
Do not duplicate or partially reimplement that workflow inside Claude Scholar.

## Route the request correctly

Use this handoff when the user is reviewing work for a journal, conference, workshop,
editor, program committee, or other authorized evaluation process. Typical requests
include confidential manuscript assessment, comments to authors, confidential editor
comments, recommendation calibration, or reviewer-form drafting.

Route nearby tasks elsewhere.

- Use `paper-self-review` when an author is checking their own draft before submission.
- Use `review-response` or `nature-response` when an author is responding to received
  reviewer comments.
- Use the literature workflow when the user wants a literature review rather than a
  publication decision on one submitted manuscript.
- Use the code-review workflow for source-code changes.

When the user's role is unclear and the distinction changes manuscript handling or the
requested deliverable, ask whether they are reviewing the submission for a venue or
checking their own paper.

## Handoff contract

Before opening, extracting, rendering, or substantively reading the manuscript,
locate the active skill named `more-than-peer-review` and invoke
`$more-than-peer-review` for the complete task. Once invoked, its current `SKILL.md`,
references, intake gate, review method, output rules, and validation requirements are
authoritative for the review workflow.

Do not read the manuscript first and hand over a summary. Do not transfer conclusions,
candidate criticisms, review text, or recommendation guesses from Claude Scholar into
the external skill. Give it the user's original request and the named manuscript so it
can begin from the original material.

Do not automatically combine `paper-self-review`, `writing-anti-ai`, generic literature
review, or rebuttal guidance with the reviewer-side workflow. The external skill owns
the substantive review and its final prose pass. Use another skill only when the user
separately requests a distinct deliverable after the review is complete.

## Missing external skill

If `more-than-peer-review` is not available in the active Codex skill registry or under
the installed skill directories, stop before reading the manuscript. Tell the user
that reviewer-side peer review is an external dependency and provide this repository
link.

https://github.com/DELONG-L/More-Than-Peer-Review-Skill

For a standard Codex installation, a local clone can be linked as follows.

```bash
git clone https://github.com/DELONG-L/More-Than-Peer-Review-Skill.git
mkdir -p ~/.codex/skills
ln -s "$(pwd)/More-Than-Peer-Review-Skill/more-than-peer-review" \
  ~/.codex/skills/more-than-peer-review
```

After installation, ask the user to open a fresh Codex session if the current session
does not discover the new skill. Do not silently fall back to Claude Scholar's
self-review checklist for a reviewer assignment.

## Ownership boundary

Claude Scholar owns only request classification and the handoff. More Than Peer Review
owns manuscript intake, substantive reviewer reasoning, recommendation mapping,
comments to authors and editors, and final validation. Keep this boundary explicit so
improvements to the external review workflow remain available without synchronizing
two copies.
