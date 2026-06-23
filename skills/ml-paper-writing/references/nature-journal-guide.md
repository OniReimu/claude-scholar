# Nature-family Journal Writing Guide (delta)

Scope: **only what differs** when targeting Nature / Science / PNAS / Cell vs the conference
workflow (NeurIPS/ICML/ICLR/CCS) that `ml-paper-writing` already covers. Use this *alongside*
`ml-paper-writing`, not instead of it. All prose policy rules (anti-AI, em-dash, claim support)
apply venue-agnostically — this doc adds the journal-specific layer.

## 1. Manuscript structure (the biggest delta)

| Journal | Abstract | Body | Methods | Caps |
|---|---|---|---|---|
| **Nature** (Article/Letter) | ~150w, unstructured, no headings | narrative main text, often no Intro/Results/Discussion headings | **at the END**, detailed | ~3–5k words main, ~50 main refs, ≤~6 display items; Extended Data + Supplementary separate |
| **Science** (Research Article/Report) | one short paragraph | narrative | end | tight word/ref caps; Supplementary carries detail |
| **PNAS** | structured | Intro / Results / Discussion / Methods (conventional) | end or inline | **Significance statement ≤120w MANDATORY** |
| **Cell** | + graphical abstract | structured | **STAR Methods** | structured, highlights |

Key moves vs a conference paper: **Methods go to the end** (main text stays narrative and
broadly readable); **strict word/figure/reference caps** push detail into Methods / Extended Data
/ Supplementary; lead for a **general scientific audience**, not a subfield. `<!-- policy:SUBMIT.PAGE_LIMIT_STRICT -->`

## 2. Significance / impact framing

The first paragraph must say **why this matters broadly**, before any technical setup — editors
triage on general-audience significance. PNAS: write the Significance statement first; Nature:
the editorial summary. Avoid jargon in the opening; it can be technical later/in Methods.

## 3. Cover letter (mandatory — conferences have none)

One page to the editor: (1) the finding in 2–3 sentences; (2) why it's important *now*; (3) why
this journal/scope; (4) statements (not under consideration elsewhere, no conflicts); (5)
suggested + opposed reviewers. Keep claims defensible — the cover letter is not the place to
over-claim. `<!-- policy:CITE.CLAIM_SUPPORT_REQUIRED -->`

## 4. Data & Code Availability + FAIR (mandatory; ML/conference papers usually skip)

- **Data Availability statement** is required: where the data lives, accession numbers, and any
  access restrictions. Deposit in a repository that mints a DOI/accession (Zenodo, figshare, OSF,
  or domain repos like GEO/PDB) and cite the dataset with a DataCite-style citation.
- **Code Availability statement**: public repo + version/commit or archived DOI.
- **FAIR** (Findable/Accessible/Interoperable/Reusable): persistent identifier + license + metadata.
- Nature life-sciences also require a **Reporting Summary** / editorial-policy checklist.

## 5. English / phrasing

Nature house English favors broad-audience clarity over density. Use the Academic Phrasebank
(https://www.phrasebank.manchester.ac.uk/) for section moves. Still obey every prose rule in
`policy/rules/` (these are author-voice + anti-AI, not venue-specific) and `policy/style-guide.md`.
`<!-- style:author-voice -->`

## 6. Rebuttal cadence (journal ≠ conference) — see `review-response`

Journal revision is multi-round and addresses the **editor + each reviewer**: a standalone
"Response to Reviewers" document (point-by-point), plus a tracked-changes manuscript, over weeks.
This differs from a conference rebuttal's single short window and word cap. Use `review-response`
with this journal cadence in mind (acknowledge → quote comment → response → pointer to revised
text/line). `<!-- policy:CITE.CLAIM_SUPPORT_REQUIRED -->`

## When to use

Target venue ∈ {Nature, Science, PNAS, Cell, *Nature X*}. For NeurIPS/ICML/ICLR/CCS/S&P stay on
the base `ml-paper-writing` + the relevant `policy/profiles/` venue profile.
