# Results — EXP-YYYY-NNN

**Status:** partial | complete
**Last updated:** YYYY-MM-DD

Report only what the promoted artifacts support, and name the artifact each number
comes from.

## Verdicts

One block per claim in `analysis-plan.md`.

- **Claim:**
- **Verdict:** supported, refuted, refuted (equivalent) or inconclusive, as `analysis-plan.md` defines them
- **Predicted vs actual:**
- **Attribution:** the component the change comes from, and the ablation or control that shows it
- **Anomalies examined:** for example a metric of exactly 0, and what explained it
- **Leads:** an unplanned or single-seed observation is recorded here as a lead (unverified) and supports no conclusion

## <family>

**Grid:** <cells × seeds × conditions>

**Outcome:**

Caveats belong in the same paragraph as the number they qualify, not appended after
it. In particular: a statistic that passes in every cell says nothing about the
test's sensitivity when no negative control was run, and any column not named in the
frozen analysis plan is exploratory regardless of what it shows.

## Not reported here

What was measured but has no promoted artifact, and what correction has not yet been
applied.

## Promoted artifacts

See `figure-manifest.yml`.
