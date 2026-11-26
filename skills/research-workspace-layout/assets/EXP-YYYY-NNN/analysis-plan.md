# Analysis Plan — EXP-YYYY-NNN

**Status:** draft            <!-- draft | frozen -->
**Pre-registered:** YYYY-MM-DD   <!-- before the run, or say when it was written -->

<!-- Set Status to frozen and commit this file before any run larger than a smoke
test. Later changes go in deviations.md, and this file stays as committed. -->

## Statistic family

| Family | Count | Statistics |
|---|---|---|

## Decision rules

<!-- One block per claim. Copy the block for each further claim and replace every
angle-bracket blank. A block with a blank left in it counts as unfilled. -->

### Claim 1: <the claim in one sentence>

- **Primary metric and direction:** <metric>, <higher or lower> is better.
- **Comparator:** <baseline>
  Use the strongest reasonable baseline, tuned as `design.md` describes.
- **Supported:** the paired-bootstrap 95% CI of the difference excludes zero on the
  favourable side, and every final seed agrees in direction, with at least
  <N, default 3> final seeds.
- **Refuted:** the CI lies entirely on the unfavourable side.
- **Refuted (equivalent):** the CI lies entirely inside ±<δ in metric units>.
  The margin is fixed here, before the run. Check this verdict first. A CI inside
  the margin is refuted (equivalent) even when it also excludes zero.
- **Inconclusive:** every other outcome.
  This includes a CI that crosses zero with a favourable mean.
- **Resampling unit:** <evaluation item>
  When seeds share one evaluation set, average each item's difference across seeds,
  then resample items. Pooling seeds and items as if independent narrows the CI by
  about the square root of the seed count. When each seed has its own evaluation
  set, resample items within each seed. Resampling over seeds alone needs at least
  5 seeds. The item-level CI describes the models that were trained. The
  seed-agreement clause is what covers training randomness, and a claim about the
  training procedure in general, with 5 or more seeds, also reports the CI from
  resampling seeds.
- **Seeds:** <count>, with randomness entering through <source of randomness>.
  Sources are initialisation, data split, subsampling or sampling temperature.
  A deterministic model needs the seed to act on the data. Seeds used to select
  hyperparameters stay out of the final statistics. Fewer than 3 seeds is allowed
  when the count and its reason are written here before the run. The
  seed-agreement clause then does not apply, and every number reported from this
  experiment states its seed count.
- **After an inconclusive verdict:** <extension rule, or "none">
  An extension rule names the added seeds or items and the CI level used at each
  look, for example "extend to 5 seeds, 97.5% CI at both looks", since two looks at
  95% raise the false-positive rate. Any other re-test uses data that took no part
  in any selection or verdict for this question. Re-folding or re-subsampling the
  same pool counts as the same data.
  Switching the test, the metric or the threshold on the same data leaves the
  verdict unchanged.
- **Worth pursuing:** <threshold in metric units>
  This decides whether more effort goes into the idea. It never stands in for a
  refutation.

Each claim gets one verdict per test set. Any later analysis on that test set is
exploratory.

## Multiplicity

The correction, and what it is taken over. Decide before compute whether ineligible
cells are screened out, so the correction is not diluted by cells that could never
have produced a signal.

## Confirmatory vs exploratory

Name which is which here. Anything added later is exploratory regardless of what it
shows.
