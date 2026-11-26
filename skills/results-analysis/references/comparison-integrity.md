# Comparison Integrity

How an experimental comparison is judged, and what makes it fair. The fill-in
version of these rules lives in the research-workspace-layout record templates:
`analysis-plan.md` (decision rules), `design.md` (fair comparison) and `results.md`
(verdicts).

## How reasonable choices add up to selecting the result

Most selected results come from a series of small decisions, each defensible on
its own, that are all taken after looking at the numbers:

- **Adding seeds until the difference is significant.** Every extra seed is a fresh
  chance to cross the threshold, so stopping at the first crossing inflates the
  false-positive rate. Fix the seed count, or an extension rule, before the run.
- **Switching tests.** A paired t-test fails, so a Wilcoxon test is tried, then a
  one-sided version. Each switch is another draw. The test is named in the plan.
- **Selecting epochs or checkpoints on the test set.** Reporting the best test epoch
  reports the maximum of a noisy curve. Selection uses validation data only.
- **Leaving the baseline at default settings.** The new method gets a tuning sweep
  and the baseline gets its paper defaults, so the comparison measures tuning
  effort. Both arms get the same selection data and a comparable search budget.

The remedy is the same in every case: write the decision rule before the run, and
treat any rule chosen afterwards as exploratory.

## Three verdicts

A comparison of a new method against its comparator ends in exactly one of these:

- **Supported.** The paired-bootstrap 95% CI of the difference excludes zero on the
  favourable side, and every final seed agrees in direction, with at least N final
  seeds (N defaults to 3).
- **Refuted.** The CI lies entirely on the unfavourable side. **Refuted
  (equivalent)** means the CI lies entirely inside an equivalence margin ±δ that was
  written down before the run, so any effect is too small to matter. This verdict
  is checked first: a CI inside the margin is refuted (equivalent) even when it
  also excludes zero.
- **Inconclusive.** Every other outcome. A CI that crosses zero with a favourable
  mean is inconclusive, and the write-up says so.

Each claim gets one verdict per test set. Later re-analysis on the same test set is
exploratory. A separate "worth pursuing" threshold can decide whether to spend more
effort on an idea, but it never substitutes for a refutation.

## The resampling unit

Resample the unit that varies independently, which is usually the evaluation item.
When several seeds are scored on the same evaluation set, the seed × item scores
are not independent: every seed sees the same hard items. Pooling them as if they
were independent narrows the CI by roughly the square root of the seed count and
makes a noise-level difference look significant. Average each item's difference
across seeds first, then resample items.

When each seed has its own evaluation set, resample items within each seed and
combine (a stratified bootstrap). Resampling over seeds alone is reasonable only
with at least 5 seeds, since a bootstrap over 3 values has almost no resolution.

The item-level interval holds the seeds fixed. It describes the models that were
actually trained and says nothing about how a fresh training run would land. The
seed-agreement clause covers that: one seed pointing the other way blocks a
supported verdict however narrow the item-level interval is. A claim about the
training procedure in general, made with 5 or more seeds, also reports the interval
from resampling seeds.

```python
import numpy as np

def paired_bootstrap(new, base, n_boot=10000, seed=0):
    """new, base: arrays of shape (n_seeds, n_items) scored on the same items.
    Returns the mean difference and its 95% CI, resampling items."""
    diff = (np.asarray(new) - np.asarray(base)).mean(axis=0)  # one value per item
    rng = np.random.default_rng(seed)
    n = diff.size
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return diff.mean(), lo, hi

def seeds_in_favour(new, base, sign=+1):
    """Number of seeds whose mean difference has the favourable sign."""
    per_seed = (np.asarray(new) - np.asarray(base)).mean(axis=1)
    return int((np.sign(per_seed) == sign).sum())
```

Set `sign=-1` when lower is better. Where the randomness enters matters as much as
the count: a deterministic model evaluated on fixed data gives identical scores for
every seed, so its seed has to act on the data (split, subsample or sampling
temperature). Seeds used to select hyperparameters are excluded from the final
statistics. An experiment may declare fewer than 3 seeds in its plan, with the
reason, before it runs. The seed-agreement clause then does not apply, and every
number reported from it states its seed count.

## Fair comparison

- The baseline is the strongest reasonable one, tuned on the same selection data
  with a comparable number of configurations as the new method.
- A selected value on the edge of a search grid means the optimum may lie outside
  it. Extend the grid past that edge for every arm.
- Training or inference budgets match within a stated tolerance, in wall-clock time
  or steps.
- Without training, as in LLM evaluation, inference settings (temperature, token
  limits, reasoning effort) and the effort spent tuning prompts match across arms.
- Selection uses validation data, and the test set is used once for the final
  report. A validation set used to select a configuration cannot also report it.
- The predicted result is written down before the run, so the record shows whether
  the outcome was expected.

## What counts as new data

After an inconclusive verdict, a re-test needs data that took no part in any
selection or verdict for the same question. Re-folding a cross-validation split or
re-subsampling the same pool reuses that data. A re-test can also follow an
extension rule written in the plan beforehand, such as "extend to N seeds". The
rule fixes the CI level for each look, for example 97.5% at both of two looks,
since two looks at 95% raise the false-positive rate to about 8%.
Changing the test, the metric or the threshold on the same data leaves the verdict
where it was.

## Reviewing a verdict

A supported or refuted verdict gets an independent, read-only review before it is
used. The reviewer opens the plan, the run code, the raw result files and the
results file, and checks:

1. The run code matches the frozen plan on grid size, search budget, split,
   subsampling, seeds and metric.
2. The baseline was treated the same way as the new method.
3. The test data was untouched during selection, and no single validation set was
   used both to select and to report.
4. The numbers in the results file match the raw result files, and the wording is
   no stronger than the evidence.
5. The statistics follow the plan: resampling unit, seed count and direction
   agreement.
6. For a refutation, the method had a fair chance: tuning was adequate, the
   selected value sits away from a grid edge, and the implementation was checked.
