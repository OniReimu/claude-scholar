# Policy Test Corpus

Annotated prose fixtures measuring **rule accuracy** — distinct from `policy/test-lint.sh`,
which tests lint *mechanics* (flags, fix emission, exit codes) and never asks whether a rule
fires on the right sentence.

Each case declares the rule IDs it expects. The runner (`policy/test-corpus.sh`) reports:

- **Recall miss** — an `@EXPECT`ed rule did not fire (rule is blind)
- **False positive** — a rule fired on a case that did not expect it (rule is noisy)

Precision cases (`@EXPECT none`) carry the weight here. Over-firing is the documented
failure mode of this engine — a rule that flags every threat-model sentence gets
switched off by the author, which costs more than the rule ever bought.

## Case format

```
% @CASE <kebab-case-label>
% @EXPECT RULE.ID[, RULE.ID...]   |   none
<prose lines>
% @ENDCASE
```

A case that fires a rule it should not is not automatically a rule bug — read the
sentence first. Some fixtures are deliberately near the boundary, and the correct
resolution is sometimes to narrow the fixture rather than the pattern. Record that
judgment in `@EXPECT` so it is checked from then on.

## Known-limitation cases

`@EXPECT` may name a rule the engine currently cannot catch only if the case is also
marked `@XFAIL <reason>`. Those are reported separately and do not fail the suite; they
exist so the gap stays visible and so a later fix flips them to passing loudly.
