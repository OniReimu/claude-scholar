<!-- Thanks for contributing to claude-scholar. Keep changes surgical and proportionate. -->

## What & why
<!-- One or two sentences: what this changes and the gap it fills. -->

## Type
- [ ] Skill
- [ ] Command / Agent / Hook
- [ ] Policy rule / profile / lint
- [ ] Orchestrator
- [ ] Docs / infra
- [ ] Other

## Checklist
- [ ] `bash policy/validate.sh` — the four structural sections PASS (Profile / Integration markers / Orphan detection / Registry), no **new** FAILs vs baseline
- [ ] New policy rule (if any) is registered in `policy/README.md`, added to the relevant profile `## Includes`, and referenced by a `<!-- policy:RULE_ID -->` marker in an entry skill (no orphan)
- [ ] New skill (if any) has third-person `description` with trigger conditions; does not duplicate an existing skill
- [ ] `.claude-plugin/plugin.json` and `marketplace.json` versions match (bump if this is a release)
- [ ] Borrowed/adapted content credits its source and respects its license (MIT/Apache = attribute; NC/custom = method-only, no copied text)

## Notes for reviewers
<!-- Anything staged, deferred, or intentionally out of scope. -->
