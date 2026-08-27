#!/usr/bin/env bash
# policy/test-lint.sh — Automated tests for policy/lint.sh
#
# Tests: --constraint-type filter, --autofix filter, --fix mode,
#        fix_patterns correctness, post-fix verification, edge cases.
#
# Usage: bash policy/test-lint.sh
# Exit: 0 = all pass, 1 = failures

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINT="$SCRIPT_DIR/lint.sh"
TEST_DIR=$(mktemp -d)
PASS=0
FAIL=0

cleanup() { rm -rf "$TEST_DIR"; }
trap cleanup EXIT

# ─── Helpers ────────────────────────────────────────────────────────────

assert_exit() {
  local desc="$1" expected="$2"
  shift 2
  local actual
  "$@" >/dev/null 2>&1 && actual=0 || actual=$?
  if [[ "$actual" == "$expected" ]]; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (expected exit $expected, got $actual)"
  fi
}

assert_contains() {
  local desc="$1" pattern="$2"
  shift 2
  local output
  output=$("$@" 2>&1) || true
  if echo "$output" | grep -q "$pattern"; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (pattern '$pattern' not found in output)"
  fi
}

assert_file_contains() {
  local desc="$1" file="$2" pattern="$3"
  if grep -q "$pattern" "$file" 2>/dev/null; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (pattern '$pattern' not found in $file)"
  fi
}

assert_file_not_contains() {
  local desc="$1" file="$2" pattern="$3"
  if ! grep -q "$pattern" "$file" 2>/dev/null; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (pattern '$pattern' should NOT be in $file)"
  fi
}

reset_test_file() {
  cat > "$TEST_DIR/test.tex" << 'EOF'
In order to improve efficiency, we propose a new framework.
It is important to note that this serves as the foundation.
Gallery 825 stands as the main exhibition space.
The system plays a crucial role in maintaining stability.
This approach is sort of like the original but a lot of work.
The arrow → points right.
Experts argue that this is correct.
EOF
}

# ─── Test Suite ─────────────────────────────────────────────────────────

echo "=== 1. Filter Tests ==="

echo "1.1 --constraint-type guardrail"
reset_test_file
assert_contains "guardrail filter finds violations" "Warnings:" \
  bash "$LINT" --constraint-type guardrail --quiet "$TEST_DIR"

echo "1.2 --constraint-type guidance"
assert_exit "guidance filter on test file passes" 0 \
  bash "$LINT" --constraint-type guidance --quiet "$TEST_DIR"

echo "1.3 --autofix safe"
reset_test_file
assert_contains "autofix=safe filter runs" "Rules checked" \
  bash "$LINT" --autofix safe --quiet "$TEST_DIR"

echo "1.4 --autofix assisted"
assert_contains "autofix=assisted filter runs" "Rules checked" \
  bash "$LINT" --autofix assisted --quiet "$TEST_DIR"

echo "1.5 invalid --constraint-type"
assert_exit "invalid constraint-type rejects" 2 \
  bash "$LINT" --constraint-type invalid "$TEST_DIR"

echo "1.6 invalid --autofix"
assert_exit "invalid autofix rejects" 2 \
  bash "$LINT" --autofix invalid "$TEST_DIR"

echo ""
echo "=== 2. Fix Mode Tests ==="

echo "2.1 --fix replaces filler phrases"
reset_test_file
bash "$LINT" --fix --quiet "$TEST_DIR" >/dev/null 2>&1
assert_file_not_contains "In order to → to" "$TEST_DIR/test.tex" "In order to"
assert_file_contains "replaced with to" "$TEST_DIR/test.tex" "^to improve"

echo "2.2 --fix replaces copula dodges"
assert_file_not_contains "serves as removed" "$TEST_DIR/test.tex" "serves as"
assert_file_not_contains "stands as removed" "$TEST_DIR/test.tex" "stands as"
assert_file_contains "replaced with is" "$TEST_DIR/test.tex" " is the "

echo "2.3 --fix replaces informal vocabulary"
assert_file_not_contains "sort of removed" "$TEST_DIR/test.tex" "sort of"
# "a lot of" 是 flag-only（autofix 已撤销：产物 "many" 会在量词+名词组合中触发
# PROSE.VAGUE_QUANTIFIERS，修复循环）——只报不改
assert_file_contains "a lot of flagged but not auto-fixed" "$TEST_DIR/test.tex" "a lot of"

echo "2.4 --fix replaces unicode arrows"
assert_file_not_contains "unicode arrow removed" "$TEST_DIR/test.tex" "→"
assert_file_contains "latex arrow inserted" "$TEST_DIR/test.tex" '\\rightarrow'

echo "2.5 --fix replaces vague attributions"
assert_file_not_contains "vague attribution removed" "$TEST_DIR/test.tex" "Experts argue that"

echo "2.6 --fix skips non-safe rules"
# Use a fresh directory with ONLY an em-dash violation
emdash_dir=$(mktemp -d)
echo "This result is important --- very important indeed." > "$emdash_dir/test.tex"
bash "$LINT" --fix --quiet "$emdash_dir" >/dev/null 2>&1
assert_file_contains "em-dash preserved (assisted)" "$emdash_dir/test.tex" "\-\-\-"
rm -rf "$emdash_dir"

echo ""
echo "=== 3. Post-Fix Verification Tests ==="

echo "3.1 --fix reports VERIFIED when clean"
reset_test_file
assert_contains "verified clean output" "VERIFIED" \
  bash "$LINT" --fix "$TEST_DIR"

echo ""
echo "=== 4. Edge Case Tests ==="

echo "4.1 empty directory"
local_empty=$(mktemp -d)
assert_exit "empty directory passes" 0 \
  bash "$LINT" --quiet "$local_empty"
rm -rf "$local_empty"

echo "4.2 --fix on already clean file"
cat > "$TEST_DIR/clean.tex" << 'EOF'
We propose a method to improve accuracy by 15.3 percent.
EOF
# Remove test.tex to only have clean.tex
rm -f "$TEST_DIR/test.tex"
assert_exit "--fix on clean file passes" 0 \
  bash "$LINT" --fix --quiet "$TEST_DIR"

echo "4.3 --fix idempotency"
reset_test_file
bash "$LINT" --fix --quiet "$TEST_DIR" >/dev/null 2>&1
cp "$TEST_DIR/test.tex" "$TEST_DIR/after_first.tex"
bash "$LINT" --fix --quiet "$TEST_DIR" >/dev/null 2>&1
if diff -q "$TEST_DIR/test.tex" "$TEST_DIR/after_first.tex" >/dev/null 2>&1; then
  ((PASS++)); echo "  ✓ fix is idempotent"
else
  ((FAIL++)); echo "  ✗ fix is NOT idempotent (second run changed file)"
fi

echo "4.4 --constraint-type + --fix combination"
reset_test_file
bash "$LINT" --fix --constraint-type guardrail --quiet "$TEST_DIR" >/dev/null 2>&1
assert_file_not_contains "guardrail fix works with filter" "$TEST_DIR/test.tex" "serves as"

echo ""
echo "=== 5. Filter Count Tests ==="

echo "5.1 guardrail rule count"
# 期望值随 guardrail 规则增减而变化，新增规则时同步更新
EXPECTED_GUARDRAIL=30
count=$(bash "$LINT" --constraint-type guardrail --quiet "$TEST_DIR" 2>&1 | grep "Rules checked:" | grep -o '[0-9]*')
if [[ "$count" == "$EXPECTED_GUARDRAIL" ]]; then
  ((PASS++)); echo "  ✓ guardrail rules: $count"
else
  ((FAIL++)); echo "  ✗ guardrail rules: expected $EXPECTED_GUARDRAIL, got $count"
fi

echo "5.2 safe rule count"
count=$(bash "$LINT" --autofix safe --quiet "$TEST_DIR" 2>&1 | grep "Rules checked:" | grep -o '[0-9]*')
if [[ "$count" == "5" ]]; then
  ((PASS++)); echo "  ✓ safe rules: $count"
else
  ((FAIL++)); echo "  ✗ safe rules: expected 5, got $count"
fi

echo ""
echo "=== 6. PROSE.NO_INTERNAL_PROVENANCE Acceptance Tests ==="
# Fixtures are the real leaks from the ARGUS sweep (2026-08-15) and the real
# false positives it produced. See docs/req-prose-no-internal-provenance-v2.md §8.
PROV_DIR="$TEST_DIR/provenance"
mkdir -p "$PROV_DIR"

# lint.sh exits 1 on error-severity findings, and this harness runs under
# `set -eo pipefail` — without `|| true` the pipeline reads as a harness failure
# and every must-flag case would report "got none".
prov_lint() { bash "$LINT" --rule PROSE.NO_INTERNAL_PROVENANCE "$PROV_DIR" 2>&1 || true; }

assert_flags() {
  local desc="$1" content="$2"
  printf '%s\n' "$content" > "$PROV_DIR/case.tex"
  if prov_lint | grep -qE '(ERROR|WARN)'; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (expected a finding, got none)"
  fi
}

assert_clean() {
  local desc="$1" content="$2"
  printf '%s\n' "$content" > "$PROV_DIR/case.tex"
  local out; out=$(prov_lint)
  if echo "$out" | grep -qE '(ERROR|WARN)'; then
    ((FAIL++)); echo "  ✗ $desc (false positive)"
    echo "$out" | grep -E '(ERROR|WARN)' | sed 's/^/      /'
  else
    ((PASS++)); echo "  ✓ $desc"
  fi
}

echo "6.1 must flag"
assert_flags "P1/P2/P3 result path in body prose" \
  'Dispersion is reported in \path{experiments/results/rc1_pdisp/rc1_pdisp.csv}.'
assert_flags "same path inside a caption (scope: not body-only)" \
  '\caption{Spam curve, from \path{experiments/results/rc2_spam/rc2_spam_curve.csv}.}'
assert_flags "P4 schema identifiers as a table cell" \
  'Coverage & \texttt{empirical\_rate, empirical\_ucl} \\'
assert_flags "P5 retraction narrative" \
  'The old refined general bound is retracted.'
assert_flags "P5 collision-avoidance narrative" \
  'The variants are named V0--V3 to avoid collision with the theorem labels C1/C2.'

echo "6.2 must not flag (exclusions)"
assert_clean "LaTeX source plumbing" \
  '\includegraphics[width=\linewidth]{figures/fig_f3_stats.pdf}'
assert_clean "artifact URL" \
  'Code is at \url{https://anonymous.4open.science/r/Argus-B943}.'
assert_clean "model identifiers and seeds" \
  'The Llama-3.1-70B arm uses seed 42 with a grid over 0.1, 0.2, 0.5.'
assert_clean "domain terms that look like code" \
  'The L1 commit carries weighted mass rather than the unweighted count.'
assert_clean "EXP-mandated fabricated-results disclosure" \
  '\caption{[SIMULATED] Projected throughput; numbers are not from a real run.}'

echo "6.3 undefined-identifier extractor (stage 1)"
printf '%s\n' 'As Golden G4, at $\tilde r=0$ the branch is inactive. C6 holds throughout.' \
  'We evaluate the Llama-3.1-70B arm.' > "$PROV_DIR/case.tex"
extract_out=$(bash "$SCRIPT_DIR/scripts/extract-undefined-identifiers.sh" "$PROV_DIR" 2>&1)
if echo "$extract_out" | grep -q "C6"; then
  ((PASS++)); echo "  ✓ extracts undefined claim-register label C6"
else
  ((FAIL++)); echo "  ✗ extracts undefined claim-register label C6"
fi
if echo "$extract_out" | grep -qE '\bLlama'; then
  ((FAIL++)); echo "  ✗ model identifier Llama-3.1-70B must be excluded"
else
  ((PASS++)); echo "  ✓ model identifier excluded from candidates"
fi
rm -rf "$PROV_DIR"

echo ""
echo "=== 7. PROSE.INFORMAL_VOCABULARY Class-1 Acceptance Tests ==="
# Fixtures from the CISU/ct_unlearning sweep (NDSS, 2026-08-17). Only class 1 is
# regex-judged; classes 2-5 need an LLM and live in the rule card. The keep-cases
# matter as much as the flag-cases — over-execution is the documented failure mode.
INF_DIR="$TEST_DIR/informal"
mkdir -p "$INF_DIR"
inf_lint() { bash "$LINT" --rule PROSE.INFORMAL_VOCABULARY "$INF_DIR" 2>&1 || true; }

inf_flags() {
  local desc="$1" content="$2"
  printf '%s\n' "$content" > "$INF_DIR/case.tex"
  if inf_lint | grep -qE '(ERROR|WARN)'; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (expected a finding, got none)"
  fi
}
inf_clean() {
  local desc="$1" content="$2"
  printf '%s\n' "$content" > "$INF_DIR/case.tex"
  local out; out=$(inf_lint)
  if echo "$out" | grep -qE '(ERROR|WARN)'; then
    ((FAIL++)); echo "  ✗ $desc (false positive)"; echo "$out" | grep -E '(ERROR|WARN)' | sed 's/^/      /'
  else
    ((PASS++)); echo "  ✓ $desc"
  fi
}

echo "7.1 class-1 idiomatic adverbials must flag"
inf_flags "at all" 'The frozen-state baseline does not help at all.'
inf_flags "in the first place" 'The operator must decide in the first place which segments to quarantine.'
inf_flags "ahead of time" 'The schedule must be fixed ahead of time.'
inf_flags "up front" 'The cost is paid up front.'
inf_flags "at all (capitalised, sentence start)" 'At all, the estimator is unchanged.'

echo "7.2 must not flag (allowlist + legitimate collocations)"
inf_clean "from scratch is a term of art" 'We retrain the model from scratch as the exact-deletion reference.'
inf_clean "at all times / at all scales" 'The invariant holds at all times and at all scales.'
inf_clean "phrasal-verb allowlist" 'Theorem 3 rules out unilateral deviations; the bound follows from Lemma 2.'
inf_clean "falls back is a field term" 'The estimator falls back to the retrospective regime.'
inf_clean "cheap unlearning is field terminology" 'Cheap unlearning is the design goal for segment-level deletion.'

echo "7.3 class-1 autofix is one-to-one and safe"
printf '%s\n' 'The schedule is fixed ahead of time and the cost is paid up front.' > "$INF_DIR/case.tex"
bash "$LINT" --fix --quiet "$INF_DIR" >/dev/null 2>&1 || true
assert_file_contains "ahead of time → in advance" "$INF_DIR/case.tex" "in advance"
assert_file_not_contains "no 'ahead of time' left" "$INF_DIR/case.tex" "ahead of time"
rm -rf "$INF_DIR"

echo ""
echo "=== 8. PROSE.SELF_UNDERMINING Acceptance Tests ==="
# Only the closed-set lexis layer is regex-judged; responsibility-scope and the
# three-step disposition need an LLM and live in the rule card. The keep-cases are
# the point of the rule: a neutral, quantified, local statement of an unfavourable
# result is correct writing and must survive the lint.
SU_DIR="$TEST_DIR/self-undermining"
mkdir -p "$SU_DIR"
su_lint() { bash "$LINT" --rule PROSE.SELF_UNDERMINING "$SU_DIR" 2>&1 || true; }

su_flags() {
  local desc="$1" content="$2"
  printf '%s\n' "$content" > "$SU_DIR/case.tex"
  if su_lint | grep -qE '(ERROR|WARN)'; then
    ((PASS++)); echo "  ✓ $desc"
  else
    ((FAIL++)); echo "  ✗ $desc (expected a finding, got none)"
  fi
}
su_clean() {
  local desc="$1" content="$2"
  printf '%s\n' "$content" > "$SU_DIR/case.tex"
  local out; out=$(su_lint)
  if echo "$out" | grep -qE '(ERROR|WARN)'; then
    ((FAIL++)); echo "  ✗ $desc (false positive)"; echo "$out" | grep -E '(ERROR|WARN)' | sed 's/^/      /'
  else
    ((PASS++)); echo "  ✓ $desc"
  fi
}

echo "8.1 self-weakening lexis must flag"
su_flags "unfortunately + does not outperform" 'Unfortunately, our method does not outperform the strongest baseline.'
su_flags "regrettably + falls short + lags behind" 'Regrettably, recall falls short of the ceiling and lags behind prior work.'

echo "8.2 must not flag (neutral, anchored, or non-self-directed)"
su_clean "quantified local unfavourable result" 'Our recall is 3.1 points lower on ImageNet-LT than the strongest baseline (Table 4).'
su_clean "falls short of a theoretical limit" 'The bound falls short of the information-theoretic limit by a factor of two.'
su_clean "lags behind by N time steps" 'The estimator lags behind by two time steps under the causal constraint.'
rm -rf "$SU_DIR"

echo ""
echo "═══ Test Summary ═══"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Total:  $((PASS + FAIL))"

if (( FAIL > 0 )); then
  exit 1
else
  echo "  ALL TESTS PASSED"
  exit 0
fi
