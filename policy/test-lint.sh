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
assert_file_not_contains "a lot of removed" "$TEST_DIR/test.tex" "a lot of"

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
count=$(bash "$LINT" --constraint-type guardrail --quiet "$TEST_DIR" 2>&1 | grep "Rules checked:" | grep -o '[0-9]*')
if [[ "$count" == "18" ]]; then
  ((PASS++)); echo "  ✓ guardrail rules: $count"
else
  ((FAIL++)); echo "  ✗ guardrail rules: expected 18, got $count"
fi

echo "5.2 safe rule count"
count=$(bash "$LINT" --autofix safe --quiet "$TEST_DIR" 2>&1 | grep "Rules checked:" | grep -o '[0-9]*')
if [[ "$count" == "5" ]]; then
  ((PASS++)); echo "  ✓ safe rules: $count"
else
  ((FAIL++)); echo "  ✗ safe rules: expected 5, got $count"
fi

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
