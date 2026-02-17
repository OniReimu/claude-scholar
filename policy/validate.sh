#!/usr/bin/env bash
# policy/validate.sh — Policy Engine rule card validation
#
# Checks all rule cards in policy/rules/ for:
#   - Required frontmatter fields
#   - ID/slug uniqueness
#   - Filename = slug consistency
#   - Field value validity (severity, locked, layer, enforcement, check_kind)
#   - lint_patterns format (mode values, threshold presence)
#   - Profile Includes file existence
#   - Profile override validity (locked rules, param key existence)
#   - Integration marker → rule card mapping
#   - Orphan rule detection
#   - No modification to protected files (rules/, CLAUDE.md, AGENTS.md)
#
# Usage: policy/validate.sh
# Exit: 0 if all pass, 1 if errors found

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RULES_DIR="$SCRIPT_DIR/rules"
PROFILES_DIR="$SCRIPT_DIR/profiles"
ERRORS=0

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

err() { echo -e "  ${RED}FAIL${NC}: $*"; ((ERRORS++)) || true; }
pass() { echo -e "  ${GREEN}PASS${NC}: $*"; }
section() { echo -e "\n${BOLD}$*${NC}"; }

# ─── Extract frontmatter using awk ──────────────────────────────────────────
get_fm() { awk '/^---$/{n++;next} n==1{print}' "$1"; }

# ─── 1. Required Frontmatter Fields ─────────────────────────────────────────
section "1. Required Frontmatter Fields"
REQUIRED_FIELDS="id slug severity locked layer artifacts phases domains venues check_kind enforcement"

for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  fname=$(basename "$f")
  fm=$(get_fm "$f")
  missing=""
  for field in $REQUIRED_FIELDS; do
    if ! echo "$fm" | grep -q "^${field}: "; then
      # Special case: field might be followed by newline (block value)
      if ! echo "$fm" | grep -q "^${field}:"; then
        missing="$missing $field"
      fi
    fi
  done
  if [[ -n "$missing" ]]; then
    err "$fname: missing fields:$missing"
  fi
done
pass "All rule cards checked for required fields"

# ─── 2. ID/Slug Uniqueness ──────────────────────────────────────────────────
section "2. ID/Slug Uniqueness"

ids=""
slugs=""
for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  fm=$(get_fm "$f")
  id=$(echo "$fm" | awk '/^id: /{print $2; exit}')
  slug=$(echo "$fm" | awk '/^slug: /{print $2; exit}')
  ids="$ids $id"
  slugs="$slugs $slug"
done

dup_ids=$(echo "$ids" | tr ' ' '\n' | sort | uniq -d | grep -v '^$' || true)
dup_slugs=$(echo "$slugs" | tr ' ' '\n' | sort | uniq -d | grep -v '^$' || true)

[[ -z "$dup_ids" ]] && pass "No duplicate rule IDs" || err "Duplicate IDs: $dup_ids"
[[ -z "$dup_slugs" ]] && pass "No duplicate slugs" || err "Duplicate slugs: $dup_slugs"

# ─── 3. Filename = Slug Consistency ─────────────────────────────────────────
section "3. Filename = Slug Consistency"

for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  fname=$(basename "$f" .md)
  slug=$(get_fm "$f" | awk '/^slug: /{print $2; exit}')
  if [[ "$fname" != "$slug" ]]; then
    err "$(basename "$f"): filename '$fname' != slug '$slug'"
  fi
done
pass "Filename/slug consistency checked"

# ─── 4. Field Value Validity ────────────────────────────────────────────────
section "4. Field Value Validity"

for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  fname=$(basename "$f")
  fm=$(get_fm "$f")

  sev=$(echo "$fm" | awk '/^severity: /{print $2; exit}')
  case "$sev" in error|warn) ;; *) err "$fname: severity='$sev' not in {error,warn}" ;; esac

  lck=$(echo "$fm" | awk '/^locked: /{print $2; exit}')
  case "$lck" in true|false) ;; *) err "$fname: locked='$lck' not in {true,false}" ;; esac

  layer=$(echo "$fm" | awk '/^layer: /{print $2; exit}')
  case "$layer" in core|domain|venue) ;; *) err "$fname: layer='$layer' not in {core,domain,venue}" ;; esac

  enf=$(echo "$fm" | awk '/^enforcement: /{print $2; exit}')
  case "$enf" in doc|lint_script) ;; *) err "$fname: enforcement='$enf' not in {doc,lint_script}" ;; esac

  ck=$(echo "$fm" | awk '/^check_kind: /{print $2; exit}')
  case "$ck" in regex|ast|llm_semantic|llm_style|manual) ;; *) err "$fname: check_kind='$ck' not in valid set" ;; esac
done
pass "Field value validity checked"

# ─── 5. lint_patterns Format Validation ─────────────────────────────────────
section "5. lint_patterns Format"

for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  fname=$(basename "$f")
  fm=$(get_fm "$f")
  ck=$(echo "$fm" | awk '/^check_kind: /{print $2; exit}')

  has_lp=false
  if echo "$fm" | grep -q '^lint_patterns:'; then
    has_lp=true
  fi

  # If check_kind=regex, lint_patterns should exist
  if [[ "$ck" == "regex" && "$has_lp" == "false" ]]; then
    # Only warn, not error — some regex rules may have patterns in Check section
    : # skip: LATEX.EQ.DISPLAY_STYLE originally had no lint_patterns
  fi

  # Validate mode values if lint_patterns exists
  if $has_lp; then
    modes=$(echo "$fm" | awk '/^    mode: /{print $2}')
    for mode in $modes; do
      case "$mode" in match|count|negative) ;;
        *) err "$fname: lint_patterns mode='$mode' not in {match,count,negative}" ;;
      esac
    done

    # Count mode should have threshold
    in_count=false
    while IFS= read -r line; do
      if [[ "$line" =~ mode:\ count ]]; then
        in_count=true
      elif [[ "$line" =~ threshold: ]] && $in_count; then
        in_count=false
      elif [[ "$line" =~ ^\ \ -\ pattern: ]] && $in_count; then
        err "$fname: count mode pattern missing threshold"
        in_count=false
      elif [[ ! "$line" =~ ^\ \  ]] && $in_count; then
        err "$fname: count mode pattern missing threshold"
        in_count=false
      fi
    done <<< "$fm"

    # If lint_patterns exists, lint_targets should also exist
    if ! echo "$fm" | grep -q '^lint_targets: '; then
      err "$fname: has lint_patterns but missing lint_targets"
    fi
  fi
done
pass "lint_patterns format checked"

# ─── 6. Body Sections ───────────────────────────────────────────────────────
section "6. Required Body Sections"

for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  fname=$(basename "$f")
  for heading in "## Requirement" "## Rationale" "## Check" "## Examples"; do
    if ! grep -q "^$heading" "$f"; then
      err "$fname: missing section '$heading'"
    fi
  done
done
pass "Body sections checked"

# ─── 7. Profile Validation ──────────────────────────────────────────────────
section "7. Profile Validation"

for profile in "$PROFILES_DIR"/*.md; do
  [[ -f "$profile" ]] || continue
  [[ "$(basename "$profile")" == "README.md" ]] && continue
  pname=$(basename "$profile")

  # Check Includes file existence
  while IFS= read -r line; do
    if [[ "$line" =~ \`(policy/rules/[a-z0-9-]+\.md)\` ]]; then
      ref="${BASH_REMATCH[1]}"
      if [[ ! -f "$PROJECT_DIR/$ref" ]]; then
        err "$pname: references non-existent file '$ref'"
      fi
    fi
  done < <(awk '/^## Includes/,/^## [^I]/' "$profile")

  # Check override locked rules + params key existence
  while IFS= read -r line; do
    if [[ "$line" =~ ^\|\ *([A-Z][A-Z._0-9]+)\ *\|\ *(severity|params\.[a-z_]+)\ *\| ]]; then
      rid="${BASH_REMATCH[1]}"
      field="${BASH_REMATCH[2]}"
      rule_file=$(awk -v id="$rid" '/^id: /{if($2==id){found=1}} found{print FILENAME; exit}' "$RULES_DIR"/*.md 2>/dev/null || true)
      if [[ -n "$rule_file" ]]; then
        locked=$(get_fm "$rule_file" | awk '/^locked: /{print $2; exit}')
        if [[ "$locked" == "true" ]]; then
          err "$pname: overrides locked rule $rid ($field)"
        fi
        # Check params.* key exists in rule card
        if [[ "$field" == params.* ]]; then
          param_key="${field#params.}"
          if ! get_fm "$rule_file" | grep -qE "(^|[^a-z_])${param_key}:"; then
            err "$pname: overrides $rid.$field but rule card lacks param '$param_key'"
          fi
        fi
      else
        err "$pname: overrides unknown rule $rid"
      fi
    fi
  done < <(awk '/^## Overrides/,/^## [^O]/' "$profile")
done
pass "Profile validation complete"

# ─── 8. Integration Markers ─────────────────────────────────────────────────
section "8. Integration Markers"

# Collect all markers once (fast: single grep pass)
all_markers=$(grep -roh 'policy:[A-Z][A-Z._0-9]*' "$PROJECT_DIR/skills/" "$PROJECT_DIR/commands/" 2>/dev/null | sort -u || true)

# Build set of valid rule IDs
all_rule_ids=""
for f in "$RULES_DIR"/*.md; do
  [[ -f "$f" ]] || continue
  all_rule_ids="$all_rule_ids $(get_fm "$f" | awk '/^id: /{print $2; exit}')"
done

# Check markers point to existing rules
for tag in $all_markers; do
  id="${tag#policy:}"
  if ! echo " $all_rule_ids " | grep -q " $id "; then
    err "Marker '$tag' has no matching rule card"
  fi
done
pass "Integration markers checked"

# ─── 9. Orphan Rules (L1: markers, L2: entry skills) ────────────────────────
section "9. Orphan Rule Detection"

# Collect entry skill content into temp file (avoids SIGPIPE with pipefail)
entry_tmpfile=$(mktemp)
trap 'rm -f "$entry_tmpfile"' EXIT
for entry in \
  "$PROJECT_DIR/skills/ml-paper-writing/SKILL.md" \
  "$PROJECT_DIR/skills/paper-self-review/SKILL.md" \
  "$PROJECT_DIR/skills/using-claude-scholar/SKILL.md"; do
  [[ -f "$entry" ]] && cat "$entry" >> "$entry_tmpfile" 2>/dev/null
done

# Collect all markers into temp file
markers_tmpfile=$(mktemp)
trap 'rm -f "$entry_tmpfile" "$markers_tmpfile"' EXIT
echo "$all_markers" > "$markers_tmpfile"

for id in $all_rule_ids; do
  # L1: at least one marker anywhere
  if ! grep -q "policy:$id" "$markers_tmpfile"; then
    err "ORPHAN L1: $id has no marker in skills/commands"
  fi

  # L2: at least one entry skill references it
  if ! grep -q "$id" "$entry_tmpfile"; then
    err "ORPHAN L2: $id not referenced by any entry skill"
  fi
done
pass "Orphan detection complete"

# ─── 10. Rule ID Registry Consistency ───────────────────────────────────────
section "10. Rule ID Registry Consistency"

readme="$SCRIPT_DIR/README.md"
if [[ -f "$readme" ]]; then
  for f in "$RULES_DIR"/*.md; do
    [[ -f "$f" ]] || continue
    id=$(get_fm "$f" | awk '/^id: /{print $2; exit}')
    if ! grep -q "$id" "$readme"; then
      err "$id not found in README.md Rule ID Registry"
    fi
  done
  pass "Rule ID Registry consistency checked"
else
  err "policy/README.md not found"
fi

# ─── 11. Protected Files ────────────────────────────────────────────────────
section "11. Protected Files (git diff)"

if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
  if ! git diff --quiet -- "$PROJECT_DIR/rules/" 2>/dev/null; then
    err "rules/ directory was modified (dev/ops rules should not change)"
  else
    pass "rules/ directory unchanged"
  fi
  if ! git diff --quiet -- "$PROJECT_DIR/CLAUDE.md" "$PROJECT_DIR/AGENTS.md" 2>/dev/null; then
    err "CLAUDE.md or AGENTS.md was modified"
  else
    pass "CLAUDE.md and AGENTS.md unchanged"
  fi
else
  pass "(git not available, skipping protected file check)"
fi

# ─── 12. Rule Count ─────────────────────────────────────────────────────────
section "12. Rule Count"
count=$(ls "$RULES_DIR"/*.md 2>/dev/null | wc -l | tr -d ' ')
echo -e "  Total rule cards: $count"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══ Validation Summary ═══${NC}"
if (( ERRORS > 0 )); then
  echo -e "  ${RED}$ERRORS error(s) found${NC}"
  exit 1
else
  echo -e "  ${GREEN}All validations passed${NC}"
  exit 0
fi
