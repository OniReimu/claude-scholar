#!/usr/bin/env bash
# policy/lint.sh — Policy Engine M2 regex linter
#
# Reads lint_patterns/lint_targets from rule card frontmatter,
# runs regex checks on target files, reports violations.
#
# Usage: policy/lint.sh [OPTIONS] [TARGET_DIR]
#   --profile FILE    Apply profile overrides (severity, params)
#   --strict-warn     Treat warnings as errors (exit 1)
#   --quiet           Only show summary, suppress per-match output
#   --rule RULE_ID    Lint only a specific rule
#   --layer LAYER     Lint only rules of a specific layer (core|domain|venue)
#   -h, --help        Show help
#
# Exit codes:
#   0 - All pass (or warnings only without --strict-warn)
#   1 - At least one error-severity violation found (or warn with --strict-warn)
#   2 - Script/configuration error
#
# Threshold semantics: count > threshold = violation
#   e.g. threshold=6 means up to 6 allowed, 7+ is a violation.
#
# Negative mode: scoped to files matching any 'match' pattern in the same rule.
#   If no match patterns exist in the rule, checks all target files.
#
# Pattern engine: prefers grep -P (GNU grep), falls back to perl (macOS default).

set -eo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Globals ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RULES_DIR="$SCRIPT_DIR/rules"
STRICT_WARN=false
QUIET=false
PROFILE=""
TARGET_DIR=""
FILTER_RULE=""
FILTER_LAYER=""
GREP_MODE=""     # "ggrep" | "grep" | "perl"
TOTAL_ERRORS=0
TOTAL_WARNINGS=0
RULES_CHECKED=0
RULES_PASSED=0

# ─── Help ────────────────────────────────────────────────────────────────────
show_help() {
  sed -n '2,/^$/s/^# //p' "$0"
  exit 0
}

# ─── Argument Parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)     PROFILE="$2"; shift 2 ;;
    --strict-warn) STRICT_WARN=true; shift ;;
    --quiet)       QUIET=true; shift ;;
    --rule)        FILTER_RULE="$2"; shift 2 ;;
    --layer)       FILTER_LAYER="$2"; shift 2 ;;
    -h|--help)     show_help ;;
    -*)            echo "Unknown option: $1" >&2; exit 2 ;;
    *)             TARGET_DIR="$1"; shift ;;
  esac
done

TARGET_DIR="${TARGET_DIR:-.}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: Target directory '$TARGET_DIR' does not exist" >&2
  exit 2
fi

if [[ ! -d "$RULES_DIR" ]]; then
  echo "ERROR: Rules directory '$RULES_DIR' not found" >&2
  exit 2
fi

# ─── Pattern Engine Detection ────────────────────────────────────────────────
# Perl regex support is required for \b, \d, \w, (?!...) in patterns.
# Priority: ggrep -P (Homebrew) > grep -P (Linux) > perl (macOS default)
detect_engine() {
  if command -v ggrep &>/dev/null && echo "test" | ggrep -P "test" &>/dev/null 2>&1; then
    GREP_MODE="ggrep"; return 0
  fi
  if echo "test" | grep -P "test" &>/dev/null 2>&1; then
    GREP_MODE="grep"; return 0
  fi
  if command -v perl &>/dev/null; then
    GREP_MODE="perl"; return 0
  fi
  return 1
}

if ! detect_engine; then
  echo "ERROR: No Perl-compatible regex engine found (need grep -P or perl)" >&2
  exit 2
fi

# ─── Regex Helpers ───────────────────────────────────────────────────────────
# Pattern passed via env var to avoid shell interpolation issues.

# Returns matching lines as "filename:lineno: content"
regex_match() {
  local pattern="$1" file="$2"
  case "$GREP_MODE" in
    ggrep) ggrep -Pn "$pattern" "$file" 2>/dev/null || true ;;
    grep)  grep -Pn "$pattern" "$file" 2>/dev/null || true ;;
    perl)  LINT_PAT="$pattern" perl -ne 'print "$ARGV:$.: $_" if /$ENV{LINT_PAT}/' "$file" 2>/dev/null || true ;;
  esac
}

# Returns count of matching lines
regex_count() {
  local pattern="$1" file="$2"
  case "$GREP_MODE" in
    ggrep) ggrep -Pc "$pattern" "$file" 2>/dev/null || echo 0 ;;
    grep)  grep -Pc "$pattern" "$file" 2>/dev/null || echo 0 ;;
    perl)  LINT_PAT="$pattern" perl -ne '$c++ if /$ENV{LINT_PAT}/; END{print $c//0}' "$file" 2>/dev/null || echo 0 ;;
  esac
}

# Returns 0 if pattern found, 1 if not
regex_quiet() {
  local pattern="$1" file="$2"
  case "$GREP_MODE" in
    ggrep) ggrep -Pq "$pattern" "$file" 2>/dev/null ;;
    grep)  grep -Pq "$pattern" "$file" 2>/dev/null ;;
    perl)  LINT_PAT="$pattern" perl -ne 'BEGIN{$f=1} $f=0 if /$ENV{LINT_PAT}/; END{exit $f}' "$file" 2>/dev/null ;;
  esac
}

# ─── YAML Unescape ──────────────────────────────────────────────────────────
# YAML double-quoted strings: \\ → \
yaml_unescape() {
  printf '%s' "$1" | sed 's/\\\\/\\/g'
}

# ─── Find Target Files ──────────────────────────────────────────────────────
find_target_files() {
  local glob="$1" dir="$2"
  local name_pattern="${glob##*/}"  # **/*.tex → *.tex
  find "$dir" -name "$name_pattern" -type f 2>/dev/null | sort
}

# ─── Profile Override Parsing ────────────────────────────────────────────────
declare -a PROFILE_OVERRIDES=()

parse_profile() {
  local profile_file="$1"
  if [[ ! -f "$profile_file" ]]; then
    echo "ERROR: Profile file '$profile_file' not found" >&2
    exit 2
  fi

  local in_overrides=false
  while IFS= read -r line; do
    if [[ "$line" == "## Overrides" ]]; then
      in_overrides=true; continue
    fi
    # Any other ## heading ends the Overrides section
    if $in_overrides && [[ "$line" =~ ^##\  ]] && [[ "$line" != "## Overrides" ]]; then
      in_overrides=false; continue
    fi
    if $in_overrides; then
      # Skip header and separator
      [[ "$line" =~ Rule\ ID ]] && continue
      [[ "$line" =~ ^\|\ *-+ ]] && continue
      # Parse: | RULE_ID | field | value | reason |
      if [[ "$line" =~ ^\|\ *([A-Z][A-Z._]+)\ *\|\ *([a-z._]+)\ *\|\ *([^|]+)\ *\| ]]; then
        local rid="${BASH_REMATCH[1]}"
        local field="${BASH_REMATCH[2]}"
        local value="${BASH_REMATCH[3]}"
        value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        PROFILE_OVERRIDES+=("${rid}|${field}|${value}")
      fi
    fi
  done < "$profile_file"
}

# Returns effective severity after profile override (respects locked)
get_effective_severity() {
  local rule_id="$1" locked="$2" default_sev="$3"
  for override in "${PROFILE_OVERRIDES[@]}"; do
    IFS='|' read -r oid field value <<< "$override"
    if [[ "$oid" == "$rule_id" && "$field" == "severity" ]]; then
      if [[ "$locked" == "true" ]]; then
        $QUIET || echo -e "  ${DIM}(profile severity override ignored: $rule_id is locked)${NC}" >&2
        echo "$default_sev"; return
      fi
      echo "$value"; return
    fi
  done
  echo "$default_sev"
}

# ─── Frontmatter Parser ─────────────────────────────────────────────────────
# Single-pass parser using pure bash (no grep/sed in pipelines).
# Sets: RULE_ID, RULE_SEVERITY, RULE_LOCKED, RULE_LAYER, RULE_CHECK_KIND,
#       RULE_LINT_TARGETS, PATTERNS[] (entries: "pattern\tmode\tthreshold")
RULE_ID="" RULE_SEVERITY="" RULE_LOCKED="" RULE_LAYER=""
RULE_CHECK_KIND="" RULE_LINT_TARGETS=""
declare -a PATTERNS=()

parse_rule() {
  local file="$1"

  RULE_ID="" RULE_SEVERITY="" RULE_LOCKED="" RULE_LAYER=""
  RULE_CHECK_KIND="" RULE_LINT_TARGETS=""
  PATTERNS=()

  # Extract frontmatter (between --- markers)
  local frontmatter
  frontmatter=$(awk '/^---$/{n++;next} n==1{print}' "$file")

  # Single-pass: parse simple fields and lint_patterns block together
  local in_lp=false
  local cur_pat="" cur_mode="match" cur_thresh=""

  while IFS= read -r line; do
    if $in_lp; then
      if [[ "$line" =~ ^\ \ -\ pattern:\ \"(.+)\" ]]; then
        [[ -n "$cur_pat" ]] && PATTERNS+=("${cur_pat}"$'\t'"${cur_mode}"$'\t'"${cur_thresh}")
        cur_pat="${BASH_REMATCH[1]}"; cur_mode="match"; cur_thresh=""
      elif [[ "$line" =~ ^\ \ \ \ mode:\ (.+) ]]; then
        cur_mode="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^\ \ \ \ threshold:\ ([0-9]+) ]]; then
        cur_thresh="${BASH_REMATCH[1]}"
      else
        # Non-indented line ends the block; save last entry
        [[ -n "$cur_pat" ]] && PATTERNS+=("${cur_pat}"$'\t'"${cur_mode}"$'\t'"${cur_thresh}")
        cur_pat=""; in_lp=false
        # Fall through to parse this line as a regular key
      fi
    fi
    if ! $in_lp; then
      case "$line" in
        "id: "*)           RULE_ID="${line#id: }" ;;
        "severity: "*)     RULE_SEVERITY="${line#severity: }" ;;
        "locked: "*)       RULE_LOCKED="${line#locked: }" ;;
        "layer: "*)        RULE_LAYER="${line#layer: }" ;;
        "check_kind: "*)   RULE_CHECK_KIND="${line#check_kind: }" ;;
        "lint_targets: "*) local t="${line#lint_targets: }"; RULE_LINT_TARGETS="${t//\"/}" ;;
        "lint_patterns:")  in_lp=true ;;
      esac
    fi
  done <<< "$frontmatter"

  # Save last pattern entry if block was at end of frontmatter
  [[ -n "$cur_pat" ]] && PATTERNS+=("${cur_pat}"$'\t'"${cur_mode}"$'\t'"${cur_thresh}")
  return 0
}

# ─── Report Finding ─────────────────────────────────────────────────────────
report_finding() {
  local severity="$1" rule_id="$2" detail="$3"
  if [[ "$severity" == "error" ]]; then
    ((TOTAL_ERRORS++)) || true
    $QUIET || printf '    \033[0;31mERROR\033[0m %s\n' "$detail"
  else
    ((TOTAL_WARNINGS++)) || true
    $QUIET || printf '    \033[1;33mWARN\033[0m  %s\n' "$detail"
  fi
}

# ─── Lint Single Rule ───────────────────────────────────────────────────────
lint_rule() {
  local severity="$1"
  local findings=0

  # Categorize patterns by mode
  local -a m_pats=() c_pats=() c_threshs=() n_pats=()
  for entry in "${PATTERNS[@]}"; do
    IFS=$'\t' read -r pat mode thresh <<< "$entry"
    case "$mode" in
      match)    m_pats+=("$pat") ;;
      count)    c_pats+=("$pat"); c_threshs+=("$thresh") ;;
      negative) n_pats+=("$pat") ;;
    esac
  done

  # Collect target files
  local -a tfiles=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && tfiles+=("$f")
  done < <(find_target_files "$RULE_LINT_TARGETS" "$TARGET_DIR")

  if [[ ${#tfiles[@]} -eq 0 ]]; then return; fi

  # Files flagged by match patterns (for negative scoping)
  local -a flagged_files=()

  # ── Match mode: each line match is a violation ──
  for raw_pat in "${m_pats[@]}"; do
    local pat
    pat=$(yaml_unescape "$raw_pat")
    for file in "${tfiles[@]}"; do
      local matches
      matches=$(regex_match "$pat" "$file")
      if [[ -n "$matches" ]]; then
        flagged_files+=("$file")
        while IFS= read -r mline; do
          ((findings++)) || true
          report_finding "$severity" "$RULE_ID" "$mline"
        done <<< "$matches"
      fi
    done
  done

  # ── Count mode: per-file, count > threshold is a violation ──
  for i in "${!c_pats[@]}"; do
    local raw_pat="${c_pats[$i]}"
    local thresh="${c_threshs[$i]:-0}"
    local pat
    pat=$(yaml_unescape "$raw_pat")
    for file in "${tfiles[@]}"; do
      local cnt
      cnt=$(regex_count "$pat" "$file")
      if (( cnt > thresh )); then
        ((findings++)) || true
        report_finding "$severity" "$RULE_ID" "${file}: count=${cnt} > threshold=${thresh}"
      fi
    done
  done

  # ── Negative mode: pattern NOT found in scoped files = violation ──
  if [[ ${#n_pats[@]} -gt 0 ]]; then
    local -a scope=()
    if [[ ${#m_pats[@]} -gt 0 && ${#flagged_files[@]} -gt 0 ]]; then
      # Scope to files that matched the 'match' patterns
      while IFS= read -r f; do
        scope+=("$f")
      done < <(printf '%s\n' "${flagged_files[@]}" | sort -u)
    else
      scope=("${tfiles[@]}")
    fi

    for raw_pat in "${n_pats[@]}"; do
      local pat
      pat=$(yaml_unescape "$raw_pat")
      for file in "${scope[@]}"; do
        if ! regex_quiet "$pat" "$file"; then
          ((findings++)) || true
          report_finding "$severity" "$RULE_ID" "${file}: MISSING required pattern"
        fi
      done
    done
  fi
}

# ─── Load Profile ───────────────────────────────────────────────────────────
if [[ -n "$PROFILE" ]]; then
  parse_profile "$PROFILE"
  $QUIET || echo -e "${CYAN}Profile:${NC} $PROFILE (${#PROFILE_OVERRIDES[@]} overrides)"
fi

# ─── Main Loop ───────────────────────────────────────────────────────────────
$QUIET || echo -e "${BOLD}Policy Engine Lint${NC} — scanning ${TARGET_DIR}  (engine: ${GREP_MODE})"
$QUIET || echo ""

for rule_file in "$RULES_DIR"/*.md; do
  [[ -f "$rule_file" ]] || continue

  parse_rule "$rule_file"

  # Skip non-regex rules
  [[ "$RULE_CHECK_KIND" == "regex" ]] || continue
  # Skip rules without lint_patterns
  [[ ${#PATTERNS[@]} -gt 0 ]] || continue
  # Skip rules without lint_targets
  [[ -n "$RULE_LINT_TARGETS" ]] || continue

  # Apply filters
  [[ -z "$FILTER_RULE" || "$RULE_ID" == "$FILTER_RULE" ]] || continue
  [[ -z "$FILTER_LAYER" || "$RULE_LAYER" == "$FILTER_LAYER" ]] || continue

  # Apply profile severity override
  local_severity=$(get_effective_severity "$RULE_ID" "$RULE_LOCKED" "$RULE_SEVERITY")

  $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (${local_severity}) ${#PATTERNS[@]} pattern(s) → ${RULE_LINT_TARGETS}"

  ((RULES_CHECKED++)) || true

  prev_e=$TOTAL_ERRORS
  prev_w=$TOTAL_WARNINGS

  lint_rule "$local_severity"

  if (( TOTAL_ERRORS == prev_e && TOTAL_WARNINGS == prev_w )); then
    ((RULES_PASSED++)) || true
    $QUIET || echo -e "    ${GREEN}PASS${NC}"
  fi
  $QUIET || echo ""
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══ Summary ═══${NC}"
echo -e "  Rules checked:  ${RULES_CHECKED}"
echo -e "  Rules passed:   ${GREEN}${RULES_PASSED}${NC}"
echo -e "  Errors:         ${RED}${TOTAL_ERRORS}${NC}"
echo -e "  Warnings:       ${YELLOW}${TOTAL_WARNINGS}${NC}"
if $STRICT_WARN; then
  echo -e "  ${DIM}(--strict-warn active: warnings treated as errors)${NC}"
fi

# ─── Exit Code ───────────────────────────────────────────────────────────────
if (( TOTAL_ERRORS > 0 )); then
  exit 1
elif $STRICT_WARN && (( TOTAL_WARNINGS > 0 )); then
  exit 1
else
  exit 0
fi
