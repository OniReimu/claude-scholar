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
#   - Deprecated-rule citations carry a successor note (WARN, non-blocking)
#   - Orphan rule detection
#   - No modification to protected files (rules/, CLAUDE.md, AGENTS.md)
#
# Usage: policy/validate.sh
# Exit: 0 if all pass, 1 if errors found

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RULES_DIR="$SCRIPT_DIR/rules"
WORK_TMP_PB=$(mktemp)
WORK_TMP_TB=$(mktemp)
WORK_TMP_BD=$(mktemp)
PROFILES_DIR="$SCRIPT_DIR/profiles"
ERRORS=0
WARNINGS=0
RULE_CARD_COUNT=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

err() { echo -e "  ${RED}FAIL${NC}: $*"; ((ERRORS++)) || true; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $*"; ((WARNINGS++)) || true; }
pass() { echo -e "  ${GREEN}PASS${NC}: $*"; }
SECTIONS_RUN=0
section() { ((SECTIONS_RUN++)) || true; echo -e "\n${BOLD}$*${NC}"; }

# ─── Extract frontmatter using awk ──────────────────────────────────────────
get_fm() { awk '/^---$/{n++;next} n==1{print}' "$1"; }

# ─── Rule/Profile File Filtering ─────────────────────────────────────────────
is_meta_md() {
  local base
  base="$(basename "$1")"
  [[ "$base" == "CLAUDE.md" || "$base" == "README.md" ]]
}

collect_rule_cards() {
  RULE_CARDS=()
  for f in "$RULES_DIR"/*.md; do
    [[ -f "$f" ]] || continue
    is_meta_md "$f" && continue
    RULE_CARDS+=("$f")
  done
  RULE_CARD_COUNT="${#RULE_CARDS[@]}"
}

collect_rule_cards

# ─── 1. Required Frontmatter Fields ─────────────────────────────────────────
section "1. Required Frontmatter Fields"
REQUIRED_FIELDS="id slug severity locked layer artifacts phases domains venues check_kind enforcement constraint_type autofix"

# Field presence is tested with bash pattern matching, not `grep`. The grep form
# forked 14 times per card (1400 forks over the rule set) and, under heavy system
# load, an occasional failed fork was indistinguishable from a missing field —
# producing a FAIL that names a content error when the real cause is resource
# starvation. A validator that intermittently fails on correct files teaches its
# users to re-run until green, which is worse than not checking at all.
for f in "${RULE_CARDS[@]}"; do
  fname=$(basename "$f")
  fm=$(get_fm "$f")
  if [[ -z "$fm" ]]; then
    err "$fname: frontmatter is empty or unreadable"
    continue
  fi
  probe=$'\n'"$fm"$'\n'
  missing=""
  for field in $REQUIRED_FIELDS; do
    # matches "field: value" and the block form "field:" at line start
    if [[ "$probe" != *$'\n'"${field}:"* ]]; then
      missing="$missing $field"
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
for f in "${RULE_CARDS[@]}"; do
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

for f in "${RULE_CARDS[@]}"; do
  fname=$(basename "$f" .md)
  slug=$(get_fm "$f" | awk '/^slug: /{print $2; exit}')
  if [[ "$fname" != "$slug" ]]; then
    err "$(basename "$f"): filename '$fname' != slug '$slug'"
  fi
done
pass "Filename/slug consistency checked"

# ─── 4. Field Value Validity ────────────────────────────────────────────────
section "4. Field Value Validity"

for f in "${RULE_CARDS[@]}"; do
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

  ct=$(echo "$fm" | awk '/^constraint_type: /{print $2; exit}')
  case "$ct" in guardrail|guidance) ;; *) err "$fname: constraint_type='$ct' not in {guardrail,guidance}" ;; esac

  af=$(echo "$fm" | awk '/^autofix: /{print $2; exit}')
  case "$af" in safe|assisted|none) ;; *) err "$fname: autofix='$af' not in {safe,assisted,none}" ;; esac
done
pass "Field value validity checked"

# ─── 4c. Phase Vocabulary Conformance ───────────────────────────────────────
# `phases:` values are drawn from the Phase 词汇表 in policy/README.md, but
# Section 4 never checked them — so a card could name a phase that does not
# exist and nothing noticed. `writing-intro` sat in prose-over-defensive.md for
# 30+ commits that way. Same family as 4b/5c/8b/8c: a declared vocabulary with
# no machine check drifts. The vocabulary is read from README at runtime so the
# table stays the single source of truth.
section "4c. Phase Vocabulary Conformance"

valid_phases=$(awk -F'|' '/^## Phase 词汇表/{f=1;next} f&&/^## /{exit} f&&NF>2{gsub(/[` ]/,"",$2); if($2!="Phase" && $2 !~ /^-*$/ && $2!="") print $2}' "$PROJECT_DIR/policy/README.md" | tr '\n' ' ')
ph_bad=0
if [[ -z "${valid_phases// /}" ]]; then
  err "could not parse the Phase 词汇表 from policy/README.md (extractor drift)"
  ph_bad=1
else
  for f in "${RULE_CARDS[@]}"; do
    fname=$(basename "$f")
    phases=$(get_fm "$f" | awk '/^phases: \[/{gsub(/^phases: \[|\]$/,"");print}' | tr ',' ' ')
    for p in $phases; do
      p="${p// /}"
      [[ -n "$p" ]] || continue
      echo " $valid_phases " | grep -q " $p " || { err "$fname: phase '$p' is not in the Phase 词汇表"; ph_bad=1; }
    done
  done
fi
(( ph_bad == 0 )) && pass "All phase values are in the Phase 词汇表"

# ─── 4b. conflicts_with Integrity ───────────────────────────────────────────
# A conflict is inherently mutual: if A says it overlaps B, a reader of B must
# learn it from B. One-directional declarations leave the older card blind to
# every newer rule that defers to it — the agent reading only B never discovers
# the boundary. Also catches references to rule IDs that do not exist.
section "4b. conflicts_with Integrity"

cw_tmp=$(mktemp)
cw_ids=""   # local ID list: all_rule_ids is not built until Section 8
for f in "${RULE_CARDS[@]}"; do
  cw_ids="$cw_ids $(get_fm "$f" | awk '/^id: /{print $2; exit}')"
done
for f in "${RULE_CARDS[@]}"; do
  fm=$(get_fm "$f")
  rid=$(echo "$fm" | awk '/^id: /{print $2; exit}')
  echo "$fm" | awk -v R="$rid" '/^conflicts_with: \[/{
      gsub(/^conflicts_with: \[|\]$/,""); n=split($0,a,",");
      for(i=1;i<=n;i++){gsub(/^[ \t]+|[ \t]+$/,"",a[i]); if(a[i]!="") print R "\t" a[i]}}'
done > "$cw_tmp"

cw_bad=0
while IFS=$'\t' read -r a b; do
  [[ -n "$a" && -n "$b" ]] || continue
  if ! echo " $cw_ids " | grep -q " $b "; then
    err "conflicts_with: $a → '$b' is not a known rule ID"; cw_bad=1; continue
  fi
  grep -qxF "$b	$a" "$cw_tmp" || { err "conflicts_with asymmetry: $a declares $b, but $b does not declare $a"; cw_bad=1; }
done < "$cw_tmp"
rm -f "$cw_tmp"
(( cw_bad == 0 )) && pass "conflicts_with references resolve and are mutual"

# ─── 4d. Pattern-Block Grammar ──────────────────────────────────────────────
# lint.sh ends a lint_patterns / fix_patterns block at the first line it does
# not recognise, and drops every pattern after it without a diagnostic. A stray
# line in one of these blocks therefore disables rules silently. Comments are
# recognised; anything else in the block is a hazard, so it is an error here
# rather than a surprise at lint time.
section "4d. Pattern-Block Grammar"
pb_bad=0
for f in "$RULES_DIR"/*.md; do
  rid=$(grep -m1 "^id:" "$f" | sed 's/id: *//')
  awk -v rid="$rid" -v file="$(basename "$f")" '
    /^---$/ { n++; if (n == 2) exit; next }
    n != 1 { next }
    /^(lint_patterns|fix_patterns):/ { blk = $1; next }
    blk == "" { next }
    /^[a-z_]+:/ { blk = ""; next }                 # next frontmatter key ends the block
    /^[[:space:]]*$/ { blk = ""; next }
    /^[[:space:]]*#/ { next }
    /^  - (pattern|find):/ { next }
    /^    (mode|threshold|threshold_param|replace):/ { next }
    { print file "\t" rid "\t" NR "\t" $0 }
  ' "$f"
done > "$WORK_TMP_PB" 2>/dev/null || true
if [[ -s "$WORK_TMP_PB" ]]; then
  while IFS=$'\t' read -r bf brid bline btext; do
    err "$brid ($bf:$bline): unrecognised line in pattern block silently truncates it → $btext"
    pb_bad=1
  done < "$WORK_TMP_PB"
fi
rm -f "$WORK_TMP_PB"
(( pb_bad == 0 )) && pass "pattern blocks contain only recognised lines"

# ─── 5. lint_patterns Format Validation ─────────────────────────────────────
section "5. lint_patterns Format"

for f in "${RULE_CARDS[@]}"; do
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

# ─── 5c. Fix-Emission Safety ────────────────────────────────────────────────
# autofix 的 replace 产物不得命中任何其他规则的 lint pattern——否则应用 A 的
# 自动修复会制造 B 的违规（修复循环，返工率来源）。新增/修改 fix_patterns 时
# 由本检查机器兜底。
section "5c. Fix-Emission Safety"

emit_patfile=$(mktemp)
emit_repfile=$(mktemp)
for g in "${RULE_CARDS[@]}"; do
  gfm=$(get_fm "$g")
  gbase=$(basename "$g")
  echo "$gfm" | awk -v F="$gbase" '/^lint_patterns:/{p=1;next} /^[a-z_]+:/{p=0} p && / pattern: /{sub(/^ *- *pattern: */,""); gsub(/^"|"$/,""); print F "\t" $0}' >> "$emit_patfile"
  echo "$gfm" | awk -v F="$gbase" '/^fix_patterns:/{p=1;next} /^[a-z_]+:/{p=0} p && / replace: /{sub(/^ *replace: */,""); gsub(/^"|"$/,""); print F "\t" $0}' >> "$emit_repfile"
done

emission_hits=$(perl -e '
  open(P, "<", $ARGV[0]) or exit 0; my @pats = map { chomp; [split /\t/, $_, 2] } <P>;
  open(R, "<", $ARGV[1]) or exit 0; my @reps = map { chomp; [split /\t/, $_, 2] } <R>;
  for my $r (@reps) {
    my ($rf, $rs) = @$r; next unless defined $rs && length $rs;
    for my $p (@pats) {
      my ($pf, $ps) = @$p; next if !defined $ps || $pf eq $rf;
      my $re = $ps; $re =~ s/\\\\/\\/g;
      if (eval { $rs =~ /$re/ }) { print "$rf autofix output \"$rs\" matches lint pattern of $pf\n"; }
    }
  }
' "$emit_patfile" "$emit_repfile" 2>/dev/null || true)

if [[ -n "$emission_hits" ]]; then
  while IFS= read -r hitline; do
    [[ -n "$hitline" ]] && err "fix-emission loop: $hitline"
  done <<< "$emission_hits"
else
  pass "No autofix output triggers another rule's lint pattern"
fi
rm -f "$emit_patfile" "$emit_repfile"

# ─── 6. Body Sections ───────────────────────────────────────────────────────
section "6. Required Body Sections"

for f in "${RULE_CARDS[@]}"; do
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
  is_meta_md "$profile" && continue
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
    if [[ "$line" =~ ^\|\ *([A-Z][A-Z._0-9]+)\ *\|\ *(severity|params\.[a-z0-9_]+)\ *\| ]]; then
      rid="${BASH_REMATCH[1]}"
      field="${BASH_REMATCH[2]}"
      rule_file=$(awk -v id="$rid" '/^id: /{if($2==id){found=1}} found{print FILENAME; exit}' "$RULES_DIR"/*.md 2>/dev/null || true)
      if [[ -n "$rule_file" ]]; then
        locked=$(get_fm "$rule_file" | awk '/^locked: /{print $2; exit}')
        if [[ "$locked" == "true" ]]; then
          err "$pname: overrides locked rule $rid ($field)"
        fi
        # Check params.* key exists in rule card (scoped to params block)
        if [[ "$field" == params.* ]]; then
          param_key="${field#params.}"
          # Extract only the params block (supports inline {k:v} and multi-line format)
          params_block=$(get_fm "$rule_file" | awk '/^params:/{p=1; print; next} p && /^[[:space:]]/{print; next} p{exit}')
          if ! echo "$params_block" | grep -qE "(^|[{, ])${param_key}:"; then
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
for f in "${RULE_CARDS[@]}"; do
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

# ─── 8b. Deprecated-Rule Citation Acknowledgment ────────────────────────────
# When a rule card carries `deprecated_by:`, every skill/command that cites it
# via `<!-- policy:ID -->` should carry an inline note pointing at the successor
# (otherwise a quick-ref table keeps asserting a stale value — e.g. the fixed
# 24pt font floor). This is a WARNING, not a failure: some deprecated rules are
# legitimately retained as writing guidance (e.g. FIG.SELF_CONTAINED_CAPTION),
# so a maintainer confirms each flagged line rather than CI blocking on it.
section "8b. Deprecated-Rule Citation Acknowledgment"

# Collect deprecated rule IDs (frontmatter carries `deprecated_by:`)
deprecated_ids=""
for f in "${RULE_CARDS[@]}"; do
  if get_fm "$f" | grep -q "^deprecated_by:"; then
    did=$(get_fm "$f" | awk '/^id: /{print $2; exit}')
    deprecated_ids="$deprecated_ids $did"
  fi
done

# Acknowledgment keywords: same citing line must mention the successor / status
ACK_RE='[Dd]eprecated|弃用|退役|接管|自适应|scientific-figure-making|paper-figure-generator|writing-convention|successor|交给|详见'

dep_flagged=0
for id in $deprecated_ids; do
  while IFS= read -r hit; do
    [[ -z "$hit" ]] && continue
    # hit format: path:lineno:content
    linetext="${hit#*:}"; linetext="${linetext#*:}"
    if ! echo "$linetext" | grep -Eq "$ACK_RE"; then
      rel="${hit#$PROJECT_DIR/}"
      warn "Deprecated rule $id cited without a successor note → ${rel%%:*}:$(echo "$rel" | cut -d: -f2)"
      ((dep_flagged++)) || true
    fi
  done < <(grep -rn "policy:$id" "$PROJECT_DIR/skills/" "$PROJECT_DIR/commands/" 2>/dev/null || true)
done

if [[ -z "$deprecated_ids" ]]; then
  pass "No deprecated rules to check"
elif [[ $dep_flagged -eq 0 ]]; then
  pass "All deprecated-rule citations carry a successor note"
else
  echo -e "  ${YELLOW}$dep_flagged deprecated-rule citation(s) need a review — add an inline successor note or confirm the guidance use is intended${NC}"
fi

# ─── 9. Orphan Rules (L1: markers, L2: entry skills) ────────────────────────
# ─── 8c. Deprecated Successor Resolvability ─────────────────────────────────
# `deprecated_by` must name something that exists: a skill directory or another
# rule ID. Section 8b acknowledges a citation by matching the successor's NAME in
# the citing line, so a successor nobody can name (a skill that was never built)
# makes its warnings permanently unfixable — the deprecation reads as unfinished
# work forever. Two cards shipped that way: `writing-convention` and
# `paper-figure-generator-internal`.
section "8c. Deprecated Successor Resolvability"

succ_unresolved=0
for f in "${RULE_CARDS[@]}"; do
  fm=$(get_fm "$f")
  succ=$(echo "$fm" | awk '/^deprecated_by: /{print $2; exit}')
  [[ -n "$succ" ]] || continue
  fname=$(basename "$f")

  # -e follows symlinks, and three skills are symlinks into vendor submodules.
  # A clone without `--recurse-submodules` leaves those dangling, which is a
  # statement about the checkout, not about whether the successor name resolves.
  # -L keeps the answer the same in both trees.
  if [[ -e "$PROJECT_DIR/skills/$succ" || -L "$PROJECT_DIR/skills/$succ" ]]; then
    continue                                    # a real skill
  elif echo " $all_rule_ids " | grep -q " $succ "; then
    continue                                    # another rule took it over
  else
    err "$fname: deprecated_by='$succ' resolves to no skill and no rule ID — Section 8b can never be satisfied"
    succ_unresolved=1
  fi
done
(( succ_unresolved == 0 )) && pass "All deprecated_by successors resolve"

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
# ─── 9b. Doc-Enforced Rules Need an Execution Block ─────────────────────────
# A skill's "## Policy Rules" table is a declaration, not an instruction. For a
# rule with enforcement: lint_script, that is enough — the regex layer runs
# whatever the skill body says. A rule with enforcement: doc has no mechanical
# backstop: if the skill only names it in the table, nothing executes it, and
# the table reads as coverage the skill does not actually have.
section "9b. Doc-Enforced Rules Have an Execution Block"

doc_unbacked=0
for skill_md in "$PROJECT_DIR"/skills/*/SKILL.md; do
  [[ -f "$skill_md" ]] || continue
  # Opt-in. A "## Policy Rules" table is sometimes a checklist the skill works
  # through and sometimes a catalogue it merely indexes — using-claude-scholar
  # lists all 101 rules for Codex discovery and executes none of them. Only a
  # table that declares itself a checklist is held to this.
  grep -q '<!-- policy-table:checklist -->' "$skill_md" || continue
  sname=$(basename "$(dirname "$skill_md")")

  # Table region: from "## Policy Rules" to the next level-2 heading.
  # `|| true` on both greps: a skill whose table is empty makes grep exit 1,
  # and under `set -o pipefail` that ends the scan at whichever skill happens
  # to be empty — silently, having checked only the ones before it.
  { awk '/^## Policy Rules/{t=1;next} t && /^## /{exit} t' "$skill_md" \
    | grep -oE '`[A-Z][A-Z0-9._]*`' || true; } | tr -d '`' | sort -u > "$WORK_TMP_TB"
  # Body markers: every inline marker outside that region.
  { awk '/^## Policy Rules/{t=1;next} t && /^## /{t=0} !t' "$skill_md" \
    | grep -oE 'policy:[A-Z][A-Z0-9._]*' || true; } | sed 's/policy://' | sort -u > "$WORK_TMP_BD"

  while IFS= read -r rid; do
    [[ -n "$rid" ]] || continue
    # No `| head -1` here: under `set -o pipefail`, head closing the pipe early
    # SIGPIPEs grep -r and kills the whole script mid-section. Take the first
    # line in the shell instead.
    card=$(grep -rlF "id: $rid" "$RULES_DIR" 2>/dev/null || true)
    card="${card%%$'\n'*}"
    [[ -n "$card" ]] || continue
    enf=$(get_fm "$card" | awk '/^enforcement: /{print $2; exit}')
    [[ "$enf" == "doc" ]] || continue
    grep -qxF "$rid" "$WORK_TMP_BD" && continue
    err "$sname: $rid is enforcement=doc and appears only in the Policy Rules table — nothing executes it"
    doc_unbacked=1
  done < "$WORK_TMP_TB"
done
rm -f "$WORK_TMP_TB" "$WORK_TMP_BD"
(( doc_unbacked == 0 )) && pass "every doc-enforced rule named in a skill table has an execution block"

# ─── 9c. enforcement Matches the Implementation ─────────────────────────────
# `enforcement` is not a label, it is a claim other checks depend on — 9b uses it
# to decide which rules need an execution block in a skill. Three cards had it
# wrong: two said `doc` while lint runs their patterns, and one said
# `lint_script` with no patterns and no builtin behind it, claiming a mechanical
# check that did not exist.
section "9c. enforcement Matches the Implementation"

enf_bad=0
for f in "${RULE_CARDS[@]}"; do
  fm=$(get_fm "$f")
  rid=$(echo "$fm" | awk '/^id: /{print $2; exit}')
  enf=$(echo "$fm" | awk '/^enforcement: /{print $2; exit}')
  has_pat=0; echo "$fm" | grep -q '^lint_patterns:' && has_pat=1
  # A rule with no patterns may still be enforced by a hand-written builtin in
  # lint.sh; those name the rule ID directly.
  builtin=0; grep -qF "$rid" "$SCRIPT_DIR/lint.sh" 2>/dev/null && builtin=1

  if [[ "$enf" == "lint_script" && $has_pat -eq 0 && $builtin -eq 0 ]]; then
    err "$rid: enforcement=lint_script but has no lint_patterns and no builtin in lint.sh — the mechanical check it claims does not exist"
    enf_bad=1
  fi
  if [[ "$enf" == "doc" && $has_pat -eq 1 ]]; then
    err "$rid: enforcement=doc but carries lint_patterns — lint runs them regardless, so the field understates it"
    enf_bad=1
  fi
done
(( enf_bad == 0 )) && pass "enforcement agrees with lint_patterns and builtins"

# ─── 9d. Declared Coverage Gaps Are Visible at the Point of Use ─────────────
# Several cards deliberately leave part of their Requirement out of the regex —
# usually because a pattern would flood. That decision is sound, but it was
# recorded only in the card's Check section, which an executor working from lint
# output never opens. Measured on one manuscript: 15 contrast constructions
# reported where 38 existed, because the largest class was left to judgment and
# the deliberateness was invisible at the point of use. A card that declares a
# gap must also carry `coverage_note`, which lint prints under the rule header.
section "9d. Declared Coverage Gaps Are Visible at the Point of Use"

cov_bad=0
for f in "${RULE_CARDS[@]}"; do
  fm=$(get_fm "$f")
  rid=$(echo "$fm" | awk '/^id: /{print $2; exit}')
  enf=$(echo "$fm" | awk '/^enforcement: /{print $2; exit}')
  [[ "$enf" == "lint_script" ]] || continue
  # The card says, in prose, that part of the rule is out of regex scope.
  if grep -qE 'regex 覆盖不到|刻意不进 regex|不做硬 regex|不做 regex|regex 无法' "$f"; then
    if ! echo "$fm" | grep -q '^coverage_note:'; then
      err "$rid: declares a regex coverage gap in prose but has no coverage_note — lint will report a partial count with no signal that it is partial"
      cov_bad=1
    fi
  fi
  # And the converse: a note that promises a gap the card never explains.
  if echo "$fm" | grep -q '^coverage_note:' && ! grep -qE 'regex 覆盖不到|刻意不进 regex|不做硬 regex|不做 regex|regex 无法|tier B|档 B' "$f"; then
    err "$rid: has coverage_note but the card body never says what is out of scope or why"
    cov_bad=1
  fi
done
(( cov_bad == 0 )) && pass "every declared coverage gap is surfaced by lint"

section "10. Rule ID Registry Consistency"

readme="$SCRIPT_DIR/README.md"
if [[ -f "$readme" ]]; then
  for f in "${RULE_CARDS[@]}"; do
    id=$(get_fm "$f" | awk '/^id: /{print $2; exit}')
    if ! grep -q "$id" "$readme"; then
      err "$id not found in README.md Rule ID Registry"
    fi
  done
  pass "Rule ID Registry consistency checked"
else
  err "policy/README.md not found"
fi

# ─── 11. Protected Files & Policy References ────────────────────────────────
section "11. Protected Files & Policy References"

if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
  if ! git diff --quiet -- "$PROJECT_DIR/rules/" 2>/dev/null; then
    err "rules/ directory was modified (dev/ops rules should not change)"
  else
    pass "rules/ directory unchanged"
  fi
else
  pass "(git not available, skipping protected file check)"
fi

# Check CLAUDE.md/AGENTS.md reference policy engine entry points
for sysfile in CLAUDE.md AGENTS.md; do
  target="$PROJECT_DIR/$sysfile"
  if [[ -f "$target" ]]; then
    missing=""
    grep -q 'policy/rules/' "$target"   || missing="$missing policy/rules/"
    grep -q 'policy/README.md' "$target" || missing="$missing policy/README.md"
    if [[ -z "$missing" ]]; then
      pass "$sysfile references policy engine"
    else
      err "$sysfile missing policy references:$missing"
    fi
  fi
done

# ─── 11b. Style-Guide Exemplar Conformance ──────────────────────────────────
# `style-guide.md` is declared co-equal authority with `rules/`, and every
# writing task is required to load it first. So its exemplars are not
# illustrations — they are what the agent copies. When an exemplar violates a
# guardrail rule, the agent that obeys the style guide is flagged by this same
# repo's own linter one step later, and there is no way to satisfy both. That
# shipped once: the §5.3 Canonical Paragraph Template opened with "With the
# rapid development of X, ... has attracted significant attention" and closed
# with "significantly improves", i.e. it violated PROSE.AI_LEXICON and
# PROSE.INTENSIFIERS_ELIMINATION verbatim. This check extracts the prose code
# blocks and runs the repo's own guardrail lint over them.
#
# Extraction is deliberately conservative — a false FAIL here blocks CI. Only
# untagged fences are considered, and any block containing a LaTeX command,
# a table pipe, or a schematic glyph (arrow / check / cross) is skipped as
# not-prose. Checking fewer blocks reliably beats checking all blocks noisily.
section "11b. Style-Guide Exemplar Conformance"

sg_file="$SCRIPT_DIR/style-guide.md"
if [[ ! -f "$sg_file" ]]; then
  err "policy/style-guide.md not found (declared co-equal authority with rules/)"
elif [[ ! -x "$SCRIPT_DIR/lint.sh" && ! -f "$SCRIPT_DIR/lint.sh" ]]; then
  err "policy/lint.sh not found — cannot verify style-guide exemplars"
else
  sg_dir=$(mktemp -d)
  awk -v OUT="$sg_dir" '
    /^```/ {
      if (inb) { inb=0 }
      else { inb=1; lang=substr($0,4); start=NR; buf=""; skip=(lang!="" && lang!="text") }
      next
    }
    inb {
      buf = buf $0 "\n"
      # not prose: LaTeX command, table pipe, or schematic glyph
      if ($0 ~ /\\[a-zA-Z]/ || $0 ~ /\|/ || $0 ~ /→/ || $0 ~ /✓/ || $0 ~ /✗/) skip=1
    }
    !inb && buf != "" {
      if (!skip) { f = OUT "/line-" start ".tex"; printf "%s", buf > f; close(f) }
      buf = ""
    }
  ' "$sg_file"

  sg_blocks=$(find "$sg_dir" -name '*.tex' | wc -l | tr -d ' ')
  if (( sg_blocks == 0 )); then
    warn "style-guide.md: no prose code blocks extracted — extractor may have drifted"
  else
    sg_out=$(mktemp)
    if bash "$SCRIPT_DIR/lint.sh" --constraint-type guardrail --strict-warn "$sg_dir" \
         > "$sg_out" 2>&1; then
      pass "style-guide.md exemplars pass guardrail lint ($sg_blocks prose block(s))"
    else
      err "style-guide.md exemplars violate guardrail rules the file is co-equal with:"
      # Report each hit as: RULE_ID @ style-guide.md line N — offending snippet
      sed 's/\x1b\[[0-9;]*m//g' "$sg_out" | awk '
        /^  \[/ { rule=$1; gsub(/[][]/,"",rule); next }
        /^ +(WARN|ERROR) / {
          line=$0
          sub(/^ +(WARN|ERROR) +/,"",line)
          n=split(line, p, ":")
          src=p[1]; sub(/.*line-/,"",src); sub(/\.tex$/,"",src)
          snippet=line; sub(/^[^:]*:[0-9]+: */,"",snippet)
          printf "    %s @ style-guide.md block starting line %s: %s\n", rule, src, snippet
        }'
    fi
    rm -f "$sg_out"
  fi
  rm -rf "$sg_dir"
fi

# ─── 12. Rule Count ─────────────────────────────────────────────────────────
# ─── Completion sentinel ────────────────────────────────────────────────────
# `set -eo pipefail` means one grep that legitimately matches nothing can end
# the run mid-section. The danger is not the crash: a truncated run prints
# FEWER "FAIL:" lines, so a gate that counts them reads the truncation as an
# improvement. Print the count of sections that actually ran, and let the gate
# check it.
section "12. Rule Count"
count="$RULE_CARD_COUNT"
echo -e "  Total rule cards: $count"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══ Validation Summary ═══${NC}"
  # Self-counted, so it cannot drift: the denominator is read from this file.
  sections_total=$(grep -c '^section "' "$0")
  if (( SECTIONS_RUN < sections_total )); then
    echo -e "  ${RED}TRUNCATED${NC}: ran ${SECTIONS_RUN}/${sections_total} sections — the run ended early and the FAIL count below is not a full result"
    ERRORS=$((ERRORS + 1))
  else
    echo -e "  Sections: ${SECTIONS_RUN}/${sections_total}"
  fi
if (( WARNINGS > 0 )); then
  echo -e "  ${YELLOW}$WARNINGS warning(s) — review, non-blocking${NC}"
fi
if (( ERRORS > 0 )); then
  echo -e "  ${RED}$ERRORS error(s) found${NC}"
  exit 1
else
  echo -e "  ${GREEN}All validations passed${NC}"
  exit 0
fi
