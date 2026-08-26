#!/usr/bin/env bash
# policy/test-referrals.sh — referral-graph integrity across skills.
#
# A referral edge is a line in a skill that carries both a rule marker
# (<!-- policy:RULE_ID -->) and a backticked reference to a *different* skill.
# That is the shape the prose already uses; no new annotation is introduced.
#
# Three checks, matching the three ways a referral rots:
#   R1  destination skill does not exist (renamed or typo)
#   R2  destination does not execute the rule it is handed
#   R3  referral names only a skill, not the minimum operation to run
#
# R3 exists because a referral that reads "run claim-architecture-review" costs
# the author a four-pass whole-document audit when they wanted one section
# cleaned. In practice that referral is declined, which makes the rule
# unreachable from the entry point the author actually uses.
#
# Usage: bash policy/test-referrals.sh [--verbose]
# Exit:  0 = graph intact, 1 = at least one broken edge, 2 = script error

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
RULES_DIR="$SCRIPT_DIR/rules"
VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'
PASS=0; FAIL=0; WAIVED=0

[[ -d "$SKILLS_DIR" ]] || { echo "ERROR: skills dir not found" >&2; exit 2; }

# ─── Documented exceptions to R2 ─────────────────────────────────────────────
# "<RULE_ID>|<destination-skill>|<why the destination legitimately lacks the marker>"
# A skill carries a rule's marker only when it executes that rule's check. A
# referral may still be correct when the destination handles the finding through
# a *different* rule it does own; that is what these entries record.
R2_WAIVERS=(
"PROSE.SELF_UNDERMINING|claim-architecture-review|word-level rule; arch-review does no word-level work. The cross-section half of the finding is a caveat with several homes, which P2 already owns via PROSE.OVER_DEFENSIVE."
)

# Destinations where a referral must name the minimum operation.
R3_SCOPED="claim-architecture-review paper-self-review"

# Tokens that count as naming a concrete operation at the destination.
mechanism_tokens() {
  case "$1" in
    claim-architecture-review) echo "P0 P1 P2 P3 spine.md information-ledger.md relocation-map.md paragraph-audit.md lookup-before-create relocation-map 转诊说明" ;;
    ml-paper-writing)          echo "Step 选对战场 positioning" ;;
    paper-self-review)         echo "checklist Step" ;;
    *)                         echo "P0 P1 P2 P3 Step" ;;
  esac
}

known_skill() { [[ -f "$SKILLS_DIR/$1/SKILL.md" ]]; }
known_rule()  { grep -rqlF "id: $1" "$RULES_DIR" 2>/dev/null; }

echo -e "${BOLD}Referral Graph Test${NC}"
echo ""

EDGES=$(mktemp); trap 'rm -f "$EDGES"' EXIT

# A referral edge needs three things on the page, not two:
#   - a rule marker governing the block. Markers sit on the heading while the
#     referral prose sits in the body under it, so attribution is section-scoped.
#   - an explicit referral verb: 转 / 归 / 交 / refer / hand off / defer
#   - a backticked name that resolves to a real skill
# Naming another skill is not referring a rule to it. "Are figures consistent
# with `scientific-figure-making` style" is this skill running the check itself;
# counting that as an edge would demand markers the destination must not carry.
for skill_md in "$SKILLS_DIR"/*/SKILL.md; do
  src=$(basename "$(dirname "$skill_md")")
  awk -v src="$src" '
    function has_verb(s,   i) {
      # Chinese verbs are unambiguous as substrings. The English ones are not:
      # a bare "refer" matches inside "reference validation", which is a
      # checklist item, not a referral.
      split("转|归|交|转诊", v, "|")
      for (i in v) if (index(s, v[i])) return 1
      if (s ~ /[Rr]efers? to|[Hh]ands? off|[Dd]efers? to|[Hh]anded off/) return 1
      return 0
    }
    /^#+ / { delete cur; ncur = 0 }
    {
      line = $0
      n = 0; tmp = line
      while (match(tmp, /policy:[A-Z][A-Z0-9._]*/)) {
        marks[++n] = substr(tmp, RSTART + 7, RLENGTH - 7)
        tmp = substr(tmp, RSTART + RLENGTH)
      }
      if (n > 0) { delete cur; ncur = n; for (i = 1; i <= n; i++) cur[i] = marks[i] }
      if (ncur == 0) next
      if (!has_verb(line)) next
      d = 0; tmp = line
      while (match(tmp, /`[a-z][a-z0-9-]+`/)) {
        cand = substr(tmp, RSTART + 1, RLENGTH - 2)
        if (cand != src) { dests[++d] = cand }
        tmp = substr(tmp, RSTART + RLENGTH)
      }
      for (i = 1; i <= d; i++)
        for (j = 1; j <= ncur; j++)
          print src "\t" cur[j] "\t" dests[i] "\t" NR "\t" line
      delete dests
    }
  ' "$skill_md" >> "$EDGES"
done

# Drop backticked tokens that are not skills, and de-duplicate.
awk -F'\t' -v sd="$SKILLS_DIR" '
  { f = sd "/" $3 "/SKILL.md"; if ((getline t < f) >= 0) { close(f); if (!seen[$1"|"$2"|"$3]++) print } }
' "$EDGES" > "$EDGES.f" && mv "$EDGES.f" "$EDGES"

[[ -s "$EDGES" ]] || { echo "ERROR: no referral edges found — the detector is broken, not the graph" >&2; exit 2; }

while IFS=$'\t' read -r src rule dest lineno text; do
  label="$src:$lineno  $rule → $dest"

  # R1
  if ! known_skill "$dest"; then
    echo -e "  ${RED}✗ R1${NC} $label — destination skill does not exist"; ((FAIL++)) || true; continue
  fi
  if ! known_rule "$rule"; then
    echo -e "  ${RED}✗ R1${NC} $label — rule ID not in policy/rules/"; ((FAIL++)) || true; continue
  fi

  # R2
  if grep -qF "policy:$rule" "$SKILLS_DIR/$dest/SKILL.md"; then
    r2=ok
  elif [[ -L "$SKILLS_DIR/$dest" ]]; then
    # Vendored upstream skill reached through a submodule symlink. Markers are
    # not added here — the edit belongs upstream — so the marker can never
    # appear no matter how correct the referral is.
    r2=waived; why="vendored upstream skill (submodule symlink); markers are not written into vendor trees"
  else
    r2=missing
    for w in "${R2_WAIVERS[@]}"; do
      if [[ "$w" == "$rule|$dest|"* ]]; then r2=waived; why="${w##*|}"; break; fi
    done
  fi
  if [[ "$r2" == missing ]]; then
    echo -e "  ${RED}✗ R2${NC} $label — destination does not carry this rule's marker"
    echo -e "        ${DIM}either the destination should execute it, or add a waiver with the reason${NC}"
    ((FAIL++)) || true; continue
  fi

  # R3 — only for destinations where a bare skill name costs a whole-document
  # run. Elsewhere "refer to X" is a complete instruction and the check would
  # be inventing a requirement.
  case " $R3_SCOPED " in
    *" $dest "*) ;;
    *) ((PASS++)) || true; $VERBOSE && echo -e "  ${GREEN}✓${NC} $label"; continue ;;
  esac
  found=""
  for tok in $(mechanism_tokens "$dest"); do
    case "$text" in *"$tok"*) found="$tok"; break;; esac
  done
  if [[ -z "$found" ]]; then
    echo -e "  ${RED}✗ R3${NC} $label — names the skill but not the minimum operation to run"
    ((FAIL++)) || true; continue
  fi

  if [[ "$r2" == waived ]]; then
    ((WAIVED++)) || true
    echo -e "  ${YELLOW}~${NC} $label ${DIM}via waiver: ${why}${NC}"
  else
    ((PASS++)) || true
    $VERBOSE && echo -e "  ${GREEN}✓${NC} $label ${DIM}(→ $found)${NC}"
  fi
done < "$EDGES"

echo ""
echo -e "  direct $PASS   waived $WAIVED   broken $FAIL"
[[ $FAIL -eq 0 ]]
