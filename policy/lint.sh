#!/usr/bin/env bash
# policy/lint.sh — Policy Engine M2 regex linter
#
# Reads lint_patterns/lint_targets from rule card frontmatter,
# runs regex checks on target files, reports violations.
#
# Usage: policy/lint.sh [OPTIONS] [TARGET_DIR]
#   --profile FILE              Apply profile overrides (severity, params)
#   --strict-warn               Treat warnings as errors (exit 1)
#   --quiet                     Only show summary, suppress per-match output
#   --rule RULE_ID              Lint only a specific rule
#   --layer LAYER               Lint only rules of a specific layer (core|domain|venue)
#   --constraint-type TYPE      Filter by constraint_type (guardrail|guidance)
#   --autofix LEVEL             Filter by autofix (safe|assisted|none)
#   --fix                       Auto-fix safe violations (only applies to autofix=safe rules)
#   -h, --help                  Show help
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
FILTER_CONSTRAINT_TYPE=""
FILTER_AUTOFIX=""
FIX_MODE=false
GREP_MODE=""     # "ggrep" | "grep" | "perl"
TOTAL_FIXES=0
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
    --profile)          PROFILE="$2"; shift 2 ;;
    --strict-warn)      STRICT_WARN=true; shift ;;
    --quiet)            QUIET=true; shift ;;
    --rule)             FILTER_RULE="$2"; shift 2 ;;
    --layer)            FILTER_LAYER="$2"; shift 2 ;;
    --constraint-type)  FILTER_CONSTRAINT_TYPE="$2"; shift 2 ;;
    --autofix)          FILTER_AUTOFIX="$2"; shift 2 ;;
    --fix)              FIX_MODE=true; shift ;;
    -h|--help)          show_help ;;
    -*)                 echo "Unknown option: $1" >&2; exit 2 ;;
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

# Validate enum filters
if [[ -n "$FILTER_CONSTRAINT_TYPE" ]]; then
  case "$FILTER_CONSTRAINT_TYPE" in
    guardrail|guidance) ;;
    *) echo "ERROR: --constraint-type must be 'guardrail' or 'guidance'" >&2; exit 2 ;;
  esac
fi
if [[ -n "$FILTER_AUTOFIX" ]]; then
  case "$FILTER_AUTOFIX" in
    safe|assisted|none) ;;
    *) echo "ERROR: --autofix must be 'safe', 'assisted', or 'none'" >&2; exit 2 ;;
  esac
fi

# ─── Pattern Engine Detection ────────────────────────────────────────────────
# Perl regex support is required for \b, \d, \w, (?!...) in patterns.
# Priority: ggrep -P (Homebrew) > grep -P (Linux) > perl (macOS default)
# LINT_ENGINE=ggrep|grep|perl forces one engine. The engines are not
# interchangeable in practice — macOS resolves to perl and Linux CI to GNU grep,
# so an engine-specific defect is invisible to whichever half does not run it.
# policy/test-corpus.sh --engine uses this to check both against one corpus.
detect_engine() {
  if [[ -n "${LINT_ENGINE:-}" ]]; then
    GREP_MODE="$LINT_ENGINE"; return 0
  fi
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

# ─── Per-Pattern Timeout ─────────────────────────────────────────────────────
# One pattern must not be able to hang the whole run. A regex with ambiguous
# adjacent quantifiers backtracks exponentially on the right input, and the
# failure is silent from the caller's side — the tool simply never returns.
# Worse, killing the caller leaves the engine process orphaned and still
# burning CPU, so each hang makes the next run slower.
# A timed-out pattern is REPORTED, never treated as "no matches": a pattern
# that silently returns nothing reads as a clean file.
LINT_TIMEOUT=""
if command -v timeout &>/dev/null; then LINT_TIMEOUT="timeout 20"
elif command -v gtimeout &>/dev/null; then LINT_TIMEOUT="gtimeout 20"; fi
PATTERN_TIMEOUTS=0

# ─── Comment-Blanked View (.tex only) ────────────────────────────────────────
# Prose sitting behind a % is not part of the manuscript. Linting it reports
# violations the author cannot act on — deleting a comment is not an edit to the
# paper — and an author who learns to ignore one report learns to ignore them
# all. Blank the comment body rather than drop the line, so file:line still
# points at the source. \% is an escaped percent and does not open a comment.
# .bib is excluded: % is not a BibTeX comment.
# The cache key is a hash of the path, so an existing view file *is* the cache
# entry. Associative arrays are bash 4; macOS ships bash 3.2 and this repo runs
# there.
VIEW_DIR=""
lint_view() {
  local file="$1"
  case "$file" in
    *.tex) ;;
    *) printf '%s' "$file"; return ;;
  esac
  [[ -n "$VIEW_DIR" ]] || VIEW_DIR=$(mktemp -d)
  local out
  out="$VIEW_DIR/$(printf '%s' "$file" | cksum | tr -d ' /').tex"
  if [[ ! -f "$out" ]]; then
    # Two passes. First blank comment bodies. Then rejoin hard-wrapped prose:
    # every pattern here matches within one line, and traditional LaTeX wraps at
    # ~72 columns, so a 35-word sentence — or a plain "X, so Y" — is split across
    # lines and silently never matches. Measured on 20 pre-2022 arXiv sources:
    # 1920 sentences split by a hard wrap, against 476 in a 2026 sample. Linting
    # them as-is compares typesetting convention, not prose.
    # A joined paragraph is emitted on its FIRST line with the remaining lines
    # blanked, so file:line still points at where the author must look.
    sed -E 's/(^|[^\\])%.*$/\1/' "$file" 2>/dev/null \
      | perl -0777 -ne '
          my @in = split /\n/, $_, -1;
          my @out = ("") x scalar(@in);
          my ($start, $buf) = (-1, "");
          my $flush = sub { if ($start >= 0) { $out[$start] = $buf; $start = -1; $buf = ""; } };
          for my $i (0 .. $#in) {
            my $l = $in[$i];
            # Only prose lines are joined. A blank line, or one that begins or
            # ends with LaTeX markup, keeps its own line so environments,
            # display math and \\ breaks are not welded together.
            if ($l =~ /^\s*$/ || $l =~ /^\s*\\/ || $l =~ /\\\\\s*$/) {
              $flush->(); $out[$i] = $l; next;
            }
            if ($start < 0) { $start = $i; $buf = $l }
            else { $buf .= " " . $l =~ s/^\s+//r }
          }
          $flush->();
          print join("\n", @out);
        ' > "$out" 2>/dev/null || sed -E 's/(^|[^\\])%.*$/\1/' "$file" > "$out" 2>/dev/null || cp "$file" "$out"
  fi
  printf '%s' "$out"
}
# Preserve the exit status: an EXIT trap returns the status of its last command,
# and a bare [[ ]] that tests false would turn a clean run into exit 1.
cleanup_views() { local st=$?; [[ -n "$VIEW_DIR" ]] && rm -rf "$VIEW_DIR"; return $st; }
trap cleanup_views EXIT

# ─── Regex Helpers ───────────────────────────────────────────────────────────
# Pattern passed via env var to avoid shell interpolation issues, then decoded
# before use: %ENV hands back bytes, and a byte-wise character class matches any
# character sharing a lead byte with a member (an em dash reads as an arrow).

# Returns matching lines as "filename:lineno: content"
regex_match() {
  local pattern="$1" file="$2"
  case "$GREP_MODE" in
    ggrep) $LINT_TIMEOUT ggrep -Pn "$pattern" "$file" 2>/dev/null; [[ $? -eq 124 ]] && echo "__LINT_TIMEOUT__:${file}"; true ;;
    grep)  $LINT_TIMEOUT grep -Pn "$pattern" "$file" 2>/dev/null; [[ $? -eq 124 ]] && echo "__LINT_TIMEOUT__:${file}"; true ;;
    perl)  LINT_PAT="$pattern" $LINT_TIMEOUT perl -CSD -ne 'BEGIN{$p=$ENV{LINT_PAT}; utf8::decode($p); $re=qr/$p/} print "$ARGV:$.: $_" if /$re/' "$file" 2>/dev/null; [[ $? -eq 124 ]] && echo "__LINT_TIMEOUT__:${file}"; true ;;
  esac
}

# Returns count of individual matches (not matching lines)
regex_count() {
  local pattern="$1" file="$2"
  case "$GREP_MODE" in
    ggrep) ($LINT_TIMEOUT ggrep -oP "$pattern" "$file" 2>/dev/null || true) | wc -l | tr -d ' ' ;;
    grep)  ($LINT_TIMEOUT grep -oP "$pattern" "$file" 2>/dev/null || true) | wc -l | tr -d ' ' ;;
    perl)  LINT_PAT="$pattern" $LINT_TIMEOUT perl -CSD -ne 'BEGIN{$p=$ENV{LINT_PAT}; utf8::decode($p); $re=qr/$p/} $c++ while /$re/g; END{print $c//0}' "$file" 2>/dev/null || echo 0 ;;
  esac
}

# Returns 0 if pattern found, 1 if not
regex_quiet() {
  local pattern="$1" file="$2"
  case "$GREP_MODE" in
    ggrep) $LINT_TIMEOUT ggrep -Pq "$pattern" "$file" 2>/dev/null ;;
    grep)  $LINT_TIMEOUT grep -Pq "$pattern" "$file" 2>/dev/null ;;
    perl)  LINT_PAT="$pattern" $LINT_TIMEOUT perl -CSD -ne 'BEGIN{$f=1; $p=$ENV{LINT_PAT}; utf8::decode($p); $re=qr/$p/} $f=0 if /$re/; END{exit $f}' "$file" 2>/dev/null ;;
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
  # policy/test-corpus/ and policy/test-pipeline/ are deliberately full of
  # violations — they are fixture sets for their own runners, which pass the
  # directory explicitly. Left in scope, a repo-wide lint would report the
  # fixtures as findings.
  # ...unless the fixture dir itself is the target, which is how a runner calls it.
  local filter='cat'
  case "$dir" in
    *policy/test-corpus*|*policy/test-pipeline*) ;;
    *) filter='grep -vE /policy/(test-corpus|test-pipeline)/' ;;
  esac
  find "$dir" -name "$name_pattern" -type f 2>/dev/null | $filter | sort
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
      if [[ "$line" =~ ^\|\ *([A-Z][A-Z._0-9]+)\ *\|\ *([a-z0-9._]+)\ *\|\ *([^|]+)\ *\| ]]; then
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

# Returns effective threshold after profile params override (respects locked)
get_effective_threshold() {
  local rule_id="$1" locked="$2" thresh_param="$3" default_thresh="$4"
  [[ -z "$thresh_param" ]] && { echo "$default_thresh"; return; }
  for override in "${PROFILE_OVERRIDES[@]}"; do
    IFS='|' read -r oid field value <<< "$override"
    if [[ "$oid" == "$rule_id" && "$field" == "params.$thresh_param" ]]; then
      if [[ "$locked" == "true" ]]; then
        $QUIET || echo -e "  ${DIM}(profile params.$thresh_param override ignored: $rule_id is locked)${NC}" >&2
        echo "$default_thresh"; return
      fi
      echo "$value"; return
    fi
  done
  echo "$default_thresh"
}

# ─── Frontmatter Parser ─────────────────────────────────────────────────────
# Single-pass parser using pure bash (no grep/sed in pipelines).
# Sets: RULE_ID, RULE_SEVERITY, RULE_LOCKED, RULE_LAYER, RULE_CHECK_KIND,
#       RULE_CONSTRAINT_TYPE, RULE_AUTOFIX, RULE_LINT_TARGETS,
#       PATTERNS[] (entries: "pattern\tmode\tthreshold"),
#       FIX_PATTERNS[] (entries: "find\treplace")
RULE_ID="" RULE_SEVERITY="" RULE_LOCKED="" RULE_LAYER=""
RULE_CHECK_KIND="" RULE_CONSTRAINT_TYPE="" RULE_AUTOFIX=""
RULE_LINT_TARGETS="" RULE_COVERAGE_NOTE=""
declare -a PATTERNS=()
declare -a FIX_PATTERNS=()

parse_rule() {
  local file="$1"

  RULE_ID="" RULE_SEVERITY="" RULE_LOCKED="" RULE_LAYER=""
  RULE_CHECK_KIND="" RULE_CONSTRAINT_TYPE="" RULE_AUTOFIX=""
  RULE_LINT_TARGETS="" RULE_COVERAGE_NOTE=""
  PATTERNS=()
  FIX_PATTERNS=()

  # Extract frontmatter (between --- markers)
  local frontmatter
  frontmatter=$(awk '/^---$/{n++;next} n==1{print}' "$file")

  # Single-pass: parse simple fields, lint_patterns and fix_patterns blocks
  local in_lp=false in_fp=false
  local cur_pat="" cur_mode="match" cur_thresh="" cur_thresh_param=""
  local cur_find="" cur_replace=""

  while IFS= read -r line; do
    # A YAML comment inside a pattern block must not terminate it. Rule cards
    # are documentation-heavy by design, and silently dropping every pattern
    # after the first comment disables rules with no diagnostic at all.
    if { $in_lp || $in_fp; } && [[ "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    # ── lint_patterns block ──
    if $in_lp; then
      if [[ "$line" =~ ^\ \ -\ pattern:\ \"(.+)\" ]]; then
        [[ -n "$cur_pat" ]] && PATTERNS+=("${cur_pat}"$'\t'"${cur_mode}"$'\t'"${cur_thresh}"$'\t'"${cur_thresh_param}")
        cur_pat="${BASH_REMATCH[1]}"; cur_mode="match"; cur_thresh=""; cur_thresh_param=""
      elif [[ "$line" =~ ^\ \ \ \ mode:\ (.+) ]]; then
        cur_mode="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^\ \ \ \ threshold:\ ([0-9]+) ]]; then
        cur_thresh="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^\ \ \ \ threshold_param:\ ([a-z0-9_]+) ]]; then
        cur_thresh_param="${BASH_REMATCH[1]}"
      else
        [[ -n "$cur_pat" ]] && PATTERNS+=("${cur_pat}"$'\t'"${cur_mode}"$'\t'"${cur_thresh}"$'\t'"${cur_thresh_param}")
        cur_pat=""; in_lp=false
      fi
    fi
    # ── fix_patterns block ──
    if $in_fp; then
      if [[ "$line" =~ ^\ \ -\ find:\ \"(.+)\" ]]; then
        [[ -n "$cur_find" ]] && FIX_PATTERNS+=("${cur_find}"$'\t'"${cur_replace}")
        cur_find="${BASH_REMATCH[1]}"; cur_replace=""
      elif [[ "$line" =~ ^\ \ \ \ replace:\ \"(.*)\" ]]; then
        cur_replace="${BASH_REMATCH[1]}"
      else
        [[ -n "$cur_find" ]] && FIX_PATTERNS+=("${cur_find}"$'\t'"${cur_replace}")
        cur_find=""; in_fp=false
      fi
    fi
    if ! $in_lp && ! $in_fp; then
      case "$line" in
        "id: "*)               RULE_ID="${line#id: }" ;;
        "severity: "*)         RULE_SEVERITY="${line#severity: }" ;;
        "locked: "*)           RULE_LOCKED="${line#locked: }" ;;
        "layer: "*)            RULE_LAYER="${line#layer: }" ;;
        "check_kind: "*)       RULE_CHECK_KIND="${line#check_kind: }" ;;
        "constraint_type: "*)  RULE_CONSTRAINT_TYPE="${line#constraint_type: }" ;;
        "autofix: "*)          RULE_AUTOFIX="${line#autofix: }" ;;
        "lint_targets: "*)     local t="${line#lint_targets: }"; RULE_LINT_TARGETS="${t//\"/}" ;;
        "coverage_note: "*)    local c="${line#coverage_note: }"; RULE_COVERAGE_NOTE="${c//\"/}" ;;
        "lint_patterns:")      in_lp=true ;;
        "fix_patterns:")       in_fp=true ;;
      esac
    fi
  done <<< "$frontmatter"

  # Save last entries if blocks were at end of frontmatter
  [[ -n "$cur_pat" ]] && PATTERNS+=("${cur_pat}"$'\t'"${cur_mode}"$'\t'"${cur_thresh}"$'\t'"${cur_thresh_param}")
  [[ -n "$cur_find" ]] && FIX_PATTERNS+=("${cur_find}"$'\t'"${cur_replace}")
  return 0
}

# ─── Builtin: PROSE.ADHOC_COMPOUND_MODIFIER ─────────────────────────────────
# Hyphenated compound modifiers built from a productive suffix (-based, -aware,
# -driven, …) are ordinary technical English when the field already uses them.
# What is not ordinary is coining a fresh one and using it once: the reader pays
# to decode `community-shift-aware` and the decoding is never reused.
#
# The discriminator is therefore FREQUENCY, which no per-line regex can express —
# hence a builtin. A term the field uses recurs; a coinage is a hapax. Measured
# over 40 arXiv sources: hapax compounds run 0.16 per 1000 words in 2019–2021
# against 0.48 in 2025–2026.
#
# The allowlist is a floor, not the boundary. Whether a hapax compound is a real
# term of the paper's field is a judgment the skill makes; this check only
# surfaces the candidates.
# Suffix tiers. `-based` is excluded by default because it is a CONSTRUCTION,
# not a coinage: `X-based` is the compressed form of "based on X", so it composes
# freely and both eras use it. Measured on the 40-paper corpus: 22 of 25 hapax
# compounds in 2019-2021 sources end in -based, against 43 of 83 in 2025-2026.
# Including it dilutes the signal from 16x to 4x. Set LINT_ADHOC_INCLUDE_BASED=1
# to fold it back in — `concatenation-based` and `euclidean-based` show that
# -based can still be coined awkwardly; it is just not where coinage lives.
ADHOC_SUFFIXES='aware|driven|guided|centric|oriented|enabled|agnostic|informed|preserving|grounded|conditioned|specific|augmented|enhanced'
[[ "${LINT_ADHOC_INCLUDE_BASED:-0}" == "1" ]] && ADHOC_SUFFIXES="based|$ADHOC_SUFFIXES"
ADHOC_ALLOW=' agent-based model-based rule-based gradient-based learning-based sampling-based physics-based energy-based attention-based transformer-based content-based knowledge-based feature-based graph-based template-based search-based simulation-based optimization-based data-driven model-driven event-driven task-specific domain-specific application-specific model-agnostic privacy-preserving structure-preserving '

lint_prose_adhoc_compound_modifier() {
  local severity="$1"
  local -a tex_files=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && tex_files+=("$f")
  done < <(find_target_files "*.tex" "$TARGET_DIR")
  [[ ${#tex_files[@]} -gt 0 ]] || return 0

  local file view hits
  for file in "${tex_files[@]}"; do
    view=$(lint_view "$file")
    # Two passes over the same view: tally every compound, then report only the
    # ones seen exactly once. Line number comes from the single occurrence.
    hits=$(ADHOC_SUF="$ADHOC_SUFFIXES" ADHOC_OK="$ADHOC_ALLOW" perl -CSD -e '
      my $suf = $ENV{ADHOC_SUF}; my %ok = map { $_ => 1 } split " ", $ENV{ADHOC_OK};
      my (%n, %line, %shown, %attr, %exempt);
      # Function words and finite verbs that follow a compound in PREDICATIVE
      # position ("the estimator is model-agnostic, and ..."). A compound only
      # costs the reader mid-parse when it modifies a following noun.
      my $FUNC = qr/^(is|are|was|were|be|been|and|or|but|which|that|to|in|on|of|for|with|when|if|as|at|by|from|than|so|then|thus|hence|we|it|this|these|those|there|however|therefore)$/i;
      open(my $fh, "<:raw", $ARGV[0]) or exit 0;
      my @L = <$fh>; close $fh;
      for my $i (0 .. $#L) {
        my $l = $L[$i];
        $l =~ s/\$[^\$]*\$//g;
        while ($l =~ /\b([A-Za-z]{3,}(?:-[A-Za-z]{2,})*-(?:$suf))\b([^\n]{0,40})/gi) {
          my ($raw, $after) = ($1, $2 // "");
          my $w = lc $raw;
          $n{$w}++;
          $line{$w} //= $i + 1;
          $shown{$w} //= $raw;
          # Attributive: the next token is a word, and not a function word or copula.
          if ($after =~ /^\s+([A-Za-z][A-Za-z-]*)/) { $attr{$w}++ unless $1 =~ $FUNC }
          # Exempt: the author defines an acronym for it — that is an explicit
          # naming act, which is the path PROSE.INVENTED_CONCEPT_LABEL asks for.
          $exempt{$w} = 1 if $after =~ /^\s*\(\s*[A-Z]{2,}\s*\)/;
          # Exempt: every segment capitalised (Community-Shift-Aware) is a naming
          # convention. Sentence-initial capitalisation only raises the first
          # segment, so this does not swallow it.
          my @seg = split /-/, $raw;
          $exempt{$w} = 1 if @seg > 1 && !grep { !/^[A-Z]/ } @seg;
        }
      }
      for my $w (sort { $line{$a} <=> $line{$b} } keys %n) {
        next if $n{$w} != 1 || $ok{$w} || $exempt{$w};
        next unless $attr{$w};                      # attributive position only
        my $parts = () = $w =~ /-/g;
        my $risk = $parts >= 2 ? "\thigh" : "\t";
        print "$line{$w}\t$shown{$w}$risk\n";
      }
    ' "$view" 2>/dev/null || true)
    [[ -n "$hits" ]] || continue
    while IFS=$'\t' read -r ln word risk; do
      [[ -n "$ln" ]] || continue
      # A multi-part left element (community-shift-aware) is the highest-risk
      # shape and is tagged so the author can triage; it is not a separate rule.
      local tag=""; [[ "$risk" == "high" ]] && tag=" [multi-part left element]"
      report_finding "$severity" "$RULE_ID" "${file}:${ln}: coined once, attributive, never reused — ${word}${tag}"
    done <<< "$hits"
  done
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

# ─── Built-in Rule: CITE.VERIFY_VIA_API ────────────────────────────────────
# Hard-gate citation hygiene:
#   1) unresolved [CITATION NEEDED] marker is forbidden at lint time
#   2) obvious placeholder/hallucination patterns in .bib are forbidden
#   3) every BibTeX entry must include doi/url/eprint for API traceability
lint_cite_verify_via_api() {
  local severity="$1"

  local -a bib_files=() tex_files=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && bib_files+=("$f")
  done < <(find_target_files "*.bib" "$TARGET_DIR")

  while IFS= read -r f; do
    [[ -n "$f" ]] && tex_files+=("$f")
  done < <(find_target_files "*.tex" "$TARGET_DIR")

  # 1) unresolved marker check
  local unresolved_pat="\\[CITATION NEEDED\\]"
  for file in "${bib_files[@]}" "${tex_files[@]}"; do
    local matches
    matches=$(regex_match "$unresolved_pat" "$file")
    if [[ -n "$matches" ]]; then
      while IFS= read -r mline; do
        report_finding "$severity" "$RULE_ID" "$mline"
      done <<< "$matches"
    fi
  done

  # 2) placeholder / likely hallucination pattern checks (BibTeX only)
  local -a suspicious_pats=(
    "@\\w+\\{(ref[0-9]+|paper[0-9]+|unknown[0-9]*|todo[0-9]*)[,}]"
    "author\\s*=\\s*\\{[^}]*\\.\\.\\.[^}]*\\}"
    "author\\s*=\\s*\\{[^}]*\\b(et al\\.?|and others)\\b[^}]*\\}"
    "title\\s*=\\s*\\{[^}]*\\b([Tt][Oo][Dd][Oo]|[Tt][Bb][Dd])\\b[^}]*\\}"
    "title\\s*=\\s*\\{[^}]*\\.\\.\\.[^}]*\\}"
  )
  for raw_pat in "${suspicious_pats[@]}"; do
    local pat
    pat=$(yaml_unescape "$raw_pat")
    for file in "${bib_files[@]}"; do
      local matches
      matches=$(regex_match "$pat" "$file")
      if [[ -n "$matches" ]]; then
        while IFS= read -r mline; do
          report_finding "$severity" "$RULE_ID" "$mline"
        done <<< "$matches"
      fi
    done
  done

  # 3) each entry requires at least one verifiable identifier
  for file in "${bib_files[@]}"; do
    while IFS= read -r detail; do
      [[ -n "$detail" ]] || continue
      report_finding "$severity" "$RULE_ID" "$detail"
    done < <(
      awk -v file="$file" '
        function flush_entry() {
          if (in_entry && has_identifier == 0) {
            key_out = (entry_key == "" ? "<unknown>" : entry_key)
            printf "%s:%d: entry '\''%s'\'' missing doi/url/eprint for API verification\n", file, entry_line, key_out
          }
        }

        BEGIN {
          in_entry = 0
          has_identifier = 0
          entry_key = ""
          entry_line = 0
        }

        /^[[:space:]]*@/ {
          flush_entry()

          in_entry = 1
          has_identifier = 0
          entry_line = NR
          entry_key = $0
          sub(/^[[:space:]]*@[^{]+[{]/, "", entry_key)
          sub(/[[:space:]]*,[[:space:]]*$/, "", entry_key)
          sub(/,.*/, "", entry_key)
        }

        {
          if (in_entry) {
            lower = tolower($0)
            if (lower ~ /^[[:space:]]*(doi|url|eprint)[[:space:]]*=/) {
              has_identifier = 1
            }
          }
        }

        /^[[:space:]]*}[[:space:]]*,?[[:space:]]*$/ {
          flush_entry()
          in_entry = 0
          has_identifier = 0
          entry_key = ""
          entry_line = 0
        }

        END {
          flush_entry()
        }
      ' "$file"
    )
  done
}

# ─── Built-in Rule: PROSE.NO_INTERNAL_PROVENANCE ───────────────────────────
# Development artifacts that reached a compiled PDF (paths, schema identifiers,
# internal fixture names, revision narrative). Implemented as a builtin rather
# than lint_patterns because the exclusion list is the difference between a
# guardrail people keep and one they switch off: LaTeX source plumbing
# (\includegraphics, \input), reference keys, and the required artifact \url
# must be stripped BEFORE matching, which the generic pattern engine cannot do.
#
# Scope is every .tex line, not body prose: in the sweep that motivated this
# check, 8 of 11 leaks sat in captions, notation tables and appendices.
#
# Matching goes through the same GREP_MODE dispatch as regex_match/regex_count:
# BSD grep has no -P, so a bare `grep -P` here would silently match nothing.

# ─── Builtin: PROSE.SEMICOLON_RESTRICTION ───────────────────────────────────
# A plain pattern would fire inside inline math: `p(y \mid x; \theta)` is
# ordinary ML notation, and measured across the reference corpora 7-16% of all
# semicolons sit inside `$...$`. `\;` is a thin-space macro, not punctuation.
# Both have to be stripped before anything is judged, hence a builtin.
# List items ending in a semicolon are a list separator convention, not a
# mid-paragraph join, so list environments are skipped entirely.
lint_prose_semicolon_restriction() {
  local severity="$1"
  local -a tex_files=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && tex_files+=("$f")
  done < <(find_target_files "*.tex" "$TARGET_DIR")
  [[ ${#tex_files[@]} -gt 0 ]] || return 0

  local file view hits stripped
  for file in "${tex_files[@]}"; do
    view=$(lint_view "$file")
    hits=$(perl -CSD -e '
      my ($math, $list, $verb) = (0, 0, 0);
      my $depth_list = 0; my $in_verb = 0;
      open(my $fh, "<:encoding(UTF-8)", $ARGV[0]) or exit 0;
      my @out;
      while (my $l = <$fh>) {
        my $ln = $.;
        if ($l =~ /\\begin\{(algorithm|algorithmic|lstlisting|verbatim|tikzpicture|equation|align)\*?\}/) { $in_verb++ }
        if ($l =~ /\\end\{(algorithm|algorithmic|lstlisting|verbatim|tikzpicture|equation|align)\*?\}/) { $in_verb-- if $in_verb > 0; next }
        if ($in_verb) { $verb += ($l =~ tr/;//); next }
        if ($l =~ /\\begin\{(itemize|enumerate|description)\}/) { $depth_list++ }
        if ($l =~ /\\end\{(itemize|enumerate|description)\}/)   { $depth_list-- if $depth_list > 0; next }
        if ($depth_list) { $list += ($l =~ tr/;//); next }

        my $c = $l;
        my $thin = ($c =~ s/\\;/ /g) || 0;         # thin-space macro: literal backslash-semicolon,
        #   NOT /\;/ — in a regex that is just `;` and silently deletes every one.
        my $before = ($c =~ tr/;//);
        $c =~ s/\$[^\$]*\$/ MATH /g;                 # inline math
        my $after = ($c =~ tr/;//);
        $math += $before - $after;
        next unless $after;
        while ($c =~ /(\S*\s?\S*);(\s?\S*\s?\S*)/g) {
          my $ctx = "$1;$2"; $ctx =~ s/\s+/ /g; $ctx =~ s/^\s+|\s+$//g;
          push @out, "HIT\t$ln\t$ctx";
        }
      }
      close $fh;
      print "$_\n" for @out;
      print "SKIP\t$math\t$list\t$verb\n";
    ' "$view" 2>/dev/null || true)
    [[ -n "$hits" ]] || continue
    while IFS=$'\t' read -r kind a b c; do
      case "$kind" in
        HIT)
          report_finding "$severity" "$RULE_ID" "${file}:${a}: semicolon joins two clauses — split, or comma + conjunction when the four tier-1 criteria hold (see card) — ${b}"
          ;;
        SKIP)
          # Silence is not cleanliness: say what was stripped, so the author can
          # tell "checked and clean" from "never looked".
          if (( a > 0 || b > 0 || c > 0 )); then
            $QUIET || echo -e "    ${DIM}cleared in ${file##*/}: ${a} in math, ${b} in list items, ${c} in verbatim/algorithm${NC}"
          fi
          ;;
      esac
    done <<< "$hits"
  done
}

# Emits "lineno<TAB>matched-span" so the caller can attach the real filename.
prov_grep_spans() {
  local pat="$1" f="$2"
  case "$GREP_MODE" in
    ggrep) ggrep -noP "$pat" "$f" 2>/dev/null || true ;;
    grep)  grep  -noP "$pat" "$f" 2>/dev/null || true ;;
    perl)  LINT_PAT="$pat" perl -CSD -ne 'BEGIN{$p=$ENV{LINT_PAT}; utf8::decode($p); $re=qr/$p/} while (/$re/g) { print "$.:$&\n" }' "$f" 2>/dev/null || true ;;
  esac
}

lint_prose_no_internal_provenance() {
  local severity="$1"

  local -a tex_files=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && tex_files+=("$f")
  done < <(find_target_files "*.tex" "$TARGET_DIR")

  # P1-P5 per the rule card's Check table. P5 is medium confidence (revision
  # narrative needs adjudication); the rest are high or near-certain.
  local -a pats=(
    '\\path\{[^}]*\}'
    '(experiments|results|scripts|notebooks)/[A-Za-z0-9_./-]+'
    '\.(csv|py|jsonl|json|sh|ipynb|log|pkl|npz|yaml)\b'
    '\\texttt\{[^}]*[a-z0-9]+\\?_[a-z0-9]+[^}]*\}'
    '(retracted|supersed(es|ed)|legacy [a-z]+|no longer (used|the original version)|old (bound|formula|version)|previously we|earlier draft|to avoid collision with)'
  )
  local -a pids=(P1 P2 P3 P4 P5)

  local scrub_file
  scrub_file=$(mktemp)

  for file in "${tex_files[@]}"; do
    # Strip, in order: comments; source plumbing and reference keys; the
    # artifact URL; EXP-mandated status disclosures (those two rules win).
    # Blanking rather than deleting keeps line numbers aligned with the
    # original file, so reported file:line points where the author must look.
    sed -E \
      -e 's/(^|[^\\])%.*$/\1/' \
      -e 's/\\(includegraphics|input|include|bibliography|usepackage|documentclass)(\[[^]]*\])?\{[^}]*\}//g' \
      -e 's/\\(label|ref|Cref|cref|eqref|autoref|cite[a-z]*)\{[^}]*\}//g' \
      -e 's/\\(url|href)\{[^}]*\}//g' \
      -e 's/.*\[(FABRICATED|SIMULATED|PROJECTED|NOT EXECUTED)\].*//' \
      -e 's/.*(SIMULATED|FABRICATED|PROJECTED) RESULTS?.*//' \
      "$file" > "$scrub_file" 2>/dev/null

    local i
    for i in "${!pats[@]}"; do
      local hits
      hits=$(prov_grep_spans "${pats[$i]}" "$scrub_file")
      [[ -n "$hits" ]] || continue
      while IFS= read -r hline; do
        [[ -n "$hline" ]] || continue
        report_finding "$severity" "$RULE_ID" "${file}:${hline%%:*}: [${pids[$i]}] ${hline#*:}"
      done <<< "$hits"
    done
  done

  rm -f "$scrub_file"
}

# ─── Lint Single Rule ───────────────────────────────────────────────────────
lint_rule() {
  local severity="$1"
  local findings=0

  # Categorize patterns by mode
  local -a m_pats=() c_pats=() c_threshs=() c_thresh_params=() n_pats=()
  for entry in "${PATTERNS[@]}"; do
    IFS=$'\t' read -r pat mode thresh thresh_param <<< "$entry"
    case "$mode" in
      match)    m_pats+=("$pat") ;;
      count)    c_pats+=("$pat"); c_threshs+=("$thresh"); c_thresh_params+=("$thresh_param") ;;
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
      local matches view
      view=$(lint_view "$file")
      matches=$(regex_match "$pat" "$view")
      # A timed-out pattern is a defect in the pattern, reported as an error.
      # Falling through would record zero matches, which reads as a clean file.
      if [[ "$matches" == *"__LINT_TIMEOUT__"* ]]; then
        ((PATTERN_TIMEOUTS++)) || true
        report_finding "error" "$RULE_ID" "${file}: pattern timed out after ${LINT_TIMEOUT##* }s — catastrophic backtracking, this file was NOT checked against it"
        matches=$(printf '%s\n' "$matches" | grep -v '__LINT_TIMEOUT__' || true)
      fi
      # regex_match stamps the view path; report the path the author edits.
      [[ "$view" == "$file" ]] || matches="${matches//$view/$file}"
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
    local thresh_param="${c_thresh_params[$i]:-}"
    local effective_thresh
    effective_thresh=$(get_effective_threshold "$RULE_ID" "$RULE_LOCKED" "$thresh_param" "$thresh")
    local pat
    pat=$(yaml_unescape "$raw_pat")
    for file in "${tfiles[@]}"; do
      local cnt
      cnt=$(regex_count "$pat" "$(lint_view "$file")")
      if (( cnt > effective_thresh )); then
        ((findings++)) || true
        report_finding "$severity" "$RULE_ID" "${file}: count=${cnt} > threshold=${effective_thresh}"
      fi
    done
  done

  # ── Negative mode: pattern NOT found in scoped files = violation ──
  if [[ ${#n_pats[@]} -gt 0 ]]; then
    local -a scope=()
    if [[ ${#m_pats[@]} -gt 0 ]]; then
      # When match patterns exist, scope negative checks to files with violations only.
      # If no files matched, skip negative checks (no violations = no scope).
      if [[ ${#flagged_files[@]} -eq 0 ]]; then
        # No match hits → nothing to scope against → skip negative checks
        true  # fall through with empty scope
      else
        while IFS= read -r f; do
          scope+=("$f")
        done < <(printf '%s\n' "${flagged_files[@]}" | sort -u)
      fi
    else
      # No match patterns defined → check all target files
      scope=("${tfiles[@]}")
    fi

    for raw_pat in "${n_pats[@]}"; do
      local pat
      pat=$(yaml_unescape "$raw_pat")
      for file in "${scope[@]}"; do
        if ! regex_quiet "$pat" "$(lint_view "$file")"; then
          ((findings++)) || true
          report_finding "$severity" "$RULE_ID" "${file}: MISSING required pattern"
        fi
      done
    done
  fi
}

# ─── Fix Rule (autofix=safe) ─────────────────────────────────────────────
# Applies fix_patterns (find→replace) using Python re.sub with lambda
# for truly literal replacement (no interpolation of $, \, @ etc.).
# Only called when --fix is active and rule has autofix=safe.
fix_rule() {
  local -a tfiles=()
  while IFS= read -r f; do
    [[ -n "$f" ]] && tfiles+=("$f")
  done < <(find_target_files "$RULE_LINT_TARGETS" "$TARGET_DIR")

  if [[ ${#tfiles[@]} -eq 0 ]]; then return; fi
  if [[ ${#FIX_PATTERNS[@]} -eq 0 ]]; then
    $QUIET || echo -e "    ${DIM}(no fix_patterns defined, skipping auto-fix)${NC}"
    return
  fi

  local rule_fixes=0
  for entry in "${FIX_PATTERNS[@]}"; do
    IFS=$'\t' read -r find_pat replace_str <<< "$entry"
    local uf
    uf=$(yaml_unescape "$find_pat")
    local ur
    ur=$(yaml_unescape "$replace_str")

    for file in "${tfiles[@]}"; do
      # Count matches before fixing
      local before_count
      before_count=$(regex_count "$uf" "$file")
      if (( before_count > 0 )); then
        # Apply fix using Python re.sub with lambda (truly literal replacement)
        LINT_FIND="$uf" LINT_REPLACE="$ur" python3 -c "
import re, os
find = os.environ['LINT_FIND']
repl = os.environ['LINT_REPLACE']
path = '$file'
with open(path, 'r') as f:
    content = f.read()
content = re.sub(find, lambda m: repl, content)
with open(path, 'w') as f:
    f.write(content)
" 2>/dev/null
        ((rule_fixes += before_count)) || true
        $QUIET || printf '    \033[0;32mFIXED\033[0m %s: %d replacement(s) [%s → %s]\n' \
          "$file" "$before_count" "$find_pat" "${replace_str:-<delete>}"
      fi
    done
  done

  ((TOTAL_FIXES += rule_fixes)) || true
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

  # Apply filters
  [[ -z "$FILTER_RULE" || "$RULE_ID" == "$FILTER_RULE" ]] || continue
  [[ -z "$FILTER_LAYER" || "$RULE_LAYER" == "$FILTER_LAYER" ]] || continue
  [[ -z "$FILTER_CONSTRAINT_TYPE" || "$RULE_CONSTRAINT_TYPE" == "$FILTER_CONSTRAINT_TYPE" ]] || continue
  [[ -z "$FILTER_AUTOFIX" || "$RULE_AUTOFIX" == "$FILTER_AUTOFIX" ]] || continue

  # --fix mode: skip rules that aren't autofix=safe
  if $FIX_MODE && [[ "$RULE_AUTOFIX" != "safe" ]]; then continue; fi

  # Apply profile severity override
  local_severity=$(get_effective_severity "$RULE_ID" "$RULE_LOCKED" "$RULE_SEVERITY")

  # Built-in rule checks (non-regex) with hard enforcement
  if [[ "$RULE_ID" == "CITE.VERIFY_VIA_API" ]]; then
    $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (${local_severity}) builtin checks → *.bib, *.tex"

    ((RULES_CHECKED++)) || true

    prev_e=$TOTAL_ERRORS
    prev_w=$TOTAL_WARNINGS

    lint_cite_verify_via_api "$local_severity"

    if (( TOTAL_ERRORS == prev_e && TOTAL_WARNINGS == prev_w )); then
      ((RULES_PASSED++)) || true
      $QUIET || echo -e "    ${GREEN}PASS${NC}"
    fi
    $QUIET || echo ""
    continue
  fi

  if [[ "$RULE_ID" == "PROSE.ADHOC_COMPOUND_MODIFIER" ]]; then
    $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (${local_severity}) builtin hapax-compound → *.tex"
    ((RULES_CHECKED++)) || true
    prev_e=$TOTAL_ERRORS
    prev_w=$TOTAL_WARNINGS
    lint_prose_adhoc_compound_modifier "$local_severity"
    if (( TOTAL_ERRORS == prev_e && TOTAL_WARNINGS == prev_w )); then
      ((RULES_PASSED++)) || true
      $QUIET || echo -e "    ${GREEN}PASS${NC}"
    fi
    $QUIET || echo ""
    continue
  fi

  if [[ "$RULE_ID" == "PROSE.SEMICOLON_RESTRICTION" ]]; then
    $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (${local_severity}) builtin semicolon → *.tex"
    ((RULES_CHECKED++)) || true
    prev_e=$TOTAL_ERRORS
    prev_w=$TOTAL_WARNINGS
    lint_prose_semicolon_restriction "$local_severity"
    if (( TOTAL_ERRORS == prev_e && TOTAL_WARNINGS == prev_w )); then
      ((RULES_PASSED++)) || true
      $QUIET || echo -e "    ${GREEN}PASS${NC}"
    fi
    $QUIET || echo ""
    continue
  fi

  if [[ "$RULE_ID" == "PROSE.NO_INTERNAL_PROVENANCE" ]]; then
    $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (${local_severity}) builtin P1-P5 → *.tex"

    ((RULES_CHECKED++)) || true

    prev_e=$TOTAL_ERRORS
    prev_w=$TOTAL_WARNINGS

    lint_prose_no_internal_provenance "$local_severity"

    if (( TOTAL_ERRORS == prev_e && TOTAL_WARNINGS == prev_w )); then
      ((RULES_PASSED++)) || true
      $QUIET || echo -e "    ${GREEN}PASS${NC}"
    fi
    $QUIET || echo ""
    continue
  fi

  # Run whatever machine patterns exist, regardless of check_kind. The old gate
  # (`check_kind == regex`) silently skipped patterns on llm_* cards — a card
  # adding a regex LOCATOR to a judgment rule (PROSE.OVER_DEFENSIVE's sentence
  # layer) parsed clean, validated clean, and never executed. Presence of
  # patterns is the intent signal; 9c already guarantees pattern-bearing cards
  # say enforcement: lint_script.
  [[ ${#PATTERNS[@]} -gt 0 || ($FIX_MODE && ${#FIX_PATTERNS[@]} -gt 0) ]] || continue
  [[ -n "$RULE_LINT_TARGETS" ]] || continue

  if $FIX_MODE; then
    $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (autofix: ${RULE_AUTOFIX}) ${#FIX_PATTERNS[@]} fix pattern(s) → ${RULE_LINT_TARGETS}"
  else
    $QUIET || echo -e "  ${CYAN}[$RULE_ID]${NC} (${local_severity}) ${#PATTERNS[@]} pattern(s) → ${RULE_LINT_TARGETS}"
  fi
  # A rule whose Requirement covers more forms than its patterns do must say so
  # HERE, not only in its Check section. An executor working from lint output
  # never opens the card, so an undeclared narrowing makes the denominator look
  # complete. Measured: one manuscript reported 15 contrast constructions where
  # 38 existed, because the largest class was deliberately left to judgment and
  # the deliberateness was invisible at the point of use.
  if [[ -n "$RULE_COVERAGE_NOTE" ]]; then
    $QUIET || echo -e "    ${DIM}partial coverage — ${RULE_COVERAGE_NOTE}${NC}"
  fi

  # Connective-distribution table (report-only, never a finding). The
  # punctuation-clearing rules (em-dash, semicolon, mid-sentence colon) push the
  # relation the punctuation carried into the lexical layer, and `therefore` is
  # the menu's safest answer — so monoculture accumulates ACROSS passes, one or
  # two per pass, each locally reasonable. Measured: 22 of 26 formal connectives
  # in one manuscript were `therefore` after three clearing passes, with every
  # per-instance judgment individually defensible. A distribution that passes
  # every per-instance check can still fail as a whole; only a whole-document
  # count makes it visible. No threshold: an all-entailment paper is
  # legitimately therefore-heavy — the table exists so the recheck question
  # ("is this really all the same kind of causality?") is asked against real
  # numbers instead of an impression.
  if [[ "$RULE_ID" == "PROSE.CAUSAL_CONNECTIVE" ]] && ! $FIX_MODE; then
    _cc_views=""
    while IFS= read -r f; do
      [[ -n "$f" ]] && _cc_views="$_cc_views $(lint_view "$f")"
    done < <(find_target_files "*.tex" "$TARGET_DIR")
    if [[ -n "${_cc_views// /}" ]]; then
      _cc_counts=$(cat $_cc_views 2>/dev/null | perl -ne '
        $c{lc $1}++ while /\b(therefore|thus|hence|consequently|accordingly)\b/gi;
        END { printf "therefore %d · thus %d · hence %d · consequently %d · accordingly %d",
              map { $c{$_}//0 } qw(therefore thus hence consequently accordingly) }')
      $QUIET || echo -e "    ${DIM}connective distribution (report-only) — ${_cc_counts}${NC}"
    fi
  fi

  ((RULES_CHECKED++)) || true

  if $FIX_MODE; then
    fix_rule
    # Post-fix verification: re-lint the same rule to confirm violations are gone
    if [[ ${#PATTERNS[@]} -gt 0 ]]; then
      prev_e=$TOTAL_ERRORS
      prev_w=$TOTAL_WARNINGS
      lint_rule "$local_severity"
      remaining=$(( (TOTAL_ERRORS - prev_e) + (TOTAL_WARNINGS - prev_w) ))
      if (( remaining > 0 )); then
        $QUIET || echo -e "    ${YELLOW}VERIFY${NC} ${remaining} violation(s) remain after auto-fix"
      else
        ((RULES_PASSED++)) || true
        $QUIET || echo -e "    ${GREEN}VERIFIED${NC} clean after auto-fix"
      fi
    else
      ((RULES_PASSED++)) || true
    fi
  else
    prev_e=$TOTAL_ERRORS
    prev_w=$TOTAL_WARNINGS

    lint_rule "$local_severity"

    if (( TOTAL_ERRORS == prev_e && TOTAL_WARNINGS == prev_w )); then
      ((RULES_PASSED++)) || true
      $QUIET || echo -e "    ${GREEN}PASS${NC}"
    fi
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
if $FIX_MODE; then
  echo -e "  Fixes applied:  ${GREEN}${TOTAL_FIXES}${NC}"
fi
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
