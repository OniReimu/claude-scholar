#!/usr/bin/env bash
# policy/test-corpus.sh — rule-accuracy regression over the annotated corpus.
#
# Complements policy/test-lint.sh: that one tests lint mechanics (flags, fix
# emission, exit codes) and never asks whether a rule fires on the right
# sentence. This one asks only that.
#
# Usage: bash policy/test-corpus.sh [--verbose] [--engine ggrep|grep|perl]
# Exit:  0 = every case matched its @EXPECT (XFAIL cases excluded)
#        1 = at least one recall miss or false positive
#        2 = corpus or runner error
#
# Case format is documented in policy/test-corpus/README.md.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS_DIR="$SCRIPT_DIR/test-corpus"
LINT="$SCRIPT_DIR/lint.sh"
VERBOSE=false
ENGINE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) VERBOSE=true; shift ;;
    # The engines are not interchangeable: macOS resolves to perl and Linux CI
    # to GNU grep, so a defect in one is invisible to the other half of the
    # project. Run the corpus under each to compare.
    --engine)  ENGINE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$ENGINE" ]]; then
  export LINT_ENGINE="$ENGINE"
  case "$ENGINE" in
    ggrep|grep) echo "test" | $ENGINE -P "test" &>/dev/null || { echo "SKIP: $ENGINE has no -P on this host"; exit 0; } ;;
    perl) command -v perl &>/dev/null || { echo "SKIP: no perl on this host"; exit 0; } ;;
    *) echo "Unknown engine: $ENGINE" >&2; exit 2 ;;
  esac
fi

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

[[ -d "$CORPUS_DIR" ]] || { echo "ERROR: corpus dir not found: $CORPUS_DIR" >&2; exit 2; }

# File-scoped rules count over a whole file, so they cannot be attributed to a
# case. They are covered by policy/test-lint.sh instead.
FILE_SCOPED="PAPER.SECTION_HEADINGS_MAX_6 TABLE.BOOKTABS_FORMAT SUBMIT.SECTION_NUMBERING_CONSISTENCY"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ─── 1. Collect violations: RULE_ID <TAB> file <TAB> line ────────────────────
# The corpus contains error-severity fixtures on purpose, so lint exits 1 here
# on a healthy run. Its exit code carries no information for this suite; the
# findings do.
{ bash "$LINT" "$CORPUS_DIR" 2>&1 || true; } \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | awk -v skip="$FILE_SCOPED" '
      BEGIN { n = split(skip, s, " "); for (i=1;i<=n;i++) drop[s[i]] = 1 }
      /^  \[[A-Z]/ { rule = $1; gsub(/[][]/, "", rule); next }
      /^    (WARN|ERROR)/ {
        if (rule in drop) next
        # field 2 is <path>:<line>:… — path contains no spaces or colons by
        # construction. Do NOT count segments from the right: GNU grep emits no
        # space after the line number, so field 2 absorbs the start of the
        # matched text, and a match containing a colon (`\\Cref{tab:ablation}`)
        # shifts every right-anchored index. Count from the left instead.
        loc = $2
        split(loc, p, ":")
        path = p[1]
        line = p[2]
        n2 = split(path, q, "/")
        print rule "\t" q[n2] "\t" line
      }
    ' | sort -u > "$WORK/violations.tsv"

# ─── 2. Collect cases: file <TAB> start <TAB> end <TAB> label <TAB> expect <TAB> xfail
: > "$WORK/cases.tsv"
shopt -s nullglob
corpus_files=("$CORPUS_DIR"/*.tex)
(( ${#corpus_files[@]} )) || { echo "ERROR: corpus contains no .tex fixtures" >&2; exit 2; }

for f in "${corpus_files[@]}"; do
  awk -v fname="$(basename "$f")" '
    /^% @CASE/    { label = $3; start = NR; expect = ""; xfail = ""; open = 1; next }
    /^% @EXPECT/  { sub(/^% @EXPECT[ \t]*/, ""); expect = $0; next }
    /^% @XFAIL/   { sub(/^% @XFAIL[ \t]*/, ""); xfail = $0; next }
    /^% @ENDCASE/ {
      if (!open) { print "PARSE-ERROR " fname " line " NR ": @ENDCASE without @CASE" > "/dev/stderr"; exit 2 }
      gsub(/[ \t]/, "", expect)
      print fname "\t" start "\t" NR "\t" label "\t" expect "\t" xfail
      open = 0; next
    }
    END { if (open) { print "PARSE-ERROR " fname ": unterminated @CASE " label > "/dev/stderr"; exit 2 } }
  ' "$f" >> "$WORK/cases.tsv"
done

(( $(wc -l < "$WORK/cases.tsv") )) || { echo "ERROR: no @CASE blocks parsed" >&2; exit 2; }

# ─── 3. Join and judge ───────────────────────────────────────────────────────
echo -e "${BOLD}Policy Corpus Test${NC} — $(wc -l < "$WORK/cases.tsv" | tr -d ' ') cases, $(wc -l < "$WORK/violations.tsv" | tr -d ' ') attributable findings"
echo ""

awk -F'\t' -v verbose="$VERBOSE" -v R="$RED" -v Y="$YELLOW" -v G="$GREEN" -v D="$DIM" -v N="$NC" '
  NR == FNR { v[$2 "\t" $3, $1] = 1; vfile[$2]; vline[$2 "\t" $3] = vline[$2 "\t" $3] " " $1; next }
  {
    file = $1; start = $2; end = $3; label = $4; expect = $5; xfail = $6
    delete want; nwant = 0
    if (expect != "none" && expect != "") {
      nwant = split(expect, w, ",")
      for (i = 1; i <= nwant; i++) want[w[i]] = 1
    }
    # union of rules fired anywhere inside the case body
    delete got
    for (l = start; l <= end; l++) {
      key = file "\t" l
      if (key in vline) { m = split(vline[key], g, " "); for (i = 1; i <= m; i++) if (g[i] != "") got[g[i]] = 1 }
    }
    miss = ""; fp = ""
    for (r in want) if (!(r in got)) miss = miss " " r
    for (r in got)  if (!(r in want)) fp = fp " " r
    seen[label] = 1
    if (miss == "" && fp == "") {
      pass++
      if (verbose == "true") printf "  %s✓%s %s\n", G, N, label
    } else if (xfail != "") {
      xf++
      printf "  %sXFAIL%s %-46s %s%s%s\n", Y, N, label, D, xfail, N
    } else {
      fail++
      printf "  %s✗%s %s\n", R, N, label
      if (miss != "") printf "      %smissed:%s%s\n", R, N, miss
      if (fp   != "") printf "      %sfalse positive:%s%s\n", Y, N, fp
    }
  }
  END {
    printf "\n  pass %d   fail %d   xfail %d\n", pass, fail, xf
    exit (fail > 0 ? 1 : 0)
  }
' "$WORK/violations.tsv" "$WORK/cases.tsv"
