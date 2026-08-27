#!/usr/bin/env bash
# Pipeline acceptance scorer.
#
# test-corpus.sh scores one rule against labelled snippets. This scores the
# two-stage pipeline (claim-architecture-review -> writing-anti-ai) end to end
# against a fixture whose expected outcome was sealed before the run.
#
# The agent half is manual by design (see <fixture>/PROMPT.md); this script is
# the deterministic half. With no final.tex argument it scores the recorded
# reference run, which is how CI keeps the assertions from rotting.
#
# Usage: ./test-pipeline.sh <fixture> [final.tex]
#        ./test-pipeline.sh --list

set -uo pipefail

# Resolve a caller-supplied final.tex against the ORIGINAL cwd before chdir'ing
# to the script directory. Without this, the ordinary invocation
#   ./policy/test-pipeline.sh <case> policy/test-pipeline/<case>/input.tex
# dies with "missing final text" (exit 2) because the path is re-resolved
# relative to policy/ — which reads as a broken fixture rather than a usage slip.
if [ "${2:-}" ] && [ "${2#/}" = "$2" ]; then
  set -- "${1:-}" "$PWD/$2"
fi
cd "$(dirname "$0")" || exit 1

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; NC=$'\033[0m'
[ -t 1 ] || { RED=; GREEN=; YELLOW=; DIM=; BOLD=; NC=; }

FIXTURE_DIR="test-pipeline"

if [ "${1:-}" = "--list" ]; then
  for d in "$FIXTURE_DIR"/*/; do [ -f "$d/assert.txt" ] && basename "$d"; done
  exit 0
fi

CASE="${1:-}"
if [ -z "$CASE" ]; then
  echo "usage: $0 <fixture> [final.tex]   ($0 --list)" >&2; exit 2
fi
DIR="$FIXTURE_DIR/$CASE"
[ -d "$DIR" ] || { echo "${RED}no such fixture: $CASE${NC}" >&2; exit 2; }

INPUT="$DIR/input.tex"
FINAL="${2:-$DIR/reference-final.tex}"
[ -f "$INPUT" ] || { echo "${RED}missing $INPUT${NC}" >&2; exit 2; }
[ -f "$FINAL" ] || { echo "${RED}missing final text: $FINAL${NC}" >&2; exit 2; }

# Paragraph reflow is legitimate output, so match against a single-line view.
# Keep the raw file too: VERBATIM spans are checked against original wrapping.
flat() { tr '\n' ' ' < "$1" | tr -s ' '; }
FINAL_FLAT="$(flat "$FINAL")"
INPUT_FLAT="$(flat "$INPUT")"

pass=0; fail=0
declare_fail() { fail=$((fail+1)); }

echo
echo "${BOLD}Pipeline Acceptance${NC} — $CASE"
echo "${DIM}  input:  $INPUT"
echo "  final:  $FINAL${NC}"
[ "$FINAL" = "$DIR/reference-final.tex" ] && echo "${DIM}  (scoring the recorded reference run — self-test mode)${NC}"
echo

# ─── assertions ─────────────────────────────────────────────────────────────
while IFS=$'\t' read -r verb pattern label; do
  case "$verb" in ''|'#'*) continue ;; esac
  [ -n "${label:-}" ] || label="(unlabelled)"

  case "$verb" in
    GONE)
      if printf '%s' "$FINAL_FLAT" | grep -qE -- "$pattern"; then
        echo "  ${RED}FAIL${NC} GONE      ${label}"
        echo "       ${DIM}still present: $(printf '%s' "$FINAL_FLAT" | grep -oE -- "$pattern" | head -1)${NC}"
        declare_fail
      else
        echo "  ${GREEN}pass${NC} GONE      ${label}"; pass=$((pass+1))
      fi
      ;;
    KEPT)
      if printf '%s' "$FINAL_FLAT" | grep -qE -- "$pattern"; then
        echo "  ${GREEN}pass${NC} KEPT      ${label}"; pass=$((pass+1))
      else
        echo "  ${RED}FAIL${NC} KEPT      ${label}"
        echo "       ${DIM}missing: /$pattern/${NC}"
        declare_fail
      fi
      ;;
    VERBATIM)
      start="${pattern%%|*}"; end="${pattern##*|}"
      # Pull the span from the input, then require every one of its lines in the
      # final byte-for-byte. Anchors go through the environment, not awk -v:
      # -v runs escape processing, so a LaTeX anchor like \ref{} loses its \r
      # to a carriage return and the end anchor silently never matches.
      span="$(VB_S="$start" VB_E="$end" awk '
        index($0,ENVIRON["VB_S"]){ inside=1 }
        inside{ print }
        inside && index($0,ENVIRON["VB_E"]){ exit }' "$INPUT")"
      if [ -z "$span" ]; then
        echo "  ${RED}FAIL${NC} VERBATIM  ${label}"
        echo "       ${DIM}start anchor not found in input — fixture is broken${NC}"
        declare_fail
      elif ! printf '%s' "$span" | grep -qF -- "$end"; then
        echo "  ${RED}FAIL${NC} VERBATIM  ${label}"
        echo "       ${DIM}end anchor not found after start — fixture is broken${NC}"
        declare_fail
      else
        # Compare on whitespace-normalised text, not line by line. A polish pass
        # may re-wrap a paragraph it did not otherwise touch; VERBATIM is about
        # the wording surviving, not the line breaks. The line-based form passed
        # here only because the reflow happened to leave a matching substring.
        span_flat="$(printf '%s' "$span" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//')"
        missing=""
        printf '%s' "$FINAL_FLAT" | grep -qF -- "$span_flat" || missing="$span_flat"
        if [ -z "$missing" ]; then
          echo "  ${GREEN}pass${NC} VERBATIM  ${label}"; pass=$((pass+1))
        else
          echo "  ${RED}FAIL${NC} VERBATIM  ${label}"
          echo "       ${DIM}span was modified${NC}"
          declare_fail
        fi
      fi
      ;;
    *)
      echo "  ${RED}FAIL${NC} unknown verb '$verb' in assert.txt"; declare_fail
      ;;
  esac
done < "$DIR/assert.txt"

# ─── fabrication check ──────────────────────────────────────────────────────
# Every number in the output must already exist in the input. A polish pass
# has no business producing a figure the draft did not contain.
echo
invented=""
for n in $(printf '%s' "$FINAL_FLAT" | grep -oE '[0-9]+(\.[0-9]+)?' | sort -u); do
  printf '%s' "$INPUT_FLAT" | grep -qF -- "$n" || invented="$invented $n"
done
if [ -n "$invented" ]; then
  echo "  ${RED}FAIL${NC} NUMBERS   fabricated figure(s):$invented"
  declare_fail
else
  echo "  ${GREEN}pass${NC} NUMBERS   every figure in the output came from the input"
  pass=$((pass+1))
fi

# ─── reporting (not scored) ─────────────────────────────────────────────────
wc_in=$(printf '%s' "$INPUT_FLAT" | wc -w | tr -d ' ')
wc_out=$(printf '%s' "$FINAL_FLAT" | wc -w | tr -d ' ')
pct=$(( (wc_in - wc_out) * 100 / wc_in ))
echo
echo "  ${DIM}words ${wc_in} -> ${wc_out}  (-${pct}%)  — reported, never asserted:"
echo "  compression is an outcome of deleting empty paragraphs, not a target.${NC}"

echo
if [ "$fail" -eq 0 ]; then
  echo "  ${GREEN}pass ${pass}   fail 0${NC}"
  echo
  echo "  ${YELLOW}Not checked here:${NC} whether the reasons were right. Read the agent's"
  echo "  Stage-1 verdict table and Stage-2 report against ${DIM}$DIR/expect.md${NC}."
  echo
  exit 0
else
  echo "  ${GREEN}pass ${pass}${NC}   ${RED}fail ${fail}${NC}"
  echo "  ${DIM}the sealed key explaining each checkpoint is $DIR/expect.md${NC}"
  echo
  exit 1
fi
