#!/usr/bin/env bash
# Stage 1 of the PROSE.NO_INTERNAL_PROVENANCE undefined-identifier sub-check.
#
# Mechanically extracts identifier-shaped tokens from rendered manuscript prose
# (Golden-G4, C6, RC3, P7, V0). Stage 2 is an LLM judgement that cannot be
# scripted: for each candidate, ask whether the manuscript defines it anywhere —
# a definition, a theorem caption, a notation-table row, or an explicit gloss at
# first use. Report the undefined ones; an identifier a reader cannot resolve is
# either an internal leak (Golden-G4) or a genuine omission (a claim-register
# label used nine times and never defined). Both need author action.
#
# Usage: policy/scripts/extract-undefined-identifiers.sh [TARGET_DIR]
# Output: one line per distinct candidate — "COUNT<TAB>TOKEN<TAB>first file:line"

set -uo pipefail

TARGET_DIR="${1:-.}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: not a directory: $TARGET_DIR" >&2
  exit 2
fi

# BSD grep has no -P. Mirror lint.sh's engine dispatch so this works on macOS.
CAND_RE='\b[A-Z][A-Za-z]*-?[A-Z]?[0-9]+\b'
if grep -qP '' </dev/null 2>/dev/null; then
  scan() { grep -noP "$CAND_RE" 2>/dev/null || true; }
elif command -v ggrep >/dev/null 2>&1; then
  scan() { ggrep -noP "$CAND_RE" 2>/dev/null || true; }
else
  scan() { CAND_RE="$CAND_RE" perl -ne 'while (/$ENV{CAND_RE}/g) { print "$.:$&\n" }' 2>/dev/null || true; }
fi

# Model identifiers and version-like domain terms are scientific facts, not
# internal identifiers — the rule card's exclusion table lists them explicitly.
EXCLUDE='^(GPT|Llama|LLaMA|Claude|Qwen|Gemini|Mistral|Falcon|Phi|BERT|RoBERTa|GPU|CPU|CUDA|FP16|FP32|INT8|BF16|SHA256|MD5|RFC|ISO|IEEE|ACM|USENIX|NeurIPS|ICML|ICLR|CVPR|AAAI|H100|A100|V100|RTX|TPU)'

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

while IFS= read -r file; do
  # Same scrubbing as the lint builtin: drop comments and every construct that
  # never renders into the PDF, so we only judge what a reader actually sees.
  sed -E \
    -e 's/(^|[^\\])%.*$/\1/' \
    -e 's/\\(includegraphics|input|include|bibliography|usepackage|documentclass)(\[[^]]*\])?\{[^}]*\}//g' \
    -e 's/\\(label|ref|Cref|cref|eqref|autoref|cite[a-z]*)\{[^}]*\}//g' \
    -e 's/\\(url|href)\{[^}]*\}//g' \
    "$file" 2>/dev/null \
  | $GREP_BIN -noP '\b[A-Z][A-Za-z]*-?[A-Z]?\d+\b' 2>/dev/null \
  | while IFS=: read -r lno tok; do
      [[ -n "$tok" ]] || continue
      printf '%s\t%s:%s\n' "$tok" "$file" "$lno"
    done
done < <(find "$TARGET_DIR" -name "*.tex" -type f 2>/dev/null | sort) > "$tmp"

if [[ ! -s "$tmp" ]]; then
  echo "No identifier-shaped candidates found under $TARGET_DIR"
  exit 0
fi

echo "Candidate identifiers (stage 1 of 2 — an LLM must now check each for a definition):"
echo ""
printf 'COUNT\tTOKEN\tFIRST SEEN\n'

awk -F'\t' -v excl="$EXCLUDE" '
  { count[$1]++; if (!($1 in first)) first[$1] = $2 }
  END {
    for (t in count) {
      if (t ~ excl) continue
      printf "%d\t%s\t%s\n", count[t], t, first[t]
    }
  }
' "$tmp" | sort -rn

echo ""
echo "Stage 2 (LLM): for each token above, search the manuscript for a definition,"
echo "theorem caption, notation-table row, or first-use gloss. Report the ones with none."
