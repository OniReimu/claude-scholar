#!/usr/bin/env bash
# Lightweight static checks for skills/ content.
# - Ensures each skills/*/SKILL.md has YAML frontmatter with name/description
# - Ensures referenced local files exist (best-effort, markdown-path based)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_DIR="$ROOT_DIR/skills"

fail() {
  echo "Error: $*" >&2
  exit 1
}

if [ ! -d "$SKILLS_DIR" ]; then
  fail "skills/ directory not found: $SKILLS_DIR"
fi

echo "Validating skills under: $SKILLS_DIR"

missing_frontmatter=0
missing_fields=0
missing_files=0

while IFS= read -r -d '' skill_md; do
  rel="${skill_md#$ROOT_DIR/}"
  dir="$(dirname "$skill_md")"

  # Frontmatter must start with ---
  first_line="$(head -n 1 "$skill_md" || true)"
  if [ "$first_line" != "---" ]; then
    echo "FAIL frontmatter: $rel (does not start with ---)"
    missing_frontmatter=1
    continue
  fi

  # Best-effort: extract frontmatter block (until next ---)
  fm="$(awk 'NR==1{next} /^---[[:space:]]*$/{exit} {print}' "$skill_md")"
  echo "$fm" | rg -q '^name:' || { echo "FAIL frontmatter: $rel (missing name:)"; missing_fields=1; }
  echo "$fm" | rg -q '^description:' || { echo "FAIL frontmatter: $rel (missing description:)"; missing_fields=1; }

  # Best-effort: validate referenced local files exist.
  # 1) Markdown links: [text](path)
  while IFS= read -r link; do
    # Extract inside (...) from a full markdown link match.
    path="$(echo "$link" | sed -e 's/^.*](/' -e 's/)$//')"

    # ignore URLs and non-files
    case "$path" in
      http://*|https://*|mailto:*|tel:*|javascript:*|\#*)
        continue
        ;;
    esac

    # Trim surrounding whitespace
    path="$(echo "$path" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -n "$path" ] || continue

    if [ -f "$dir/$path" ] || [ -f "$ROOT_DIR/$path" ]; then
      continue
    fi

    echo "FAIL reference: $rel -> $path (missing)"
    missing_files=1
  done < <(rg -o --no-filename '\\[[^\\]]+\\]\\([^\\)]+\\)' "$skill_md" || true)

  # 2) Code-span paths: `references/foo.md`, `scripts/bar.sh`, etc.
  while IFS= read -r code; do
    # Strip surrounding backticks without invoking command substitution.
    if [ "${#code}" -lt 2 ]; then
      continue
    fi
    path="${code:1:${#code}-2}"

    # Only consider obvious relative paths with a file extension.
    if ! echo "$path" | rg -q '/.+\\.[A-Za-z0-9]{1,6}$'; then
      continue
    fi

    if [ -f "$dir/$path" ] || [ -f "$ROOT_DIR/$path" ]; then
      continue
    fi

    echo "FAIL reference: $rel -> $path (missing)"
    missing_files=1
  done < <(rg -o --no-filename '\\x60[^\\x60]+\\x60' "$skill_md" || true)
done < <(find "$SKILLS_DIR" -mindepth 2 -maxdepth 2 -name SKILL.md -print0)

if [ "$missing_frontmatter" -ne 0 ] || [ "$missing_fields" -ne 0 ] || [ "$missing_files" -ne 0 ]; then
  echo ""
  echo "Validation failed."
  exit 1
fi

echo "OK: skills validation passed."
