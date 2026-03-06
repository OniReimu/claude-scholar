#!/usr/bin/env python3
"""Resolve the workspace root for paper-figure-generator runs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def git_root(candidate: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    root = result.stdout.strip()
    return Path(root).resolve() if root else None


def discover_workspace_root(output_dir: Path, method_file: Path, workspace_cwd: Path) -> Path:
    candidates = [
        output_dir.resolve(),
        method_file.resolve().parent,
        workspace_cwd.resolve(),
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        root = git_root(candidate)
        if root is not None:
            return root

    return workspace_cwd.resolve()


def git_rev(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    rev = result.stdout.strip()
    return rev or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--workspace-cwd", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = discover_workspace_root(
        output_dir=Path(args.output_dir),
        method_file=Path(args.method_file),
        workspace_cwd=Path(args.workspace_cwd),
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
