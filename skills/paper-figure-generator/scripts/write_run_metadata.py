#!/usr/bin/env python3
"""Write run.json for paper-figure-generator with workspace-aware git detection."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from workspace_root import discover_workspace_root, git_rev


def filter_cli_args(raw_args: list[str]) -> list[str]:
    if raw_args and raw_args[0] == "--":
        raw_args = raw_args[1:]

    filtered: list[str] = []
    skip_next = False
    for arg in raw_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--api_key":
            skip_next = True
            continue
        if arg.startswith("--api_key="):
            continue
        filtered.append(arg)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--sam-backend", required=True)
    parser.add_argument("--api-key-source", choices=["cli", "env"], required=True)
    parser.add_argument("--api-key-var", default=None)
    parser.add_argument("--workspace-cwd", default=os.getcwd())
    parser.add_argument("raw_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    method_file = Path(args.method_file).resolve()
    workspace_cwd = Path(args.workspace_cwd).resolve()
    project_root = discover_workspace_root(output_dir, method_file, workspace_cwd)
    raw_args = filter_cli_args(args.raw_args)

    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "method_file": str(method_file),
        "provider": args.provider,
        "sam_backend": args.sam_backend,
        "args": raw_args,
        "env": {
            "api_key_source": args.api_key_source,
            "api_key_var": None if args.api_key_source == "cli" else args.api_key_var,
            "has_OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
            "has_BIANXIE_API_KEY": bool(os.environ.get("BIANXIE_API_KEY")),
            "has_ROBOFLOW_API_KEY": bool(os.environ.get("ROBOFLOW_API_KEY")),
            "has_FAL_KEY": bool(os.environ.get("FAL_KEY")),
        },
        "git_rev": git_rev(project_root),
        "platform": {"system": platform.system(), "release": platform.release()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
