#!/usr/bin/env python3
"""Report expression acceptance from a source-bound ledger; optionally fetch era diagnostics.

python3 score.py PAPER_DIR --main main.tex [--evaluation evaluation.json] [--base PREVIOUS_DIR]
No URL and no ledger: exit 3 (not evaluated). No score threshold controls rewriting.
A configured, author-authorized P2G_STYLE_URL receives .tex files; this client never requests LLM evaluation.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

EXCLUDED = {"comments", "comment", "revision", "reviews", "review", "rebuttal", "responses"}


def tex_files(directory: Path) -> dict[str, str]:
    return {f.relative_to(directory).as_posix(): f.read_text(encoding="utf-8", errors="strict")
            for f in sorted(directory.rglob("*.tex"))
            if not f.is_symlink() and not any(p.startswith(".") or p.lower() in EXCLUDED
                                              or "review" in p.lower() or "rebuttal" in p.lower()
                                              for p in f.relative_to(directory).parts)}


def acceptance(directory: Path, path: Path, main: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("version") != "expression-acceptance-1":
        raise ValueError("unknown expression ledger version")
    hashes = data.get("source_hashes")
    if not isinstance(hashes, dict) or main not in hashes:
        raise ValueError("ledger must bind source_hashes including the main file")
    for name, digest in hashes.items():
        f = directory / name
        if not f.resolve().is_relative_to(directory.resolve()) or not f.is_file():
            raise ValueError("ledger source is missing or outside paper directory")
        if any(part.lower() in EXCLUDED or "review" in part.lower() for part in Path(name).parts):
            raise ValueError("excluded source in ledger")
        if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            raise ValueError("stale expression ledger: source hash mismatch")
    selected = data.get("selected_ids", [])
    judgments = data.get("judgments", [])
    if (not selected or len(set(selected)) != len(selected)
            or sorted(j.get("unit", "") for j in judgments) != sorted(selected)):
        raise ValueError("ledger must contain exactly one judgment per selected unit")
    for j in judgments:
        if j.get("verdict") not in ("pass", "fail", "uncertain") or not j.get("reason"):
            raise ValueError("invalid judgment")
        ds = j.get("defects")
        if not isinstance(ds, list) or (bool(ds) != (j["verdict"] == "fail")):
            raise ValueError("invalid defects")
        for d in ds:
            if not all(d.get(k) for k in ("quote", "obstacle", "context_unit", "context_quote", "resolution")):
                raise ValueError("failure lacks evidence or resolution")
    complete = [j for j in judgments if j["verdict"] != "uncertain"]
    passed = sum(j["verdict"] == "pass" for j in complete)
    eligible = data.get("eligible", 0)
    if not isinstance(eligible, int) or eligible < len(selected):
        raise ValueError("invalid eligible denominator")
    return {"style": round(100 * passed / len(complete), 1) if complete else None,
            "passed": passed, "completed": len(complete), "eligible": eligible,
            "coverage": len(complete) / eligible, "uncertain": len(judgments) - len(complete),
            "major_units": [j["unit"] for j in judgments if any(d.get("major") for d in j["defects"])],
            "judgments": judgments}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("--main", default="main.tex")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    out = {"style": None, "style_status": "not_evaluated"}
    try:
        if args.evaluation:
            out["expression"] = acceptance(args.paper, args.evaluation, args.main)
            out["style"] = out["expression"]["style"]
            out["style_status"] = "partial" if out["expression"]["uncertain"] else "ok"
        url = os.environ.get("P2G_STYLE_URL")
        if url:
            files = tex_files(args.paper)
            if args.main not in files:
                raise ValueError("main file missing")
            body = {"main": args.main, "files": files}
            if args.base:
                body["base_files"] = tex_files(args.base)
            headers = {"Content-Type": "application/json"}
            if token := os.environ.get("P2G_STYLE_TOKEN"):
                headers["Authorization"] = "Bearer " + token
            req = urllib.request.Request(url, json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=900) as response:
                remote = json.loads(response.read())
            if "era_similarity" not in remote:
                raise ValueError("legacy service response: upgrade service; old style is not expression quality")
            for field in ("era_similarity", "era_similarity_bounded", "wording_per_1k", "words", "gain_top3_share"):
                if field in remote:
                    out[field] = remote[field]
        print(json.dumps(out, ensure_ascii=False))
        return 0 if url or args.evaluation else 3
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
