#!/usr/bin/env python3
"""Fetch bounded historical style without LLM calls; report expression ledgers separately.

python3 score.py PAPER_DIR --main main.tex [--evaluation evaluation.json] [--base PREVIOUS_DIR]
No available style: exit 3, even if an independent expression ledger is present.
A configured, author-authorized P2G_STYLE_URL receives .tex files; this client never requests LLM evaluation.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

EXCLUDED = {"comments", "comment", "revision", "reviews", "review", "rebuttal", "responses"}
STYLE_METRIC = "pre-chatgpt-bounded-1"
STYLE_DESCRIPTION = ("Similarity to pre-ChatGPT reference papers in word rates and wording habits; "
                     "not AI usage or writing quality.")


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
    return {"score": round(100 * passed / len(complete), 1) if complete else None,
            "passed": passed, "completed": len(complete), "eligible": eligible,
            "coverage": len(complete) / eligible, "uncertain": len(judgments) - len(complete),
            "major_units": [j["unit"] for j in judgments if any(d.get("major") for d in j["defects"])],
            "judgments": judgments}


def style_result(remote: dict) -> dict:
    """Read explicit bounded values, including the transitional expression-style service.

    Never reinterpret its ambiguous style field as historical preference or expression.
    """
    if not isinstance(remote, dict) or "era_similarity_bounded" not in remote:
        raise ValueError("ambiguous service response: bounded historical style is required")
    metric = remote.get("style_metric")
    if metric is not None and metric != STYLE_METRIC:
        raise ValueError("unknown style metric")
    value = remote["era_similarity_bounded"]
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100):
        raise ValueError("invalid bounded style score")
    if metric == STYLE_METRIC and remote.get("style") != value:
        raise ValueError("style and bounded historical score disagree")
    out = {"style": value, "style_status": "ok" if value is not None else "unavailable",
           "style_metric": STYLE_METRIC, "style_description": STYLE_DESCRIPTION}
    for field in ("era_similarity", "era_similarity_bounded", "wording_per_1k", "words", "gain_top3_share"):
        if field in remote:
            out[field] = remote[field]
    share = remote.get("gain_top3_share")
    if share is not None and (type(share) not in (int, float) or not math.isfinite(share) or not 0 <= share <= 1):
        raise ValueError("invalid gain concentration")
    out["concentrated_gain"] = share > 0.5 if share is not None else None
    return out


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
        url = os.environ.get("P2G_STYLE_URL")
        if url:
            files = tex_files(args.paper)
            if args.main not in files:
                raise ValueError("main file missing")
            body = {"main": args.main, "files": files, "evaluate_expression": False}
            if args.base:
                body["base_files"] = tex_files(args.base)
            headers = {"Content-Type": "application/json"}
            if token := os.environ.get("P2G_STYLE_TOKEN"):
                headers["Authorization"] = "Bearer " + token
            req = urllib.request.Request(url, json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=900) as response:
                remote = json.loads(response.read())
            out.update(style_result(remote))
        print(json.dumps(out, ensure_ascii=False))
        return 0 if out["style"] is not None else 3
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
