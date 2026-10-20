#!/usr/bin/env python3
"""The style score of a LaTeX paper from a p2g style service.

    python3 score.py PAPER_DIR [--main main.tex] [--base PREVIOUS_DIR]

Sends every .tex file under PAPER_DIR to $P2G_STYLE_URL (with $P2G_STYLE_TOKEN as a bearer token when set) and
prints the reply: {"style", "style_within_human_range", "wording_per_1k", "words"}. The loop stops on
style_within_human_range. With --base (the previous round) the reply adds "gain_top3_share": how much of this
round's move comes from the three words that moved most; above 0.5 the round swapped words and its gain does not count. Exit 3 when P2G_STYLE_URL is not set: run the two fixed rounds instead.
The paper leaves this machine; the service says it keeps nothing, but ask the author first.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def tex_files(d: Path) -> dict[str, str]:
    return {f.relative_to(d).as_posix(): f.read_text(encoding="utf-8", errors="ignore")
            for f in sorted(d.rglob("*.tex")) if not any(p.startswith(".") for p in f.relative_to(d).parts)}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return 2
    d = Path(args[0])
    main_tex = args[args.index("--main") + 1] if "--main" in args else "main.tex"
    url = os.environ.get("P2G_STYLE_URL")
    if not url:
        print("P2G_STYLE_URL is not set: no score, run the two fixed rounds", file=sys.stderr)
        return 3
    files = tex_files(d)
    if main_tex not in files:
        print(f"{main_tex} not found under {d}", file=sys.stderr)
        return 2
    headers = {"Content-Type": "application/json"}
    if os.environ.get("P2G_STYLE_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["P2G_STYLE_TOKEN"]
    body = {"main": main_tex, "files": files}
    if "--base" in args:
        body["base_files"] = tex_files(Path(args[args.index("--base") + 1]))
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            print(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"{e.code}: {e.read().decode()[:300]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
