#!/usr/bin/env python3
"""Method 节的结构画像：同一把尺子量草稿和范文。

用法:
    uv run skills/claim-architecture-review/scripts/method_profile.py main.tex [--json]
    uv run .../method_profile.py paper_dir/ --method-regex 'approach|framework'

输入可以是 .tex 文件或目录（目录下找含 \\documentclass 的主文件），\\input / \\include 递归展开。
按 \\section 标题找 Method 节；Setup 取 Experiments 章节里的设置子章节
（\\subsection 标题含 setting / setup / implementation / configuration / detail），
找不到就取 Experiments 的第一个子章节——Setup 通常就是它；Experiments 没有子章节时退回整节并标注。

所有指标都是**代理量**，用于和范文对照取区间，不是判决：
- numeric_assignments  行内数学里右侧是纯数值的 `= 数字`、`\\in \\{数字…\\}` 与 "set (to|as) 数字"；
                        `k = 2f+1` 这类符号式定义不计。用 --show 逐条判它是实例化取值还是方法定义的一部分
- display_equations     equation / align / gather / multline / \\[ \\]
- ladder_steps          前面只隔着 ≤1 句（≤25 词）文字的展示公式个数（一句话配一行公式）
- bare_transitions      公式前的文字只以 then / thus / we have / we get / we obtain 收尾
- reasoned_transitions  公式前文字含 substituting / since / because / by \\eqref / applying / combining 等推理标记
- where_clauses         含行内数学的 where 子句数。符号解释有没有交代数据流位置，正则判不了，
                        用 --show 列出原文逐条读（PROSE.EQUATION_EXPLANATION 的粘贴测试）
- citations             \\cite 次数（标准组件靠引用压缩）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

METHOD_RE = r"method|approach|framework|design|construction|technique|proposed|algorithm|protocol|architecture"
EXPERIMENTS_RE = r"experiment|evaluation|empirical"
SETUP_SUB_RE = r"setting|setup|implementation|configuration|detail"
# 这些标题即使含上面的词也不是 Method / Setup（"Problem Setup"、"Threat Model"、"Related Work" …）
EXCLUDE_RE = r"problem|threat|preliminar|background|related|motivation|introduction|conclusion|discussion|limitation"

DISPLAY_ENVS = r"(?:equation|align|gather|multline|eqnarray)\*?"
DISPLAY_RE = re.compile(r"\\begin\{(" + DISPLAY_ENVS + r")\}.*?\\end\{\1\}|\\\[.*?\\\]", re.S)
FLOAT_RE = re.compile(r"\\begin\{(figure|table|algorithm|tabular|lstlisting)\*?\}.*?\\end\{\1\*?\}", re.S)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", re.S)
# 右侧必须是纯数值（可带单位/百分号/科学计数），`k = 2f+1` 这类符号式定义不算
NUM_IN_MATH_RE = re.compile(
    r"(?:=|\\leftarrow|\\gets)\s*-?\d+(?:[.,]\d+)*\s*(?:\\%|%|\\times\s*10\^\{?-?\d+\}?|[kKMB])?\s*(?:$|[,;)\]]|\\[,;!]|\\text|\\quad|\;)"
    r"|\\in\s*\\\{\s*-?\d")
SET_TO_RE = re.compile(r"\bset (?:to|as|at)\s+\$?-?\d", re.I)
BARE_TAIL_RE = re.compile(r"(?:then|thus|hence|we have|we get|we obtain|it follows that|as follows|yields?|gives?)\s*[:,.]?\s*$", re.I)
REASON_RE = re.compile(r"\b(?:substitut\w*|plugging|since|because|by (?:\\eqref|eq|lemma|theorem|definition|proposition|\\cref|\\ref)|applying|combining|from \\eqref|using \\eqref|rearrang\w*|taking the|summing|differentiat\w*)", re.I)
WHERE_RE = re.compile(r"\bwhere\b(.{0,400}?)(?:\.\s|\n\n|$)", re.I | re.S)


def resolve(path: Path, seen: set[Path] | None = None) -> str:
    """读入 tex 并递归展开 \\input / \\include。"""
    seen = seen or set()
    path = path.resolve()
    if path in seen or not path.exists():
        return ""
    seen.add(path)
    text = path.read_text(errors="ignore")
    text = re.sub(r"(?<!\\)%.*", "", text)  # 去注释，保留 \%

    def sub(m: re.Match) -> str:
        name = m.group(2).strip()
        cand = path.parent / name
        if not cand.suffix:
            cand = cand.with_suffix(".tex")
        return resolve(cand, seen)

    return re.sub(r"\\(input|include)\{([^}]*)\}", sub, text)


SKIP_DIRS = {"bkp", "backup", "old", "archive", "sp_back"}


def find_main(d: Path) -> Path:
    """取最浅层、名字含 main 优先的主文件，跳过备份目录。"""
    cands = [f for f in d.rglob("*.tex")
             if not SKIP_DIRS & {x.lower() for x in f.relative_to(d).parts[:-1]}
             and "\\documentclass" in f.read_text(errors="ignore")]
    if not cands:
        sys.exit(f"no file with \\documentclass under {d}")
    return min(cands, key=lambda f: (len(f.relative_to(d).parts), "main" not in f.name, f.name))


def sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\\section\*?\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def setup_part(body: str, rx: str) -> tuple[str, str] | None:
    """在 Experiments 章节里取 Setup 子章节；返回 (标题说明, 正文)。"""
    parts = re.split(r"\\subsection\*?\{((?:[^{}]|\{[^{}]*\})*)\}", body)
    subs = [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    if not subs:
        return None
    for t, b in subs:
        if re.search(rx, t, re.I):
            return t, b
    t, b = subs[0]
    return f"{t} (first subsection, no setup-titled one)", b


def prose_words(s: str) -> int:
    s = DISPLAY_RE.sub(" ", s)
    s = INLINE_MATH_RE.sub(" X ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", s))


def evidence(body: str) -> dict:
    """--show 用：列出每个计数背后的原文，供人工逐条判（代理量必须可核对）。"""
    body = FLOAT_RE.sub(" ", body)
    nums = [f"${x}$" for x in INLINE_MATH_RE.findall(body) if NUM_IN_MATH_RE.search(x)]
    nums += [body[max(0, m.start() - 40):m.end() + 10].replace("\n", " ") for m in SET_TO_RE.finditer(body)]
    wheres = [w.group(0)[:200].replace("\n", " ") for w in WHERE_RE.finditer(DISPLAY_RE.sub(" ", body)) if "$" in w.group(1)]
    return {"numeric_assignments": nums, "where_clauses": wheres}


def profile(body: str) -> dict:
    body = FLOAT_RE.sub(" ", body)
    m_eq = list(DISPLAY_RE.finditer(body))
    ladder = bare = reasoned = 0
    prev_end = 0
    for m in m_eq:
        before = body[prev_end:m.start()]
        words = prose_words(before)
        sentences = len(re.findall(r"[.!?](?:\s|$)", INLINE_MATH_RE.sub("X", before)))
        if words <= 25 and sentences <= 1:
            ladder += 1
        tail = INLINE_MATH_RE.sub("X", before).strip()[-160:]
        if BARE_TAIL_RE.search(tail) and not REASON_RE.search(tail):
            bare += 1
        if REASON_RE.search(tail):
            reasoned += 1
        prev_end = m.end()
    inline = INLINE_MATH_RE.findall(body)
    numeric = sum(1 for x in inline if NUM_IN_MATH_RE.search(x)) + len(SET_TO_RE.findall(body))
    wheres = [w.group(1) for w in WHERE_RE.finditer(DISPLAY_RE.sub(" ", body)) if "$" in w.group(1)]
    words = prose_words(body)
    paras = [p for p in re.split(r"\n\s*\n", DISPLAY_RE.sub(" ", body)) if prose_words(p) >= 15]
    return {
        "words": words,
        "paragraphs": len(paras),
        "display_equations": len(m_eq),
        "eq_per_100w": round(100 * len(m_eq) / words, 2) if words else 0.0,
        "ladder_steps": ladder,
        "bare_transitions": bare,
        "reasoned_transitions": reasoned,
        "numeric_assignments": numeric,
        "numeric_per_100w": round(100 * numeric / words, 2) if words else 0.0,
        "where_clauses": len(wheres),
        "citations": len(re.findall(r"\\(?:cite[tp]?|citep|citet|parencite)\b", body)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--method-regex", default=METHOD_RE)
    ap.add_argument("--experiments-regex", default=EXPERIMENTS_RE)
    ap.add_argument("--setup-regex", default=SETUP_SUB_RE, help="Experiments 下 Setup 子章节标题")
    ap.add_argument("--exclude-regex", default=EXCLUDE_RE)
    ap.add_argument("--list", action="store_true", help="只列出 \\section 标题")
    ap.add_argument("--show", action="store_true", help="列出数值赋值与 where 子句原文")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    main_tex = find_main(a.path) if a.path.is_dir() else a.path
    text = resolve(main_tex)
    text = re.split(r"\\begin\{document\}", text, maxsplit=1)[-1]
    text = re.split(r"\\appendix|\\bibliography\{|\\begin\{thebibliography\}", text, maxsplit=1)[0]

    secs = sections(text)
    if a.list:
        print("\n".join(t for t, _ in secs))
        return
    out: dict[str, dict] = {}
    parts: dict[str, tuple[list[str], str]] = {}
    m_hits = [(t, b) for t, b in secs if re.search(a.method_regex, t, re.I) and not re.search(a.exclude_regex, t, re.I)]
    if m_hits:
        parts["method"] = ([t for t, _ in m_hits], "\n\n".join(b for _, b in m_hits))
    e_hits = [(t, b) for t, b in secs if re.search(a.experiments_regex, t, re.I) and not re.search(a.exclude_regex, t, re.I)]
    if e_hits:
        t, b = e_hits[0]
        sub = setup_part(b, a.setup_regex)
        parts["setup"] = ([f"{t} > {sub[0]}"], sub[1]) if sub else ([f"{t} (no subsections: whole section)"], b)
    for role, (titles, body) in parts.items():
        out[role] = {"sections": titles, **profile(body)}
        if a.show:
            out[role]["evidence"] = evidence(body)
    total = prose_words(text)
    if "method" in out and total:
        out["method"]["share_of_body_words"] = round(out["method"]["words"] / total, 3)

    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    if not out:
        sys.exit("no section title matched; sections are:\n  " + "\n  ".join(t for t, _ in secs)
                 + "\npass --method-regex / --setup-regex explicitly")
    keys = [k for k in next(iter(out.values())) if k not in ("sections", "evidence")]
    print(f"# {main_tex}")
    for role, d in out.items():
        print(f"\n## {role}: {' | '.join(d['sections'])}")
        for k in keys:
            if k in d:
                print(f"  {k:24s} {d[k]}")
        for k, items in d.get("evidence", {}).items():
            print(f"  -- {k}:")
            for x in items:
                print(f"     {x}")


if __name__ == "__main__":
    main()
