#!/usr/bin/env python3
"""Method 节的结构画像：同一把尺子量草稿和范文。

用法:
    uv run skills/claim-architecture-review/scripts/method_profile.py main.tex [--json]
    uv run .../method_profile.py paper_dir/ --method-regex 'approach|framework'

输入可以是 .tex 文件或目录（目录下找含 \\documentclass 的主文件；同一层有多个主文件时报错列出，
请直接传文件），\\input / \\include 递归展开。\\iffalse 块、\\newif 开关已知取值的条件分支、
以及单参数的空宏（注释宏）与恒等宏都先展开，避免两个分支或被注释的正文都算进去。

按 \\section 标题找 Method 节（可以是多节）。Setup 是 Experiments 章节里第一个报告结果的子章节之前的部分：
章节开头的导语，加上开头连续几个标题像设置的子章节（setting / setup / implementation / dataset /
baseline / metric / task / attack …）。一个都不像时，导语够长就取导语，否则取第一个子章节（Setup 通常就是它）。

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
- words_per_paragraph   段落切得过碎时，按段比较篇幅会失真，应按组件汇总
- unvalued_params       Method 里出现的希腊字母参数和多字母下标符号（T_{min}、W_{\\max}），在 Setup、
                        附录与表格里都找不到 `= 数字` / `\\in \\{数字…\\}` / 表格取值的。反向检查用，代理量
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

METHOD_RE = r"method|approach|framework|design|construction|technique|proposed|algorithm|protocol|architecture"
EXPERIMENTS_RE = r"experiment|evaluation|empirical"
SETUP_SUB_RE = (r"setting|setup|implementation|configuration|detail|dataset|benchmark|baseline|metric"
                r"|task|attack|model|environment|hardware|protocol|workload|testbed")
# 报告结果的子章节："RQ1: …"、问句标题、results / ablation / overhead …；遇到它 Setup 就结束
RESULTS_SUB_RE = (r"^\s*(?:RQ|Q)\s*\d|\?|result|finding|ablation|comparison|effectiveness|performance|overhead"
                  r"|robustness|scalability|case stud|main eval|sensitivity")
# 这些标题即使含上面的词也不是 Method / Setup（"Problem Setup"、"Threat Model"、"Related Work" …）
EXCLUDE_RE = r"problem|threat|preliminar|background|related|motivation|introduction|conclusion|discussion|limitation"

DISPLAY_ENVS = r"(?:equation|align|gather|multline|eqnarray)\*?"
DISPLAY_RE = re.compile(r"\\begin\{(" + DISPLAY_ENVS + r")\}.*?\\end\{\1\}|\\\[.*?\\\]", re.S)
FLOAT_RE = re.compile(r"\\begin\{(figure|table|algorithm|tabular|lstlisting)\*?\}.*?\\end\{\1\*?\}", re.S)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$|\\\((.+?)\\\)", re.S)
# 右侧必须是纯数值（可带单位/百分号/科学计数），`k = 2f+1` 这类符号式定义不算
NUM_IN_MATH_RE = re.compile(
    r"(?:=\}?|\\leftarrow|\\gets)\s*-?\d+(?:[.,]\d+)*\s*(?:\\%|%|\\times\s*10\^\{?-?\d+\}?|[kKMB])?\s*(?:$|[,;)\]]|\\[,;!]|\\text|\\quad|\;)"
    r"|\\in\s*\\\{\s*-?\d[\d.,\s-]*,[\d.\s-]*,[\d.,\s-]*\\\}")  # 至少三个纯数值；{-1,1}、{1,\dots,K} 是定义域
SET_TO_RE = re.compile(r"\bset (?:to|as|at)\s+\$?-?\d", re.I)
BARE_TAIL_RE = re.compile(r"(?:then|thus|hence|we have|we get|we obtain|it follows that|as follows|yields?|gives?)\s*[:,.]?\s*$"
                          r"|\b(?:is|are|becomes?|as)\s*:\s*$", re.I)
REASON_RE = re.compile(r"\b(?:substitut\w*|plugging|since|because|by (?:\\eqref|eq|lemma|theorem|definition|proposition|\\cref|\\ref)|applying|combining|from \\eqref|using \\eqref|rearrang\w*|taking the|summing|differentiat\w*)", re.I)
WHERE_RE = re.compile(r"\bwhere\b(.{0,400}?)(?:\.\s|\n\n|$)", re.I | re.S)
# "in cases where …" 是条件句，不是符号解释
WHERE_COND_RE = re.compile(r"\b(?:case|cases|setting|settings|scenario|scenarios|situation|situations|regime|regimes|those|instances|places)\s*$", re.I)
FORMAT_MACROS = r"emph|textbf|textit|textsc|texttt|textsf|textrm|underline|mbox|text|textnormal|uline"
GREEK = (r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|kappa|lambda|mu|nu|xi|rho"
         r"|sigma|tau|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega")
PARAM_RE = re.compile(r"\\(?:" + GREEK + r")(?![A-Za-z])(?:_\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|_\\?[A-Za-z0-9]+)?"
                      r"|\b[A-Za-z]_\{\\?(?:text|mathrm|mathit|mathsf|textrm|rm|operatorname)?\{?\\?[A-Za-z]{2,}\}?\}")


def inline_math(s: str) -> list[str]:
    return [a or b for a, b in INLINE_MATH_RE.findall(s)]


def arg_end(s: str, i: int) -> int:
    """s[i] == '{'：返回配对 '}' 之后的位置；不配对返回 -1。"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def apply_macro(s: str, name: str, keep: bool) -> str:
    """把 \\name{arg} 换成 arg（恒等宏）或删掉（空宏），参数按括号配对。"""
    out, i, pat = [], 0, re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z])\s*\{")
    while m := pat.search(s, i):
        e = arg_end(s, m.end() - 1)
        if e < 0:
            break
        out += [s[i:m.start()], s[m.end():e - 1] if keep else " "]
        i = e
    return "".join(out) + s[i:]


def expand_switches(text: str) -> str:
    """\\iffalse、已知取值的 \\newif 开关、单参数空宏/恒等宏。取值不明的开关两支都留并告警。"""
    truth = {"false": False, "true": True}
    for name in re.findall(r"\\newif\s*\\if([A-Za-z]+)", text):
        sets = re.findall(r"\\" + name + r"(true|false)\b", text.split("\\begin{document}")[0])
        truth[name] = sets[-1] == "true" if sets else False  # \newif 默认 false
    cond = re.compile(r"\\if(" + "|".join(map(re.escape, truth)) + r")\b"
                      r"((?:(?!\\if(?:" + "|".join(map(re.escape, truth)) + r")\b|\\fi\b).)*?)\\fi\b", re.S)
    while (n := cond.subn(lambda m: (lambda a, b: a if truth[m.group(1)] else b)(
            *(re.split(r"\\else\b", m.group(2), maxsplit=1) + [""])[:2]), text))[1]:
        text = n[0]
    for name, body in re.findall(r"\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*\[1\]\s*\{\s*(#1)?\s*\}", text):
        text = apply_macro(text, name, keep=bool(body))
    return text


def resolve(path: Path, seen: set[Path] | None = None) -> str:
    """读入 tex 并递归展开 \\input / \\include。"""
    seen = seen or set()
    path = path.resolve()
    if path in seen or not path.exists():
        return ""
    seen.add(path)
    text = path.read_text(errors="ignore")
    text = re.sub(r"^[ \t]*%.*\n", "", text, flags=re.M)  # 整行注释连换行删掉，不留空行
    text = re.sub(r"(?<!\\)%.*", "", text)  # 行尾注释，保留 \%

    def sub(m: re.Match) -> str:
        name = m.group(2).strip()
        cand = path.parent / name
        # "1.Introduction" 这类带点的文件名：Path 会把 ".Introduction" 当后缀，所以先试补 .tex
        if not cand.name.endswith(".tex") and (path.parent / (name + ".tex")).exists():
            cand = path.parent / (name + ".tex")
        return resolve(cand, seen)

    return re.sub(r"\\(input|include)\{([^}]*)\}", sub, text)


SKIP_DIRS = {"bkp", "backup", "old", "archive", "sp_back"}


def find_main(d: Path) -> Path:
    """取最浅层的主文件，跳过备份目录；最浅层不止一个时报错，不替作者猜投哪个版本。"""
    cands = [f for f in d.rglob("*.tex")
             if not SKIP_DIRS & {x.lower() for x in f.relative_to(d).parts[:-1]}
             and re.search(r"^[^%\n]*\\documentclass", f.read_text(errors="ignore"), re.M)]
    if not cands:
        sys.exit(f"no file with \\documentclass under {d}")
    depth = min(len(f.relative_to(d).parts) for f in cands)
    top = sorted(f for f in cands if len(f.relative_to(d).parts) == depth)
    if len(top) > 1:
        sys.exit("several main files; pass one explicitly:\n  " + "\n  ".join(map(str, top)))
    return top[0]


def sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\\section\*?\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def setup_part(body: str, rx: str) -> tuple[str, str] | None:
    """Setup = 第一个报告结果的子章节之前的部分；返回 (标题说明, 正文)。"""
    parts = re.split(r"\\subsection\*?\{((?:[^{}]|\{[^{}]*\})*)\}", body)
    pre, subs = parts[0], [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    if not subs:
        return None
    lead = []
    for t, b in subs:
        if re.search(RESULTS_SUB_RE, t, re.I) or not re.search(rx, t, re.I):
            break
        lead.append((t, b))
    if lead:
        return " + ".join(t for t, _ in lead), "\n\n".join([pre] + [b for _, b in lead])
    if prose_words(pre) >= 80:
        return "(text before the first subsection)", pre
    t, b = subs[0]
    return f"{t} (first subsection, no setup-titled one)", b


def prose_words(s: str) -> int:
    s = DISPLAY_RE.sub(" ", s)
    s = INLINE_MATH_RE.sub(" X ", s)
    # \emph{…}、\textbf{…} 里是正文，先拆壳再删其余命令（\cite{…}、\ref{…} 连参数一起删）
    while (n := re.subn(r"\\(?:" + FORMAT_MACROS + r")\s*\{([^{}]*)\}", r" \1 ", s))[1]:
        s = n[0]
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", s)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", s))


def evidence(body: str) -> dict:
    """--show 用：列出每个计数背后的原文，供人工逐条判（代理量必须可核对）。"""
    body = FLOAT_RE.sub(" ", body)
    nums = [f"${x}$" for x in inline_math(body) if NUM_IN_MATH_RE.search(x)]
    nums += [body[max(0, m.start() - 40):m.end() + 10].replace("\n", " ") for m in SET_TO_RE.finditer(body)]
    return {"numeric_assignments": nums, "where_clauses": [w[:200].replace("\n", " ") for w in where_clauses(body, full=True)]}


def where_clauses(body: str, full: bool = False) -> list[str]:
    flat = DISPLAY_RE.sub(" ", body)
    return [w.group(0) if full else w.group(1) for w in WHERE_RE.finditer(flat)
            if INLINE_MATH_RE.search(w.group(1)) and not WHERE_COND_RE.search(flat[max(0, w.start() - 30):w.start()])]


def unvalued_params(method: str, valued_in: str) -> list[str]:
    """Method 里的参数符号，在 Setup / 附录 / 表格里找不到取值的。"""
    syms = sorted({m.group(0) for x in inline_math(method) + DISPLAY_RE.findall(method) for m in PARAM_RE.finditer(x)})
    out = []
    for sym in syms:
        e = re.escape(sym)
        has_value = re.search(e + r"\s*(?:\{=\}|=|\\in|\\leftarrow)\s*\\?\{?\s*-?\d", valued_in) or \
            re.search(e + r"[^&\n]{0,20}&\s*\$?-?\d", valued_in)  # 表格：$\lambda$ & 0.1
        if not has_value:
            out.append(sym)
    return out


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
    inline = inline_math(body)
    numeric = sum(1 for x in inline if NUM_IN_MATH_RE.search(x)) + len(SET_TO_RE.findall(body))
    wheres = where_clauses(body)
    words = prose_words(body)
    paras = [p for p in re.split(r"\n\s*\n|\\par\b", DISPLAY_RE.sub(" ", body)) if prose_words(p) >= 15]
    return {
        "words": words,
        "paragraphs": len(paras),
        "words_per_paragraph": round(words / len(paras)) if paras else 0,
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
    text = expand_switches(resolve(main_tex))
    text = re.split(r"\\begin\{document\}", text, maxsplit=1)[-1]
    # 附录不进正文画像，但反向检查要在附录里找取值
    text, _, appendix = (re.split(r"(\\appendix|\\bibliography\{|\\begin\{thebibliography\})", text, maxsplit=1)
                         + ["", ""])[:3]

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
    if "method" in parts:
        # 反向检查：取值可以在 Setup、整个 Experiments、附录或表格里
        valued_in = "\n".join(b for _, b in e_hits) + "\n" + appendix
        miss = unvalued_params(parts["method"][1], valued_in)
        out["method"]["unvalued_params"] = len(miss)
        out["method"].setdefault("evidence", {})["unvalued_params"] = miss
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
