#!/usr/bin/env python3
"""检查 LaTeX 正文硬规则；只有档 A 对照候选仍须语义判断。

    python3 guardrails.py MAIN.tex [--json REPORT.json] [--check-report OLD.json]
    python3 guardrails.py locks LOCKS.json [--json REPORT.json]

JSON 输出到 stdout（或 --json），简表到 stderr。0 无候选 / 锁来源合法；
1 有候选；2 解析、覆盖或锁台账错误。locks 的回退事件仍须人工核对。
只读输入，支持静态 input/include；不执行 TeX，不展开任意自定义宏。
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

RULES = {
    'colon': 'PROSE.MIDSENTENCE_COLON',
    'semicolon': 'PROSE.SEMICOLON_RESTRICTION',
    'em_dash': 'PROSE.EM_DASH_RESTRICTION',
    'contrast_a': 'PROSE.NEGATION_CONTRAST',
}
MATH = {'equation', 'align', 'alignat', 'gather', 'multline', 'displaymath',
        'math', 'eqnarray', 'flalign'}
CODE = {'verbatim', 'Verbatim', 'lstlisting', 'minted', 'algorithm',
        'algorithmic', 'algorithmicx', 'tikzpicture'}
TABLES = {'table', 'tabular', 'tabularx', 'longtable'}
TITLES = {'title', 'section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph', 'chapter', 'part'}
CONTRAST = re.compile(
    r'\bnot\b[^.!?;:]*?\bbut\b'
    r'|\b(?:is|are|was|were|remain[s]?|become[s]?)\b[^.!?;:]*?,\s*not\b'
    r'|\bneither\b[^.!?;:]*?\bnor\b', re.I)
PUNCT = re.compile(r'\\textemdash\b|---|—|[;:；：]')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blank(s: str) -> str:
    return ''.join('\n' if c == '\n' else ' ' for c in s)


def escaped(s: str, i: int) -> bool:
    n = 0
    while i > 0 and s[i - 1] == '\\':
        n += 1
        i -= 1
    return n % 2 == 1


def math_end(s: str, pos: int, delimiter: str) -> int:
    """先屏蔽数学起点后的注释，保留偏移再寻找未转义的结束符。"""
    chars = list(s[pos:])
    comment_end = pos
    for m in re.finditer('%', s[pos:]):
        start = pos + m.start()
        if start < comment_end or escaped(s, start):
            continue
        comment_end = s.find('\n', start)
        if comment_end < 0:
            comment_end = len(s)
        chars[start - pos:comment_end - pos] = ' ' * (comment_end - start)
    clean = ''.join(chars)
    at = clean.find(delimiter)
    while at >= 0 and escaped(clean, at):
        at = clean.find(delimiter, at + len(delimiter))
    return pos + at if at >= 0 else -1


def lexical(s: str) -> list[str]:
    """保留偏移；只标记可确定的注释、数学、代码，不吞掉普通环境。"""
    marks = [''] * len(s)
    i = 0
    while i < len(s):
        start, reason, end = i, '', i + 1
        if s[i] == '%' and not escaped(s, i):
            reason = 'comment'
            end = s.find('\n', i)
            if end < 0:
                end = len(s)
        elif s[i] == '\\' and not escaped(s, i):
            env = re.match(r'\\begin\{([^}]+)\}', s[i:])
            verb = re.match(r'\\(?:verb\*?|lstinline)(?![A-Za-z])', s[i:])
            if env and env[1].rstrip('*') in MATH | CODE:
                reason = 'math' if env[1].rstrip('*') in MATH else 'code'
                close = '\\end{' + env[1] + '}'
                at = (math_end(s, i + env.end(), close) if reason == 'math'
                      else s.find(close, i + env.end()))
                if at < 0:
                    raise ValueError(f'未闭合环境 {env[1]}')
                end = at + len(close)
            elif verb:
                pos = i + verb.end()
                if pos >= len(s) or s[pos].isspace():
                    raise ValueError('无法解析 verb/lstinline 分隔符')
                at = s.find(s[pos], pos + 1)
                if at < 0:
                    raise ValueError('未闭合行内代码')
                reason, end = 'code', at + 1
            elif s.startswith(('\\(', '\\['), i):
                close = '\\)' if s[i + 1] == '(' else '\\]'
                at = math_end(s, i + 2, close)
                if at < 0:
                    raise ValueError('未闭合数学')
                reason, end = 'math', at + 2
        elif s[i] == '$' and not escaped(s, i):
            delim = '$$' if s.startswith('$$', i) else '$'
            at = math_end(s, i + len(delim), delim)
            if at < 0:
                raise ValueError('未闭合 $ 数学')
            reason, end = 'math', at + len(delim)
        if reason:
            marks[start:end] = [reason] * (end - start)
        i = end
    return marks


def view(s: str, marks: list[str]) -> str:
    return ''.join(c if not reason else ('\n' if c == '\n' else ' ')
                   for c, reason in zip(s, marks))


def group(s: str, pos: int, opening: str = '{') -> tuple[int, int, int]:
    while pos < len(s) and s[pos].isspace():
        pos += 1
    if pos >= len(s) or s[pos] != opening:
        raise ValueError(f'缺少 {opening} 参数，偏移 {pos}')
    closing = '}' if opening == '{' else ']'
    depth, braces = 1, 0
    for i in range(pos + 1, len(s)):
        if escaped(s, i):
            continue
        if opening == '[':
            braces += (s[i] == '{') - (s[i] == '}')
        if braces == 0:
            depth += (s[i] == opening) - (s[i] == closing)
        if depth == 0:
            return pos + 1, i, i + 1
    raise ValueError(f'未闭合 {opening} 参数，偏移 {pos}')


class Source:
    def __init__(self, main: Path):
        self.main = main.resolve()
        self.files: dict[str, dict] = {}
        self.texts: dict[str, str] = {}

    def expand(self, path: Path, stack: tuple = ()) -> tuple[str, list[tuple[str, int]]]:
        path = path.resolve()
        if len(stack) > 100:
            raise ValueError('input/include 嵌套超过 100 层')
        if path in stack:
            raise ValueError(f'循环 input/include: {path}')
        data = path.read_bytes()
        s = data.decode('utf-8')
        key = str(path)
        self.texts[key] = s
        self.files[key] = {'file': key, 'sha256': digest(data)}
        clean = view(s, lexical(s))
        parts, origins, last = [], [], 0
        for m in re.finditer(r'\\(input|include)\b', clean):
            a, b, end = group(clean, m.end())
            name = s[a:b].strip()
            if re.search(r'[\\#{}]', name) or not name:
                raise ValueError(f'非静态输入，无法保证覆盖: {name}')
            child = self.main.parent / name
            if not child.suffix:
                child = child.with_suffix('.tex')
            text, mapping = self.expand(child, stack + (path,))
            parts.extend([s[last:m.start()], text])
            origins.extend((key, j) for j in range(last, m.start()))
            origins.extend(mapping)
            # 子文件 EOF 结束注释；合成换行不改变后续正文的源位置。
            if not text.endswith('\n'):
                parts.append('\n')
                origins.append((key, end - 1))
            last = end
        parts.append(s[last:])
        origins.extend((key, j) for j in range(last, len(s)))
        return ''.join(parts), origins

    def location(self, origins: list[tuple[str, int]], pos: int) -> dict:
        file, offset = origins[pos]
        s = self.texts[file]
        return {'file': file, 'line': s.count('\n', 0, offset) + 1,
                'column': offset - s.rfind('\n', 0, offset),
                'source_hash': self.files[file]['sha256']}


def annotate(s: str) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    marks = lexical(s)
    clean = view(s, marks)
    excluded = [''] * len(s)
    sections = []

    def mark(a: int, b: int, reason: str) -> None:
        for i in range(a, b):
            if not marks[i]:
                marks[i] = reason

    # 检查普通环境与花括号；代码/数学已由 lexical 独立检查。
    stack = []
    for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', clean):
        if m[1] == 'begin':
            stack.append((m[2], m.start()))
        elif not stack or stack[-1][0] != m[2]:
            raise ValueError(f'环境不匹配: {m[0]}')
        else:
            env, start = stack.pop()
            if env.rstrip('*') in TABLES:
                excluded[start:m.end()] = ['table'] * (m.end() - start)
    if stack:
        raise ValueError(f'未闭合环境: {stack[-1]}')
    depth = 0
    for i, c in enumerate(clean):
        if not escaped(clean, i):
            depth += (c == '{') - (c == '}')
            if depth < 0:
                raise ValueError('多余的右花括号')
    if depth:
        raise ValueError('未闭合花括号')

    for m in re.finditer(r'\\([A-Za-z]+|\\)\*?', clean):
        cmd, pos = m[1], m.end()
        mark(m.start(), pos, 'command')
        while pos < len(clean) and clean[pos].isspace():
            pos += 1
        optional = cmd in TITLES | {'caption', 'item', 'includegraphics', 'footnote', '\\'} or cmd.startswith('cite')
        if optional and pos < len(clean) and clean[pos] == '[':
            a, b, pos = group(clean, pos, '[')
            # 标题/caption 的短参数也是可见文本，不能当布局参数丢掉。
            if cmd in TITLES:
                mark(a, b, 'title')
            elif cmd == 'caption':
                excluded[a:b] = ['caption'] * (b - a)
            elif cmd != 'item' and not cmd.startswith('cite'):
                mark(a, b, 'layout')
        if cmd in TITLES | {'caption', 'captionof', 'textbf', 'texttt', 'url', 'href', 'path',
                            'label', 'ref', 'eqref', 'autoref', 'cref', 'Cref',
                            'begin', 'end', 'includegraphics', 'leadtag'} or cmd.startswith('cite'):
            a, b, end = group(clean, pos)
            if cmd in TITLES:
                mark(a, b, 'title')
                if cmd in {'section', 'subsection', 'subsubsection', 'chapter'}:
                    sections.append((m.start(), end))
            elif cmd == 'textbf':
                prefix = view(s, marks)[:m.start()].rstrip(' \t')
                prefix = re.sub(r'[{}]', ' ', prefix)
                # 段首或 item 后的小标题，冒号也可在标题中间。
                at_start = not prefix.strip() or prefix.endswith('\n\n') or re.search(r'\\item(?:\[[^]]*\])?\s*$', clean[:m.start()])
                if at_start:
                    mark(a, b, 'title')
            elif cmd == 'texttt':
                mark(a, b, 'code')
            elif cmd == 'caption':
                excluded[a:b] = ['caption'] * (b - a)
            elif cmd == 'captionof':
                mark(a, b, 'layout')
                pos = end
                while pos < len(clean) and clean[pos].isspace():
                    pos += 1
                if pos < len(clean) and clean[pos] == '[':
                    a, b, pos = group(clean, pos, '[')
                    excluded[a:b] = ['caption'] * (b - a)
                a, b, end = group(clean, pos)
                excluded[a:b] = ['caption'] * (b - a)
            elif cmd == 'leadtag':
                mark(a, b, 'title')
                a, b, end = group(clean, end)
                mark(a, b, 'layout')
                group(clean, end)  # 第三个参数是正文，保留并核覆盖。
            elif cmd in {'url', 'href', 'path'}:
                mark(a, b, 'url' if cmd != 'path' else 'code')
            else:
                mark(a, b, 'layout' if cmd in {'begin', 'end', 'includegraphics'} else 'reference')
                if cmd == 'begin':
                    pos = end
                    while pos < len(clean) and clean[pos].isspace():
                        pos += 1
                    if pos < len(clean) and clean[pos] == '[':
                        a, b, _ = group(clean, pos, '[')
                        mark(a, b, 'layout')
    for m in re.finditer(r'(?:https?://|ftp://|www\.)[^\s{}<>]+|\b\d{1,2}:\d{2}(?::\d{2})?\b', clean):
        end = m.end() if m[0][0].isdigit() else m.start() + len(m[0].rstrip(';:,.!?'))
        mark(m.start(), end, 'url' if not m[0][0].isdigit() else 'time')
    for m in re.finditer(r'\\;', s):
        if not escaped(s, m.start()) and marks[m.start()] not in {'code', 'comment'}:
            marks[m.start():m.end()] = ['thin_space'] * len(m[0])
    return marks, excluded, sections


def enumerate_body(s: str) -> tuple[list[dict], list[tuple[int, int]], list[str]]:
    marks, excluded, sections = annotate(s)
    hits = []
    for m in PUNCT.finditer(s):
        kind = 'semicolon' if m[0] in {';', '；'} else 'colon' if m[0] in {':', '：'} else 'em_dash'
        reason = marks[m.start()]
        if reason == 'command':
            reason = ''  # textemdash 是可见标点。
        if reason == 'title' and kind != 'colon':
            reason = ''
        # 原始枚举保留所有排除项，供检查分母。
        status = 'excluded' if excluded[m.start()] else 'exempt' if reason else 'violation'
        hits.append({'offset': m.start(), 'text': m[0], 'kind': kind,
                     'rule_id': RULES[kind], 'status': status,
                     'reason': excluded[m.start()] or reason or '正文硬规则禁止'})
    # 格式宏与括号变空格，保留宏里的正文；不把单纯否定报成对照。
    prose = view(s, [scope or reason for scope, reason in zip(excluded, marks)])
    prose = re.sub(r'[{}~]', ' ', prose)
    for m in CONTRAST.finditer(prose):
        hits.append({'offset': m.start(), 'text': s[m.start():m.end()],
                     'kind': 'contrast_a', 'rule_id': RULES['contrast_a'],
                     'status': 'pending', 'reason': '候选，按档 A/B 及真实对照判定'})
    for m in CONTRAST.finditer(s):
        reason = marks[m.start()]
        if not excluded[m.start()] and reason in {'math', 'code', 'comment', 'layout', 'reference', 'url'} and all(marks[i] == reason for i in range(m.start(), m.end())):
            hits.append({'offset': m.start(), 'text': m[0], 'kind': 'contrast_a',
                         'rule_id': RULES['contrast_a'], 'status': 'exempt',
                         'reason': reason})
    return sorted(hits, key=lambda h: h['offset']), sections, marks


def counts(hits: list[dict]) -> dict:
    hits = [h for h in hits if h['status'] != 'excluded']
    status = Counter(h['status'] for h in hits)
    return {'raw': len(hits), 'candidates': status['violation'] + status['pending'],
            'violation': status['violation'], 'pending': status['pending'],
            'exempt': status['exempt'],
            'exemptions': dict(Counter(h['reason'] for h in hits if h['status'] == 'exempt'))}


def check_macro_coverage(expanded: str, body: str) -> None:
    """不执行宏。正文调用的宏若藏了候选，要求展开后复核，不能报零。"""
    clean = view(expanded, lexical(expanded))
    unsupported = re.search(
        r'\\(?:DeclareRobustCommand|(?:New|Renew|Provide|Declare)(?:Expandable)?Document(?:Command|Environment))\b', clean)
    if unsupported:
        raise ValueError(f'无法静态覆盖定义方式: {unsupported[0]}，须展开后复核覆盖')
    definitions = {}
    for m in re.finditer(r'\\(newcommand|renewcommand|providecommand|newenvironment|renewenvironment)\b\*?', clean):
        pos = m.end()
        while pos < len(clean) and clean[pos].isspace():
            pos += 1
        if pos < len(clean) and clean[pos] == '{':
            a, b, pos = group(clean, pos)
            name = clean[a:b].strip().lstrip('\\')
        else:
            name_match = re.match(r'\\([A-Za-z]+)', clean[pos:])
            if not name_match:
                raise ValueError('无法解析自定义宏名')
            name = name_match[1]
            pos += name_match.end()
        while True:
            while pos < len(clean) and clean[pos].isspace():
                pos += 1
            if pos < len(clean) and clean[pos] == '[':
                _, _, pos = group(clean, pos, '[')
            else:
                break
        a, b, end = group(clean, pos)
        if m[1].endswith('environment'):
            c, d, _ = group(clean, end)
            definitions['environment:' + name] = clean[a:b] + '\n' + clean[c:d]
            continue
        definitions[name] = clean[a:b]
    marks, excluded, _ = annotate(body)
    visible = view(body, [scope or (reason if reason in {'code', 'math', 'comment'} else '')
                          for scope, reason in zip(excluded, marks)])
    def calls(text: str) -> set[str]:
        return set(re.findall(r'\\([A-Za-z]+)', text)) | {
            'environment:' + name for name in re.findall(r'\\(?:begin|end)\{([^}]+)\}', text)}

    called = calls(visible)
    pending = list(called)
    while pending:
        name = pending.pop()
        template = definitions.get(name, '')
        if PUNCT.search(template) or CONTRAST.search(template):
            raise ValueError(f'宏 \\{name} 定义中含候选，须展开后复核覆盖')
        for dependency in calls(template):
            if dependency not in called:
                called.add(dependency)
                pending.append(dependency)
    for m in re.finditer(r'\\(?:def|gdef|edef|xdef)\s*\\([A-Za-z]+)', clean):
        if m[1] in called:
            raise ValueError(f'无法静态覆盖 TeX 定义: {m[1]}')
    if called & {'csname', 'char', 'symbol', 'catcode', 'directlua', 'scantokens', 'includeonly'}:
        raise ValueError('存在动态 TeX 构造，须展开后复核覆盖')


def scan(main: Path, old_report: Path | None = None) -> dict:
    source = Source(main)
    report = {'mode': 'scan', 'files': [], 'findings': [], 'errors': [], 'excluded_counts': {}}
    excluded_hits = []
    try:
        expanded, origins = source.expand(main)
        clean = view(expanded, lexical(expanded))
        abstract = re.search(r'\\begin\{abstract\}', clean)
        if not abstract:
            raise ValueError('找不到摘要 begin{abstract}')
        tail = re.search(r'\\bibliography\b|\\begin\{thebibliography\}|\\printbibliography\b|\\end\{document\}', clean[abstract.end():])
        if not tail:
            raise ValueError('找不到正文结束边界')
        start, stop = abstract.start(), abstract.end() + tail.start()
        body = expanded[start:stop]
        mapping = origins[start:stop]
        hits, sections, marks = enumerate_body(body)
        report['body_range'] = {'start': source.location(origins, start),
                                'end': source.location(origins, stop), 'characters': len(body)}
        report['body_hash'] = digest(body.encode('utf-8'))
        for h in hits:
            h.update(source.location(mapping, h.pop('offset')))
        excluded_hits = [h for h in hits if h['status'] == 'excluded']
        report['findings'] = [h for h in hits if h['status'] != 'excluded']
        report['excluded_counts'] = dict(Counter(h['reason'] for h in excluded_hits))
        check_macro_coverage(expanded, body)
        # 数学可作为有效章节内容，标题/引用键/注释不能冒充正文。
        content = ''.join(c if r in {'', 'math', 'code'} else ' ' for c, r in zip(body, marks))
        content = re.sub(r'\\[A-Za-z]+|[{}\[\]]', ' ', content)
        end_abstract = re.search(r'\\end\{abstract\}', body)
        if not end_abstract or not re.search(r'[\w]', content[len('\\begin{abstract}'):end_abstract.start()]):
            raise ValueError('摘要或正文为空')
        for n, (_, end) in enumerate(sections):
            next_start = sections[n + 1][0] if n + 1 < len(sections) else len(body)
            # 父节紧接子节允许没有引导段；叶子节必须有内容。
            if n + 1 < len(sections):
                levels = {'chapter': 0, 'section': 1, 'subsection': 2, 'subsubsection': 3}
                current = re.match(r'\\([A-Za-z]+)', body[sections[n][0]:])[1]
                following = re.match(r'\\([A-Za-z]+)', body[next_start:])[1]
                if levels[following] > levels[current]:
                    continue
            if not re.search(r'\w', content[end:next_start]):
                loc = source.location(mapping, end - 1)
                raise ValueError(f'空章节: {loc["file"]}:{loc["line"]}')
        if old_report:
            old = json.loads(old_report.read_text(encoding='utf-8'))
            if not isinstance(old, dict) or old.get('files') != list(source.files.values()) or old.get('body_hash') != report['body_hash']:
                raise ValueError('输入哈希或文件集合变化，旧报告失效')
    except (OSError, UnicodeError, ValueError) as e:
        report['errors'].append(str(e))
    report['files'] = list(source.files.values())
    report['counts'] = counts(report['findings'])
    report['by_rule'] = {rule: counts([h for h in report['findings'] if h['kind'] == kind]) for kind, rule in RULES.items()}
    report['by_file'] = {}
    for file in source.files:
        hits = [h for h in report['findings'] if h['file'] == file]
        report['by_file'][file] = {
            'counts': counts(hits),
            'excluded_counts': dict(Counter(h['reason'] for h in excluded_hits if h['file'] == file)),
            'by_rule': {rule: counts([h for h in hits if h['kind'] == kind]) for kind, rule in RULES.items()}}
    report['exit_code'] = 2 if report['errors'] else int(report['counts']['candidates'] > 0)
    return report


def locks(path: Path) -> dict:
    report = {'mode': 'locks', 'locks': [], 'errors': []}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            raise ValueError('锁台账必须是对象数组')
        for n, lock in enumerate(data):
            try:
                fields = ('sentence', 'orig', 'rollback_reason', 'proposition')
                if not isinstance(lock, dict) or any(not isinstance(lock.get(k), str) or not lock[k].strip() for k in fields):
                    raise ValueError('缺字段或旧式字符串锁')
                if type(lock.get('rollback_round')) is not int or lock['rollback_round'] < 1:
                    raise ValueError('rollback_round 必须是正整数')
                orig = (path.parent / lock['orig']).resolve()
                raw = orig.read_bytes()
                s = raw.decode('utf-8')
                sha = digest(raw)
                if 'orig_sha256' in lock and lock['orig_sha256'] != sha:
                    raise ValueError('orig 哈希不符')
                if lock['sentence'] not in s:
                    raise ValueError('原句不在 orig 中逐字出现')
                hits, _, _ = enumerate_body(lock['sentence'])
                offset = s.index(lock['sentence'])
                report['locks'].append({**lock, 'orig': str(orig), 'orig_sha256': sha,
                    'line': s.count('\n', 0, offset) + 1,
                    'lock_mode': '只锁命题' if counts(hits)['candidates'] else '默认不动',
                    'rollback_verified': False, 'rollback_check': '须人工核对该轮 diff / 回退记录',
                    'findings': hits})
            except (OSError, UnicodeError, ValueError) as e:
                report['errors'].append(f'锁 {n + 1}: {e}')
    except (OSError, UnicodeError, ValueError) as e:
        report['errors'].append(str(e))
    report['exit_code'] = 2 if report['errors'] else 0
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', help='main.tex 或 locks')
    parser.add_argument('ledger', nargs='?', type=Path)
    parser.add_argument('--json', type=Path, help='JSON 输出路径；默认 stdout')
    parser.add_argument('--check-report', type=Path, help='重新扫描并核对旧报告的输入哈希')
    args = parser.parse_args(argv)
    if args.target == 'locks':
        if args.ledger is None or args.check_report:
            parser.error('locks 需要台账文件，不能带 --check-report')
        report = locks(args.ledger)
    else:
        if args.ledger:
            parser.error('扫描只接收一个主文件')
        report = scan(Path(args.target), args.check_report)
    output = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    try:
        if args.json:
            args.json.write_text(output, encoding='utf-8')
        else:
            print(output, end='')
    except OSError as e:
        print(f'无法输出报告: {e}', file=sys.stderr)
        return 2
    if report['mode'] == 'scan':
        print('规则 | 原始 | 候选 | 违规 | 待判 | 豁免', file=sys.stderr)
        for rule, c in report['by_rule'].items():
            print(f'{rule} | {c["raw"]} | {c["candidates"]} | {c["violation"]} | {c["pending"]} | {c["exempt"]}', file=sys.stderr)
        print(f'已排除（caption / table）: {sum(report["excluded_counts"].values())} {report["excluded_counts"]}', file=sys.stderr)
        for h in report['findings']:
            if h['status'] != 'exempt':
                print(f'{h["file"]}:{h["line"]} {h["rule_id"]} {h["status"]}', file=sys.stderr)
        for f in report['files']:
            print(f'{f["file"]} SHA256 {f["sha256"]} {report["by_file"][f["file"]]["counts"]}', file=sys.stderr)
    else:
        for item in report['locks']:
            print(f'{item["orig"]}:{item["line"]} {item["lock_mode"]} SHA256 {item["orig_sha256"]}', file=sys.stderr)
    for error in report['errors']:
        print(f'ERROR: {error}', file=sys.stderr)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
