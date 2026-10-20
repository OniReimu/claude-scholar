"""规则范围与 MSR 漏检回归。标准库运行：python3 -m unittest discover -s 此目录 -v。"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'guardrails.py'
SPEC = importlib.util.spec_from_file_location('guardrails', SCRIPT)
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


class GuardrailsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tex = self.root / 'main.tex'

    def paper(self, body: str, ending: str = r'\end{document}') -> Path:
        self.tex.write_text('\\begin{abstract}\n' + body + '\n\\end{abstract}\n' + ending, encoding='utf-8')
        return self.tex

    def scan(self, body: str) -> dict:
        return g.scan(self.paper(body))

    def hits(self, report: dict, kind: str) -> list[dict]:
        self.assertFalse(report['errors'])
        return [h for h in report['findings'] if h['kind'] == kind and h['status'] != 'exempt']

    def cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)

    def test_h_abstract_numeric_semicolon(self) -> None:
        r = self.scan('CI [61.5, 84.9]; 54.2')
        hit, = self.hits(r, 'semicolon')
        self.assertEqual((hit['line'], hit['rule_id'], hit['status']), (2, 'PROSE.SEMICOLON_RESTRICTION', 'violation'))
        self.assertEqual(hit['source_hash'], g.digest(self.tex.read_bytes()))
        self.assertEqual(r['exit_code'], 1)

    def test_letter_semicolon(self) -> None:
        self.assertEqual(len(self.hits(self.scan('a; b'), 'semicolon')), 1)

    def assert_p1_punctuation(self, body: str) -> dict:
        self.paper('Text.', body + '\n' + r'\end{document}')
        report = g.scan(self.tex)
        self.assertEqual(len(self.hits(report, 'colon')), 1)
        self.assertEqual(len(self.hits(report, 'semicolon')), 1)
        self.assertEqual(report['exit_code'], 1)
        return report

    def test_p1_brackets_after_command_are_prose(self) -> None:
        self.assert_p1_punctuation(r'We use \LaTeX [claim: result; next] for output.')

    def test_p1_citation_note_is_prose(self) -> None:
        self.assert_p1_punctuation(r'\cite[see discussion: two cases; both apply]{Smith2020}')

    def test_p1_child_eof_comment_does_not_consume_parent(self) -> None:
        (self.root / 'child.tex').write_text('Child text. % trailing comment', encoding='utf-8')
        report = self.assert_p1_punctuation('\\input{child} Body: result; next.\nClean.')
        for kind, char in (('colon', ':'), ('semicolon', ';')):
            hit, = self.hits(report, kind)
            self.assertEqual(hit['file'], str(self.tex.resolve()))
            self.assertEqual(hit['line'], 4)
            self.assertEqual(hit['column'], self.tex.read_text().splitlines()[3].index(char) + 1)

    def test_p1_math_closer_in_comment(self) -> None:
        self.assert_p1_punctuation('Value $x % $\n+y$; Hidden: text. % $\nSafe.')

    def test_p1_nested_input_uses_main_directory(self) -> None:
        (self.root / 'sections').mkdir()
        (self.root / 'sections/outer.tex').write_text(r'\input{shared}', encoding='utf-8')
        (self.root / 'shared.tex').write_text('Body: result; next.', encoding='utf-8')
        (self.root / 'sections/shared.tex').write_text('Safe content.', encoding='utf-8')
        report = self.assert_p1_punctuation(r'\input{sections/outer}')
        files = {f['file'] for f in report['files']}
        self.assertIn(str((self.root / 'shared.tex').resolve()), files)
        self.assertNotIn(str((self.root / 'sections/shared.tex').resolve()), files)
        for kind in ('colon', 'semicolon'):
            hit, = self.hits(report, kind)
            self.assertEqual((hit['file'], hit['line']), (str((self.root / 'shared.tex').resolve()), 1))

    def assert_p1_macro_coverage(self, definition: str, body: str) -> None:
        self.tex.write_text(definition + '\n' + r'\begin{abstract}Text.\end{abstract}'
                            + '\n' + body + '\n' + r'\end{document}', encoding='utf-8')
        result = self.cli(self.tex)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(json.loads(result.stdout)['errors'])
        self.assertIn('覆盖', result.stderr)

    def test_p1_robust_command_coverage(self) -> None:
        self.assert_p1_macro_coverage(r'\DeclareRobustCommand{\hidden}{Hidden: result; next.}', r'\hidden')

    def test_p1_document_command_coverage(self) -> None:
        self.assert_p1_macro_coverage(r'\NewDocumentCommand{\hidden}{}{Hidden: result; next.}', r'\hidden')

    def test_p1_newenvironment_coverage(self) -> None:
        self.assert_p1_macro_coverage(r'\newenvironment{hidden}{Hidden: result; next.}{}',
                                     r'\begin{hidden}\end{hidden}')

    def test_p1_math_comment_closer_variants(self) -> None:
        for opening, closing in (('$$', '$$'), (r'\(', r'\)'), (r'\[', r'\]'),
                                 (r'\begin{equation}', r'\end{equation}')):
            with self.subTest(opening=opening):
                self.assert_p1_punctuation('Value ' + opening + 'x % ' + closing
                                          + '\n+y' + closing + '; Hidden: text.')

    def test_p1_known_optional_arguments_and_citation_keys(self) -> None:
        report = self.scan(r'Text. \\[size:key] \includegraphics[width:key]{image:key}'
                           r'\begin{itemize}[label:key]\item[Claim: result; next] Body.'
                           r'\end{itemize} \cite[see discussion: two cases; both apply]{Smith:2020}')
        self.assertEqual(len(self.hits(report, 'colon')), 2)
        self.assertEqual(len(self.hits(report, 'semicolon')), 2)
        self.assertEqual(report['counts']['exemptions']['layout'], 4)
        self.assertEqual(report['counts']['exemptions']['reference'], 1)
        self.assertEqual(report['exit_code'], 1)

    def test_p1_environment_definition_is_checked_transitively(self) -> None:
        for definitions, expected in (
            (r'\newenvironment{custom}{Safe text.}{End.}', 0),
            (r'\newenvironment{custom}{Safe text.}{Hidden: result; next.}', 2),
            (r'\newcommand{\hidden}{Hidden: result; next.}'
             r'\newenvironment{custom}{\hidden}{}', 2),
        ):
            with self.subTest(definitions=definitions):
                self.tex.write_text(definitions + r'\begin{abstract}Text.\end{abstract}'
                                    r'\begin{custom}Text.\end{custom}\end{document}', encoding='utf-8')
                report = g.scan(self.tex)
                self.assertEqual(report['exit_code'], expected)
                self.assertEqual(bool(report['errors']), expected == 2)

    def test_digit_semicolon(self) -> None:
        self.assertEqual(len(self.hits(self.scan('1; 2'), 'semicolon')), 1)

    def test_uppercase_after_semicolon(self) -> None:
        self.assertEqual(len(self.hits(self.scan('a; B'), 'semicolon')), 1)

    def test_newline_after_semicolon(self) -> None:
        self.assertEqual(len(self.hits(self.scan('a;\nB'), 'semicolon')), 1)

    def test_math_and_thin_space_counted(self) -> None:
        r = self.scan(r'Value $x;y\;z$ and \; spaces.')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions'], {'math': 1, 'thin_space': 2})
        self.assertEqual(r['by_rule']['PROSE.SEMICOLON_RESTRICTION']['raw'], 3)

    def test_display_math_counted(self) -> None:
        r = self.scan(r'Values \[x;y\] and \(a;b\) and $$c;d$$ and \begin{align}e;f\end{align}.')
        self.assertEqual(r['counts']['exemptions']['math'], 4)
        self.assertEqual(r['exit_code'], 0)

    def test_code_verbatim_counted(self) -> None:
        r = self.scan('Value. \\verb|a;b:c---d|\n\\begin{verbatim}\na;b\n\\end{verbatim}')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions']['code'], 4)

    def test_list_end_only(self) -> None:
        r = self.scan(r'List \begin{itemize}\item a; b; \item c;\end{itemize}')
        self.assertEqual(len(self.hits(r, 'semicolon')), 3)
        self.assertNotIn('list_separator', r['counts']['exemptions'])

    def test_title_colon(self) -> None:
        r = self.scan(r'\paragraph{Setup:} text. \section{Results: summary} Data.')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions']['title'], 2)

    def test_inline_title_colon(self) -> None:
        r = self.scan('\\textbf{Setup:} Data.\n\n\\textbf{Test:} Data.\n\\begin{itemize}\\item \\textbf{Term:} Data.\\end{itemize}')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions']['title'], 3)

    def test_bold_prose_not_title(self) -> None:
        r = self.scan(r'This is \textbf{a claim: Another claim}.')
        self.assertEqual(len(self.hits(r, 'colon')), 1)

    def test_uppercase_after_colon(self) -> None:
        self.assertEqual(len(self.hits(self.scan('We observe: The value rises.'), 'colon')), 1)

    def test_macro_after_colon(self) -> None:
        self.assertEqual(len(self.hits(self.scan(r'We observe: \emph{The} value rises.'), 'colon')), 1)

    def test_url_and_time_counted(self) -> None:
        r = self.scan(r'At 12:30 visit https://example.org/x and \url{https://example.org/y} or \href{https://example.org/z}{the site}.')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions'], {'time': 1, 'url': 3})

    def test_no_comma_not_but(self) -> None:
        self.assertEqual(len(self.hits(self.scan('It is not an estimate but a bound.'), 'contrast_a')), 1)

    def test_fronted_not_but(self) -> None:
        self.assertEqual(len(self.hits(self.scan('Not a bound, but an estimate is given.'), 'contrast_a')), 1)

    def test_multiline_not_but(self) -> None:
        self.assertEqual(len(self.hits(self.scan('It is not a bound\nbut an estimate.'), 'contrast_a')), 1)

    def test_formatted_not_but(self) -> None:
        self.assertEqual(len(self.hits(self.scan(r'It is \emph{not} a bound but an estimate.'), 'contrast_a')), 1)

    def test_copular_positive_first(self) -> None:
        self.assertEqual(len(self.hits(self.scan('It is a bound, not an estimate.'), 'contrast_a')), 1)

    def test_neither_nor(self) -> None:
        self.assertEqual(len(self.hits(self.scan('It is neither a bound nor an estimate.'), 'contrast_a')), 1)

    def test_simple_negation(self) -> None:
        self.assertEqual(self.scan('That difference is not an effect estimate.')['exit_code'], 0)

    def test_rather_than_not_tier_a(self) -> None:
        self.assertEqual(self.scan('We take the median rather than the mean.')['exit_code'], 0)

    def test_macro_and_box_prose_not_layout(self) -> None:
        r = self.scan(r'\leadtag{Title:}{color:key}{Claim: New; statement.}\begin{tcolorbox}[colback=red:blue]Text; Next.\end{tcolorbox}')
        self.assertEqual(len(self.hits(r, 'semicolon')), 2)
        self.assertEqual(len(self.hits(r, 'colon')), 1)
        self.assertEqual(r['counts']['exemptions']['layout'], 2)

    def test_footnote_and_caption(self) -> None:
        r = self.scan(r'Claim\footnote{Note; More.}\caption{Caption; More: Next.}')
        self.assertEqual(len(self.hits(r, 'semicolon')), 1)
        self.assertEqual(r['excluded_counts'], {'caption': 2})
        self.assertEqual(r['counts']['violation'], 1)

    def test_em_dash_forms(self) -> None:
        self.assertEqual(len(self.hits(self.scan(r'A---B—C\textemdash D.'), 'em_dash')), 3)

    def test_comment_counted(self) -> None:
        r = self.scan('Text. % a;b:c---d\nMore.')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions']['comment'], 3)

    def test_boundaries(self) -> None:
        self.tex.write_text('Outside;\n\\begin{abstract}Text.\\end{abstract}\n\\section{End}Safe.\n\\bibliography{refs}\nOutside;', encoding='utf-8')
        r = g.scan(self.tex)
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['raw'], 0)
        self.assertEqual(r['body_range']['start']['line'], 2)

    def test_empty_body_exit_two(self) -> None:
        self.paper('')
        r = self.cli(self.tex)
        self.assertEqual(r.returncode, 2)
        self.assertTrue(json.loads(r.stdout)['errors'])

    def test_missing_abstract_exit_two(self) -> None:
        self.tex.write_text('Text.\\end{document}', encoding='utf-8')
        self.assertEqual(self.cli(self.tex).returncode, 2)

    def test_empty_section(self) -> None:
        self.paper('Text.', r'\section{Empty}\label{sec:empty}\end{document}')
        self.assertEqual(g.scan(self.tex)['exit_code'], 2)

    def test_missing_end(self) -> None:
        self.paper('Text.', '')
        self.assertEqual(g.scan(self.tex)['exit_code'], 2)

    def test_parse_failures(self) -> None:
        for body in ('Text $x;y', 'Text {unclosed', r'Text \begin{itemize}\item x'):
            with self.subTest(body=body):
                self.assertEqual(self.scan(body)['exit_code'], 2)

    def test_included_body(self) -> None:
        (self.root / 'section.tex').write_text('Included; More.', encoding='utf-8')
        self.paper('Text.', r'\section{Results}\input{section}\end{document}')
        r = g.scan(self.tex)
        hit, = self.hits(r, 'semicolon')
        self.assertEqual(Path(hit['file']).name, 'section.tex')
        self.assertEqual(len(r['files']), 2)

    def test_missing_input(self) -> None:
        self.assertEqual(self.scan(r'Text \input{missing}')['exit_code'], 2)

    def test_recursive_input(self) -> None:
        self.assertEqual(self.scan(r'Text \input{main}')['exit_code'], 2)

    def test_hash_change_invalidates_report(self) -> None:
        self.paper('Text.')
        old = self.root / 'report.json'
        old.write_text(json.dumps(g.scan(self.tex)), encoding='utf-8')
        self.assertEqual(g.scan(self.tex, old)['exit_code'], 0)
        self.paper('Changed.')
        self.assertEqual(g.scan(self.tex, old)['exit_code'], 2)

    def test_restored_snapshot_rescanned(self) -> None:
        self.paper('Clean.')
        old = self.root / 'report.json'
        old.write_text(json.dumps(g.scan(self.tex)), encoding='utf-8')
        # 模拟把带漏检分号的旧快照复制回 work，不能复用干净报告。
        self.paper('CI [61.5, 84.9]; 54.2')
        r = g.scan(self.tex, old)
        self.assertEqual(r['exit_code'], 2)
        self.assertEqual(r['counts']['candidates'], 1)
        self.assertEqual(g.scan(self.tex)['exit_code'], 1)

    def test_json_and_table(self) -> None:
        self.paper('Text; More.')
        output = self.root / 'report.json'
        r = self.cli(self.tex, '--json', output)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(output.read_text())['counts']['candidates'], 1)
        self.assertIn('PROSE.SEMICOLON_RESTRICTION', r.stderr)
        self.assertIn('SHA256', r.stderr)

    def test_math_counterpart_contrasts(self) -> None:
        for body in (r'It is not $A$ but $B$.', r'It is $A$, not $B$.', r'It is neither $A$ nor $B$.'):
            with self.subTest(body=body):
                self.assertEqual(len(self.hits(self.scan(body), 'contrast_a')), 1)

    def test_parent_section_with_subsection(self) -> None:
        self.paper('Text.', r'\section{Parent}\label{sec:parent}\subsection{Child}Text.\end{document}')
        self.assertEqual(g.scan(self.tex)['exit_code'], 0)

    def test_macro_definition_cannot_hide_candidate(self) -> None:
        self.tex.write_text(r'\newcommand{\hidden}{Text; More.}\begin{abstract}Text \hidden\end{abstract}\end{document}', encoding='utf-8')
        self.assertEqual(g.scan(self.tex)['exit_code'], 2)

    def test_transparent_macro_argument_is_scanned(self) -> None:
        self.tex.write_text(r'\newcommand{\prose}[1]{#1}\begin{abstract}Text \prose{Hidden; More.}\end{abstract}\end{document}', encoding='utf-8')
        self.assertEqual(g.scan(self.tex)['counts']['candidates'], 1)

    def test_g_to_h_remaining_candidate_not_zero(self) -> None:
        # 固定漏检机制：多种违规清理后，H 的数字分号仍须独立检出。
        g_version = 'Claim: Result. It is not A but B. CI [61.5, 84.9]; 54.2'
        h_version = 'Claim. Result. It is B. CI [61.5, 84.9]; 54.2'
        self.assertEqual(self.scan(g_version)['counts']['candidates'], 3)
        self.assertEqual(self.scan(h_version)['counts']['candidates'], 1)

    def test_input_hash_change_invalidates_report(self) -> None:
        child = self.root / 'section.tex'
        child.write_text('Clean.', encoding='utf-8')
        self.paper(r'Text \input{section}')
        old = self.root / 'old.json'
        old.write_text(json.dumps(g.scan(self.tex)), encoding='utf-8')
        child.write_text('Changed; text.', encoding='utf-8')
        self.assertEqual(g.scan(self.tex, old)['exit_code'], 2)

    def test_caption_short_title_is_counted(self) -> None:
        r = self.scan(r'Text \caption[Short; caption]{Long; caption}')
        self.assertEqual(r['excluded_counts'], {'caption': 2})
        self.assertEqual(r['counts']['candidates'], 0)

    def test_url_followed_by_visible_semicolon(self) -> None:
        r = self.scan('Visit https://example.org; Next step.')
        self.assertEqual(len(self.hits(r, 'semicolon')), 1)
        self.assertEqual(r['counts']['exemptions']['url'], 1)

    def test_per_file_exemptions(self) -> None:
        r = self.scan(r'Value $x;y$ and \; spaces.')
        self.assertEqual(r['by_file'][str(self.tex.resolve())]['counts']['exemptions'], {'math': 1, 'thin_space': 1})

    def test_colons_and_semicolons_are_violations_not_pending(self) -> None:
        r = self.scan('Claim: Result; next：结果；结束。')
        self.assertEqual(r['counts']['violation'], 4)
        self.assertEqual(r['counts']['pending'], 0)

    def test_tier_a_still_pending(self) -> None:
        r = self.scan('It is not A but B.')
        self.assertEqual(r['counts']['pending'], 1)

    def test_texttt_code(self) -> None:
        r = self.scan(r'We record \texttt{datasets: [X]; nested {Y: Z}}. Next: result.')
        self.assertEqual(r['counts']['exemptions']['code'], 3)
        self.assertEqual(len(self.hits(r, 'colon')), 1)
        self.assertEqual(r['counts']['pending'], 0)

    def test_run_in_title_middle_colon(self) -> None:
        for title in ('TV-1: Static extraction scope. ', 'TV-2: Internal validity. ',
                      'TV-3: Construct scope. ', 'TV-4: Statistical scope. '):
            with self.subTest(title=title):
                r = self.scan(r'\smallskip\noindent\textbf{' + title + '}\nText.')
                self.assertEqual(r['counts']['exemptions']['title'], 1)
                self.assertEqual(r['exit_code'], 0)

    def test_run_in_after_blank_line(self) -> None:
        r = self.scan('Text.\n\n  \\smallskip\\noindent\\textbf{TV-1: Scope. } More.')
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['counts']['exemptions']['title'], 1)

    def test_paragraph_middle_colon(self) -> None:
        r = self.scan(r'\paragraph{TV-1: Static extraction scope.} Text.')
        self.assertEqual(r['exit_code'], 0)

    def test_title_semicolon_not_exempt(self) -> None:
        r = self.scan(r'\textbf{Scope; extraction.} Text.')
        self.assertEqual(len(self.hits(r, 'semicolon')), 1)

    def test_caption_punctuation_excluded(self) -> None:
        r = self.scan(r'Text. \caption{Caption; more: \textbf{Result:} $x;y$.}')
        self.assertEqual(r['excluded_counts'], {'caption': 4})
        self.assertEqual(r['findings'], [])
        self.assertEqual(r['counts']['raw'], 0)
        self.assertEqual(r['exit_code'], 0)

    def test_captionof_optional_short_caption(self) -> None:
        r = self.scan(r'Text. \captionof{figure}[Short: caption]{Long; caption}')
        self.assertEqual(r['excluded_counts'], {'caption': 2})
        self.assertEqual(r['exit_code'], 0)

    def test_table_and_tabular_excluded(self) -> None:
        for env in ('table', 'table*', 'tabular', 'tabular*', 'tabularx', 'longtable'):
            with self.subTest(env=env):
                r = self.scan('Text. \\begin{' + env + '} Cell; value: next. \\end{' + env + '} Body; next.')
                self.assertEqual(r['excluded_counts'], {'table': 2})
                self.assertEqual(len(self.hits(r, 'semicolon')), 1)
                self.assertEqual(r['counts']['raw'], 1)

    def test_nested_table_caption_not_double_counted(self) -> None:
        r = self.scan(r'Text. \begin{table}\caption{Title: note; more.}\begin{tabular}{ll}A: B & C; D\\\end{tabular}\end{table}')
        self.assertEqual(r['excluded_counts'], {'caption': 2, 'table': 2})
        self.assertEqual(r['findings'], [])
        self.assertEqual(r['exit_code'], 0)

    def test_input_tables_excluded_by_environment(self) -> None:
        tables = self.root / 'tables'
        tables.mkdir()
        (tables / 'whole.tex').write_text(r'\begin{table}\caption{Title; note: next.}\begin{tabular}{ll}A: B & C; D\\\end{tabular}\end{table}', encoding='utf-8')
        (tables / 'rows.tex').write_text(r'A: B & C; D\\', encoding='utf-8')
        r = self.scan(r'Text. \input{tables/whole}\begin{tabular}{ll}\input{tables/rows}\end{tabular} Body; next.')
        self.assertEqual(r['excluded_counts'], {'caption': 2, 'table': 4})
        self.assertEqual(len(self.hits(r, 'semicolon')), 1)
        self.assertEqual(len(r['files']), 3)
        self.assertEqual(r['by_file'][str((tables / 'rows.tex').resolve())]['excluded_counts'], {'table': 2})
        self.assertEqual(r['by_file'][str((tables / 'whole.tex').resolve())]['counts']['raw'], 0)

    def test_tables_directory_does_not_exclude_prose(self) -> None:
        (self.root / 'tables').mkdir()
        (self.root / 'tables' / 'prose.tex').write_text('Body: result;', encoding='utf-8')
        r = self.scan(r'Text. \input{tables/prose}')
        self.assertEqual(r['counts']['violation'], 2)

    def test_footnote_and_quote_not_exempt(self) -> None:
        r = self.scan(r'Text \footnote{Note: result; next.} \begin{quote}Quoted: result; next.\end{quote}')
        self.assertEqual(r['counts']['violation'], 4)

    def test_reference_keys_exempt(self) -> None:
        r = self.scan(r'Text \cite{author:key;other} and \ref{sec:result}.')
        self.assertEqual(r['counts']['exemptions']['reference'], 3)
        self.assertEqual(r['exit_code'], 0)

    def test_macro_called_only_in_excluded_scope(self) -> None:
        self.tex.write_text(r'\newcommand{\hidden}{Text; More: result.}\begin{abstract}Text.\caption{\hidden}\begin{table}\hidden\end{table}\end{abstract}\end{document}', encoding='utf-8')
        self.assertEqual(g.scan(self.tex)['exit_code'], 0)

    def ledger(self, sentence: str = 'Original sentence.') -> tuple[Path, dict]:
        (self.root / 'orig.tex').write_text(sentence, encoding='utf-8')
        lock = {'sentence': sentence, 'orig': 'orig.tex', 'rollback_round': 1,
                'rollback_reason': 'Changed strength', 'proposition': 'Preserve original strength'}
        return self.root / 'locks.json', lock

    def check_lock(self, path: Path, data: object) -> dict:
        path.write_text(json.dumps(data), encoding='utf-8')
        return g.locks(path)

    def test_locks_missing_fields(self) -> None:
        for field in ('sentence', 'orig', 'rollback_round', 'rollback_reason', 'proposition'):
            with self.subTest(field=field):
                path, lock = self.ledger()
                del lock[field]
                self.assertEqual(self.check_lock(path, [lock])['exit_code'], 2)

    def test_locks_old_string_array(self) -> None:
        path, _ = self.ledger()
        self.assertEqual(self.check_lock(path, ['Original sentence.'])['exit_code'], 2)

    def test_locks_sentence_missing(self) -> None:
        path, lock = self.ledger()
        lock['sentence'] = 'Invented sentence.'
        self.assertEqual(self.check_lock(path, [lock])['exit_code'], 2)

    def test_locks_semicolon_proposition_only(self) -> None:
        path, lock = self.ledger('Original; sentence.')
        r = self.check_lock(path, [lock])
        self.assertEqual(r['exit_code'], 0)
        self.assertEqual(r['locks'][0]['lock_mode'], '只锁命题')
        self.assertFalse(r['locks'][0]['rollback_verified'])

    def test_locks_valid(self) -> None:
        path, lock = self.ledger()
        self.check_lock(path, [lock])
        r = self.cli('locks', path)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(json.loads(r.stdout)['locks'][0]['lock_mode'], '默认不动')

    def test_locks_hash_mismatch(self) -> None:
        path, lock = self.ledger()
        lock['orig_sha256'] = 'stale'
        self.assertEqual(self.check_lock(path, [lock])['exit_code'], 2)

    def test_locks_bad_round(self) -> None:
        for value in (True, 0, 'r1'):
            path, lock = self.ledger()
            lock['rollback_round'] = value
            self.assertEqual(self.check_lock(path, [lock])['exit_code'], 2)


if __name__ == '__main__':
    unittest.main()
