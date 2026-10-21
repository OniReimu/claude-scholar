"""Source-bound ledgers cannot turn missing evidence or stale files into passing scores."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('score_client', Path(__file__).resolve().parents[1] / 'score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'main.tex').write_text('The tank is sealed.\n')
        self.data = {'version': 'expression-acceptance-1',
            'source_hashes': {'main.tex': hashlib.sha256((self.root / 'main.tex').read_bytes()).hexdigest()},
            'eligible': 4, 'selected_ids': ['a', 'b'], 'score': 12,
            'judgments': [{'unit': 'a', 'verdict': 'pass', 'reason': 'clear', 'defects': []},
                          {'unit': 'b', 'verdict': 'uncertain', 'reason': 'missing context', 'defects': []}]}

    def read(self):
        p = self.root / 'ledger.json'
        p.write_text(json.dumps(self.data))
        return score.acceptance(self.root, p, 'main.tex')

    def test_recompute_rate_and_exclude_uncertain(self):
        actual = self.read()
        self.assertEqual((actual['score'], actual['completed'], actual['coverage'], actual['uncertain']), (100, 1, .25, 1))

    def test_stale_source_rejected(self):
        (self.root / 'main.tex').write_text('The tank is open.\n')
        with self.assertRaisesRegex(ValueError, 'stale'): self.read()

    def test_duplicate_and_missing_judgments_rejected(self):
        self.data['judgments'][1] = self.data['judgments'][0]
        with self.assertRaises(ValueError): self.read()

    def test_evidenceless_failure_rejected(self):
        self.data['judgments'][0]['verdict'] = 'fail'
        with self.assertRaises(ValueError): self.read()

    def test_no_completed_judgment_has_no_score(self):
        self.data['judgments'][0]['verdict'] = 'uncertain'
        self.assertIsNone(self.read()['score'])

    def test_source_outside_root_rejected_before_read(self):
        self.data['source_hashes']['../elsewhere.tex'] = 'bad'
        with self.assertRaises(ValueError): self.read()

    def test_transitional_service_uses_bounded_field_not_expression_style(self):
        out = score.style_result({'style': 100, 'era_similarity': 83, 'era_similarity_bounded': 71})
        self.assertEqual(out['style'], 71)
        self.assertIsNone(out['concentrated_gain'])

    def test_new_contract_and_concentration_boundary(self):
        for share, flag in [(None, None), (.5, False), (.51, True)]:
            with self.subTest(share=share):
                out = score.style_result({'style_metric': score.STYLE_METRIC, 'style': 71,
                    'era_similarity_bounded': 71, 'gain_top3_share': share})
                self.assertEqual(out['style_status'], 'ok')
                self.assertEqual(out['concentrated_gain'], flag)

    def test_missing_or_invalid_style_never_becomes_expression(self):
        bad = [{'style': 100}, {'era_similarity_bounded': True}, {'era_similarity_bounded': -1},
               {'era_similarity_bounded': float('nan')}, {'era_similarity_bounded': 101},
               {'style_metric': 'unknown', 'era_similarity_bounded': 71},
               {'style_metric': score.STYLE_METRIC, 'style': 99, 'era_similarity_bounded': 71},
               {'era_similarity_bounded': 71, 'gain_top3_share': 1.1}]
        for remote in bad:
            with self.subTest(remote=remote), self.assertRaises(ValueError):
                score.style_result(remote)
        out = score.style_result({'era_similarity_bounded': None, 'style': 100})
        self.assertIsNone(out['style'])
        self.assertEqual(out['style_status'], 'unavailable')

    def test_ledger_only_keeps_style_unavailable_and_never_calls_service(self):
        self.read()
        stdout = io.StringIO()
        with patch.object(score.sys, 'argv', ['score.py', str(self.root), '--evaluation', str(self.root / 'ledger.json')]), \
             patch.dict(score.os.environ, {}, clear=True), patch.object(score.urllib.request, 'urlopen') as fetch, \
             patch('sys.stdout', stdout):
            rc = score.main()
        self.assertEqual(rc, 3)
        fetch.assert_not_called()
        out = json.loads(stdout.getvalue())
        self.assertIsNone(out['style'])
        self.assertEqual(out['expression']['score'], 100)

    def test_cli_requests_no_llm_and_keeps_independent_ledger(self):
        self.read()
        response = io.BytesIO(json.dumps({'style_metric': score.STYLE_METRIC, 'style': 71,
            'era_similarity_bounded': 71, 'gain_top3_share': .7}).encode())
        stdout = io.StringIO()
        with patch.object(score.sys, 'argv', ['score.py', str(self.root), '--base', str(self.root),
                                             '--evaluation', str(self.root / 'ledger.json')]), \
             patch.dict(score.os.environ, {'P2G_STYLE_URL': 'https://example.invalid/api/style'}, clear=True), \
             patch.object(score.urllib.request, 'urlopen', return_value=response) as fetch, \
             patch('sys.stdout', stdout):
            rc = score.main()
        body = json.loads(fetch.call_args.args[0].data)
        self.assertIs(body['evaluate_expression'], False)
        self.assertEqual(body['files'], body['base_files'])
        out = json.loads(stdout.getvalue())
        self.assertEqual((rc, out['style'], out['expression']['score']), (0, 71, 100))
        self.assertTrue(out['concentrated_gain'])
