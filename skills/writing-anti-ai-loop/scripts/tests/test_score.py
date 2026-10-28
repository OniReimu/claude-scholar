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

    def test_old_or_unversioned_service_rejected(self):
        for metric in (None, 'pre-chatgpt-bounded-1', 'pre-chatgpt-bounded-2'):
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, 'unknown style metric'):
                score.style_result({'style_metric': metric, 'style': 71, 'era_similarity_bounded': 71})

    def test_body_scope_metadata_preserved(self):
        remote = {'style_metric': score.STYLE_METRIC, 'style': 71, 'era_similarity_bounded': 71,
                  'body_words': 2000, 'appendix': {'cut': {'kind': 'appendix_command'}}}
        out = score.style_result(remote)
        self.assertEqual(out['body_words'], remote['body_words'])
        self.assertEqual(out['appendix'], remote['appendix'])
        self.assertIn('excluding appendices', out['style_description'])

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
        out = score.style_result({'style_metric': score.STYLE_METRIC, 'era_similarity_bounded': None, 'style': None})
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

    def breakdown(self):
        return {'version': 'style-breakdown-2',
                'attribution_order': 'text_features_then_wording_then_cap',
                'text_feature_penalty_points': 20, 'wording_penalty_points': 29,
                'other_penalty_points': 0, 'wording_cleared_style': 80}

    def test_target_breakdown_and_counterfactual_gap(self):
        out = score.style_result({'style_metric': score.STYLE_METRIC, 'style': 51, 'era_similarity_bounded': 51, 'style_breakdown': self.breakdown()})
        target = score.target_diagnostics(out, 85)
        self.assertEqual(target, {'status': 'ok', 'target': 85, 'score_reached': False,
                                 'gap': 34, 'wording_cleared_style': 80,
                                 'wording_only_gain_ceiling': 29, 'gap_after_wording_clear': 5})
        self.assertTrue(score.target_diagnostics(out, 51)['score_reached'])
        self.assertEqual(score.target_diagnostics(out, 0)['gap'], 0)
        self.assertEqual(score.target_diagnostics(out, 100)['gap_after_wording_clear'], 20)
        self.assertNotIn('accepted', target)  # score reached never certifies other safeguards

    def test_target_requires_breakdown_but_default_accepts_score_without_breakdown(self):
        for remote in ({'style_metric': score.STYLE_METRIC, 'style': 51, 'era_similarity_bounded': 51},
                       {'style_metric': score.STYLE_METRIC, 'style': 51, 'era_similarity_bounded': 51, 'style_breakdown': None},
                       {'style_metric': score.STYLE_METRIC, 'style': None, 'era_similarity_bounded': None, 'style_breakdown': None}):
            out = score.style_result(remote)
            self.assertEqual(score.target_diagnostics(out, 70)['status'], 'unavailable')

    def test_invalid_breakdown_rejected(self):
        bad = [None, {}, self.breakdown() | {'version': 'unknown'},
               self.breakdown() | {'attribution_order': 'unknown'},
               self.breakdown() | {'wording_penalty_points': -1},
               self.breakdown() | {'wording_penalty_points': True},
               self.breakdown() | {'wording_penalty_points': float('nan')},
               self.breakdown() | {'wording_penalty_points': float('inf')},
               self.breakdown() | {'wording_penalty_points': 28},
               self.breakdown() | {'wording_cleared_style': 50},
               self.breakdown() | {'wording_cleared_style': 81}]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                score.breakdown_result(data, 51)
        with self.assertRaises(ValueError):
            score.breakdown_result(self.breakdown(), None)

    def test_cap_is_not_reported_as_clearable_wording(self):
        data = self.breakdown() | {'other_penalty_points': 41, 'wording_cleared_style': 10}
        out = score.style_result({'style_metric': score.STYLE_METRIC, 'style': 10, 'era_similarity_bounded': 10, 'style_breakdown': data})
        self.assertEqual(score.target_diagnostics(out, 70)['wording_only_gain_ceiling'], 0)

    def test_cli_target_unavailable_and_invalid_without_upload(self):
        for value, expected in [('70', 3), ('nan', 1), ('inf', 1), ('-1', 1), ('101', 1)]:
            with self.subTest(value=value), \
                 patch.object(score.sys, 'argv', ['score.py', str(self.root), '--target-style', value]), \
                 patch.dict(score.os.environ, {}, clear=True), \
                 patch.object(score.urllib.request, 'urlopen') as fetch, \
                 patch('sys.stdout', io.StringIO()), patch('sys.stderr', io.StringIO()):
                self.assertEqual(score.main(), expected)
                fetch.assert_not_called()

    def test_cli_target_is_local_diagnostic_and_does_not_leak_target(self):
        for breakdown, expected in [(self.breakdown(), 0), (None, 3)]:
            response = io.BytesIO(json.dumps({'style_metric': score.STYLE_METRIC, 'style': 51, 'era_similarity_bounded': 51,
                                             'style_breakdown': breakdown}).encode())
            stdout = io.StringIO()
            with self.subTest(breakdown=breakdown), \
                 patch.object(score.sys, 'argv', ['score.py', str(self.root), '--target-style', '85']), \
                 patch.dict(score.os.environ, {'P2G_STYLE_URL': 'https://example.invalid/api/style'}, clear=True), \
                 patch.object(score.urllib.request, 'urlopen', return_value=response) as fetch, \
                 patch('sys.stdout', stdout):
                self.assertEqual(score.main(), expected)
            self.assertEqual(set(json.loads(fetch.call_args.args[0].data)), {'main', 'files', 'evaluate_expression'})
            self.assertEqual(json.loads(stdout.getvalue())['target']['status'], 'ok' if breakdown else 'unavailable')
