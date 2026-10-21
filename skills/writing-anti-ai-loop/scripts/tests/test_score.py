"""Source-bound ledgers cannot turn missing evidence or stale files into passing scores."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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
        self.assertEqual((actual['style'], actual['completed'], actual['coverage'], actual['uncertain']), (100, 1, .25, 1))

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
        self.assertIsNone(self.read()['style'])

    def test_source_outside_root_rejected_before_read(self):
        self.data['source_hashes']['../elsewhere.tex'] = 'bad'
        with self.assertRaises(ValueError): self.read()
