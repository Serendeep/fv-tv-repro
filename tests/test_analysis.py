"""Regression checks for result selection and the paper's grading rule."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import analyze
from fvtv.stats import cluster_bootstrap_ci


class AnalysisRegressionTests(unittest.TestCase):
    def test_diagnostic_rows_cannot_replace_main_grid_measurements(self):
        base = dict(model='m', task='t', seed=0, method='tv', layer=2, recovery_ratio=.4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            (root / 'results/grid_m_shard0.json').write_text(json.dumps([base]))
            for suffix in ('tv_probe_t_s0', 'fv_targeted_t', 'tvextra', 'crosstask'):
                (root / f'results/grid_m_{suffix}.json').write_text(json.dumps([
                    dict(base, recovery_ratio=999), dict(base, task='diagnostic_only')]))
            with patch.object(analyze, 'ROOT', root):
                self.assertEqual(analyze.load_rows(), [base])

    def test_large_effect_without_positive_paired_interval_does_not_pass(self):
        self.assertEqual(analyze.verdict(2.4, -.01, 14), 'FAIL')
        self.assertEqual(analyze.verdict(2.4, 0., 14), 'FAIL')
        self.assertEqual(analyze.verdict(.78, .12, 14), 'WEAK PASS')
        self.assertEqual(analyze.verdict(.8, .12, 14), 'PASS')
        self.assertEqual(analyze.verdict(.49, .12, 14), 'FAIL')
        self.assertEqual(analyze.verdict(2.4, .12, 4), 'INSUFFICIENT DATA')

    def test_bootstrap_does_not_depend_on_dictionary_or_seed_order(self):
        a = {'b': [.9, .3], 'a': [-.1, .4, .8]}
        b = {'a': [.8, -.1, .4], 'b': [.3, .9]}
        self.assertEqual(cluster_bootstrap_ci(a), cluster_bootstrap_ci(b))


if __name__ == '__main__':
    unittest.main()
