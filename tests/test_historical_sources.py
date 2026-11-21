"""Historical source identity is independent of current refactored modules."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class HistoricalSourceIntegrity(unittest.TestCase):
    def test_all_13_sha_exact_and_source_count(self):
        p=ROOT/'reproducibility/historical_verified/MANIFEST.json'
        ledger=json.loads(p.read_text(encoding='utf-8'))
        self.assertEqual(ledger['source_count'],13)
        for item in ledger['sources']:
            path=ROOT/item['relative_path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item['recorded_sha256'])
            self.assertEqual(item['historical_runtime_replay'],'NOT_RUN')

    def test_paper_b0_and_t2a_original_bytes(self):
        expected={'b0_dev_s42':'6d87aa2748b981d0257df64587f3ab264c3be4c8eb730fe6d85c794d5e820d29',
                  'b0_t2_noaug_s42':'b74f54c973a4bae6cf139ba65f6b16696d165e11654ea60b0cb4778795f2f84e'}
        for run,sha in expected.items():
            name=ROOT/'reproducibility/baseline_originals'/run/'train_baseline.py'
            self.assertEqual(hashlib.sha256(name.read_bytes()).hexdigest(),sha)

    def test_no_historical_replay_claim(self):
        p=ROOT/'docs/ARCHIVED_METRIC_PARITY_LOCAL.json'
        summary=json.loads(p.read_text(encoding='utf-8'))
        self.assertEqual(summary['status'],'ARCHIVED_TABLES_II_III_IV_V_VI_NUMERIC_PARITY_PASS')
        self.assertFalse(summary['test_pixels_opened_this_review'])
        self.assertIn('PENDING',summary['full_paper_numeric_parity'])

if __name__=='__main__':unittest.main()
