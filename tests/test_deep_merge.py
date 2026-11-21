"""R5 source fusion regressions. No DICOM, trainer, or TEST pixels are opened."""
from __future__ import annotations
import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
CODE=Path(os.environ.get('BGREL_CODE',str(PACKAGE/'code')))
sys.path.insert(0,str(CODE))

class MergeSafety(unittest.TestCase):
    def test_active_modules_retired_and_core_remains(self):
        self.assertEqual(len(list(CODE.glob('*.py'))),29)
        for name in ('prepare_mask_qa','run_mask_qa','p8a_bootstrap','pilot_lung_masks','pilot_mask_policy','auto_mask_qa'):
            self.assertFalse((CODE/(name+'.py')).exists())
        for name in ('mask_qa','mask_pilots','mask_policy','evaluate_p8a','evaluate_final_test','sample_controls'):
            self.assertTrue((CODE/(name+'.py')).is_file())

    def test_archived_qa_producer_exact_source_hash(self):
        from refactor_helpers import verify_recorded_source_sha
        root=PACKAGE
        expected='1c0899e7f2eeb528f66ee5125c66e7ec2c74bf683e59bd39b41bcdb96f94e584'
        p=verify_recorded_source_sha(root,expected,'run_mask_qa.py')
        self.assertEqual(p,root/'reproducibility/pre_r3_source/run_mask_qa.py')
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected)
        with self.assertRaises(ValueError):verify_recorded_source_sha(root,'0'*64,'run_mask_qa.py')
        with self.assertRaises(ValueError):verify_recorded_source_sha(root,expected,'../run_mask_qa.py')

    def test_new_source_sha_version_identity(self):
        from refactor_helpers import verify_recorded_source_sha
        for legacy, current in (("prepare_mask_qa.py", "mask_qa.py"),
                                ("run_mask_qa.py", "mask_qa.py"),
                                ("auto_mask_qa.py", "mask_policy.py"),
                                ("pilot_lung_masks.py", "mask_pilots.py"),
                                ("pilot_mask_policy.py", "mask_pilots.py")):
            recorded=hashlib.sha256((CODE/current).read_bytes()).hexdigest()
            self.assertEqual(verify_recorded_source_sha(PACKAGE,recorded,legacy,current),CODE/current)

    def test_original_core_definitions_preserved(self):
        # Compare the original AST of public numerical/data functions, not a file-count assertion alone.
        old=PACKAGE/'reproducibility/r3_source'
        for previous, current, names in (
            ('prepare_mask_qa','mask_qa',('validate_metadata','select_sample','build_audit')),
            ('run_mask_qa','mask_qa',('output_locations','make_contact_sheets','validate_dicom_record')),
            ('evaluate_p8a','evaluate_p8a',('check_frozen_data','verify_run','evaluate','summarize_run')),
            ('auto_mask_qa','mask_policy',('largest_component','shape_features','preliminary_decision')),
            ('mask_policy','mask_policy',('decide_geometry','apply_known_review','parse_features')),
            ('pilot_lung_masks','mask_pilots',('choose_pilot','save_mask','rel_decision')),
            ('pilot_mask_policy','mask_pilots',('choose_fresh_train',)),
        ):
            original=ast.parse((old/(previous+'.py')).read_text())
            revised=ast.parse((CODE/(current+'.py')).read_text())
            for n in names:
                a=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name==n)
                b=next(x for x in revised.body if isinstance(x,ast.FunctionDef) and x.name==n)
                self.assertEqual(ast.dump(a),ast.dump(b),f'{previous}.{n}')

    def test_bootstrap_interval_parity(self):
        import evaluate_p8a
        from importlib.util import spec_from_file_location,module_from_spec
        old=PACKAGE/'reproducibility/r3_source/p8a_bootstrap.py'
        spec=spec_from_file_location('old_p8a_bootstrap',old)
        module=module_from_spec(spec);spec.loader.exec_module(module)
        for values in ([1.,2.,3.,4.],[0.,-.2,1.5,2.5],list(np.random.default_rng(42).normal(size=2000))):
            self.assertEqual(evaluate_p8a.bootstrap_interval(values),module.interval(values))

    def test_merged_cli_functions_present(self):
        import mask_qa,evaluate_p8a
        for fn in ('run_prepare','run_qa','validate_metadata','make_contact_sheets'):
            self.assertTrue(callable(getattr(mask_qa,fn)))
        for fn in ('bootstrap_check','bootstrap_load_predictions','bootstrap_interval','bootstrap_run'):
            self.assertTrue(callable(getattr(evaluate_p8a,fn)))

if __name__=='__main__':unittest.main()
