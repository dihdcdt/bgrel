"""Version identity and archived-run protection, using synthetic fixtures only."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import Dataset,RandomSampler
PACKAGE=Path(__file__).resolve().parents[1]
CODE=Path(os.environ.get('BGREL_CODE',PACKAGE/'code'))
sys.path.insert(0,str(CODE))

class BaselineVersions(unittest.TestCase):
    def test_both_original_trainers_preserved(self):
        expected={'b0_dev_s42':'6d87aa2748b981d0257df64587f3ab264c3be4c8eb730fe6d85c794d5e820d29',
                  'b0_t2_noaug_s42':'b74f54c973a4bae6cf139ba65f6b16696d165e11654ea60b0cb4778795f2f84e'}
        for run,h in expected.items():
            p=PACKAGE/'reproducibility/baseline_originals'/run/'train_baseline.py'
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),h)

    def test_current_versions_match_explicit_ledger(self):
        from refactor_helpers import verify_current_baseline_layout
        result=verify_current_baseline_layout(CODE.parent)
        ledger=json.loads((PACKAGE/'reproducibility/BASELINE_SOURCE_VERSIONS.json').read_text())
        for name,v in result['current'].items():self.assertEqual(v['sha256'],ledger['current'][name]['sha256'])
        self.assertFalse(result['historical_hashes_rewritten'])
        self.assertEqual(result['historical_runtime_replay'],'PENDING')

    def test_current_source_drift_is_fatal(self):
        from refactor_helpers import verify_current_baseline_layout
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'code').mkdir()
            for name in ('train_baseline.py','train_t2a_noaug.py'):shutil.copy2(CODE/name,root/'code'/name)
            with (root/'code/train_baseline.py').open('a') as f:f.write('\n# unreviewed edit\n')
            with self.assertRaisesRegex(RuntimeError,'version drift'):verify_current_baseline_layout(root)

    def test_missing_runner_is_fatal(self):
        from refactor_helpers import verify_current_baseline_layout
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'source missing'):verify_current_baseline_layout(Path(d))

    def test_recipes_and_self_hash_names_separate(self):
        b0=importlib.import_module('train_baseline');t2=importlib.import_module('train_t2a_noaug')
        self.assertEqual(b0.RUN_ID,'b0_dev_s42');self.assertEqual(t2.RUN_ID,'b0_t2_noaug_s42')
        self.assertIn('train_baseline.py',b0.CODE_FILES);self.assertIn('train_t2a_noaug.py',t2.CODE_FILES)
        self.assertIn('train_baseline.py',t2.CODE_FILES)
        self.assertFalse(t2.AUGMENT_TRAIN)
        self.assertEqual(b0.B0_RECIPE.entrypoint,'train_baseline.py')
        self.assertEqual(b0.T2A_RECIPE.entrypoint,'train_t2a_noaug.py')
        self.assertTrue(b0.B0_RECIPE.augment_train)
        self.assertFalse(b0.T2A_RECIPE.augment_train)
        self.assertEqual(set(t2.code_hashes()),set(t2.CODE_FILES))
        for name in ('SEED','BATCH','LR','WEIGHT_DECAY','MAX_EPOCHS','PATIENCE','MIN_IMPROVEMENT'):
            self.assertEqual(getattr(b0,name),getattr(t2,name))

    def test_b0_augments_t2a_does_not_and_both_shuffle(self):
        class FakeDataset(Dataset):
            def __init__(self,root,rows,*,train,seed,epoch):self.train=train;self.rows=rows
            def __len__(self):return len(self.rows)
            def __getitem__(self,i):return torch.zeros(3,224,224),torch.tensor(float(i%2))
        for name,expected in (('train_baseline',True),('train_t2a_noaug',False)):
            mod=importlib.import_module(name)
            with patch.object(mod,'RSNAImageDataset',FakeDataset),patch.object(torch.cuda,'is_available',return_value=False):
                loader=mod.make_loader(Path('.'),[{},{}],train=True,epoch=1)
                self.assertEqual(loader.dataset.train,expected)
                self.assertIsInstance(loader.sampler,RandomSampler)
                self.assertEqual(loader.generator.initial_seed(),42+1009+1)
                val=mod.make_loader(Path('.'),[{},{}],train=False,epoch=0)
                self.assertFalse(val.dataset.train)

    def test_direct_training_and_resume_block_before_io(self):
        for name in ('train_baseline','train_t2a_noaug'):
            mod=importlib.import_module(name)
            with tempfile.TemporaryDirectory() as d:
                for resume in (False,True):
                    with self.assertRaisesRegex(RuntimeError,'ARCHIVED_RUN_PROTECTED'):
                        mod.run_training(Path(d),torch.device('cpu'),resume=resume)
                self.assertEqual(list(Path(d).iterdir()),[])

    def test_shared_train_and_validation_math_ast_unchanged(self):
        """The common gradient/optimizer and validation kernels are preserved verbatim."""
        old=ast.parse((PACKAGE/'reproducibility/baseline_originals/b0_dev_s42/train_baseline.py').read_text())
        new=ast.parse((CODE/'train_baseline.py').read_text())
        for function in ('train_epoch','validate_epoch','atomic_pt','atomic_metrics'):
            original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==function)
            refactored=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==function)
            self.assertEqual(ast.dump(original),ast.dump(refactored),function)

    def test_baseline_dispatch_has_no_implicit_augmentation_drift(self):
        import train_baseline as b0
        import train_t2a_noaug as t2
        self.assertEqual(b0.recipe_code_files(b0.B0_RECIPE),tuple(b0.CODE_FILES))
        self.assertEqual(b0.recipe_code_files(b0.T2A_RECIPE),tuple(t2.CODE_FILES))
        self.assertEqual(b0.B0_RECIPE.run_id,'b0_dev_s42')
        self.assertEqual(b0.T2A_RECIPE.run_id,'b0_t2_noaug_s42')
        self.assertEqual(b0.T2A_RECIPE.control_run_id,'b0_dev_s42')
        self.assertIs(t2.train_epoch,b0.train_epoch)
        self.assertIs(t2.validate_epoch,b0.validate_epoch)

    def test_guard_before_any_training_side_effect(self):
        import train_baseline as b0
        from unittest.mock import patch
        with patch.object(b0,'validate_metadata',side_effect=AssertionError('IO attempted')):
            for recipe in (b0.B0_RECIPE,b0.T2A_RECIPE):
                with self.assertRaisesRegex(RuntimeError,'ARCHIVED_RUN_PROTECTED'):
                    b0.run_training(Path('/must-not-be-touched'),torch.device('cpu'),resume=False,recipe=recipe)

    def test_p6_keeps_historical_dataset_guard(self):
        import p6_preflight
        from t3_train_integration import check_b0_recipe
        source=(CODE/'p6_preflight.py').read_text()
        self.assertNotIn('LEGACY_T2A_TRAINER_SHA',source)
        self.assertIn('current_baselines = verify_current_baseline_layout(ROOT)',source)
        config={'run_id':'b0_dev_s42','seed':42,'batch':4,'epochs_cap':20,'lr':3e-5,'weight_decay':1e-3,
            'optimizer':'AdamW','scheduler':'CosineAnnealingLR_per_optimizer_step','amp_precision':'bf16',
            'grad_scaler_enabled':False,'augment_train':'brightness_contrast_[0.9,1.1]',
            'validation_augment':False,'train_samples':23748,'val_samples':2969,'test_images_opened':False,
            'source_sha256':{'dataset.py':'synthetic_historical_sha','model.py':'synthetic_model_sha'}}
        with self.assertRaisesRegex(ValueError,'B0 dataset.py SHA mismatch'):
            check_b0_recipe(config,{},dataset_sha='synthetic_current_sha',model_sha='synthetic_model_sha')

if __name__=='__main__':unittest.main()
