"""Synthetic numeric and provenance tests for shared B0/T2-A training engine."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import train_baseline as new
import train_t2a_noaug as t2a

class DummyLogger:
    def info(self,*args,**kwargs):pass

class BaselineNumericalParity(unittest.TestCase):
    def setUp(self):
        source=ROOT/'reproducibility/baseline_originals/b0_dev_s42/train_baseline.py'
        spec=importlib.util.spec_from_file_location('_historical_b0_temp',source)
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        self.old=old

    def test_training_epoch_math_and_update_exact(self):
        loader=[(torch.tensor([[.25,.7],[.2,-.1],[.8,.9],[.1,.4]],dtype=torch.float32),
                 torch.tensor([0.,1.,1.,0.],dtype=torch.float32))]
        torch.manual_seed(123)
        model0=nn.Sequential(nn.Linear(2,3),nn.Tanh(),nn.Linear(3,1),nn.Flatten(0))
        model1=nn.Sequential(nn.Linear(2,3),nn.Tanh(),nn.Linear(3,1),nn.Flatten(0))
        model1.load_state_dict(model0.state_dict())
        outputs=[]
        for module,model in ((self.old,model0),(new,model1)):
            optim=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-3)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=20)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            outputs.append(module.train_epoch(model,loader,optim,sched,scaler,torch.device('cpu'),DummyLogger(),1))
        self.assertEqual(outputs[0],outputs[1])
        for p,q in zip(model0.parameters(),model1.parameters()):self.assertTrue(torch.equal(p,q))

    def test_validation_math_exact(self):
        data=[(torch.tensor([[.3,.5],[.1,.2]],dtype=torch.float32),torch.tensor([1.,0.]))]
        class Head(nn.Module):
            def forward(self,x):return x[:,0]-.3*x[:,1]
        self.assertEqual(self.old.validate_epoch(Head(),data,torch.device('cpu')),
                         new.validate_epoch(Head(),data,torch.device('cpu')))

    def test_t2a_source_hashes_track_common_engine(self):
        source_hashes=t2a.code_hashes()
        self.assertIn('train_baseline.py',source_hashes)
        self.assertIn('train_t2a_noaug.py',source_hashes)
        self.assertEqual(source_hashes['train_baseline.py'],new.code_hashes()['train_baseline.py'])

    def test_unified_recipe_immutable(self):
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):new.B0_RECIPE.run_id='changed'

if __name__=='__main__':unittest.main()
