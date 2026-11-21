"""Behavioral and source-contract regression; all fixtures are synthetic."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[1]
CODE = Path(os.environ.get('BGREL_CODE', str(PACKAGE / 'code'))).resolve()
OLD = PACKAGE / 'reproducibility/r3_source'
sys.path.insert(0, str(CODE))
torch.set_num_threads(2)

def load_old(name):
    key = '_r3_' + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, OLD / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod

class CoreContracts(unittest.TestCase):
    def test_frozen_mathematical_sources_exact_bytes(self):
        for name in ('dataset','model','metrics','regions','losses','rel_adapter','sample_controls',
                     'evaluate_b1_val','evaluate_b1_regions','evaluate_lesion',
                     'evaluate_final_test','t3_train_integration'):
            with self.subTest(name=name):
                self.assertEqual((CODE/(name+'.py')).read_bytes(), (OLD/(name+'.py')).read_bytes())

    def test_recovered_b0_is_distinct_from_t2a(self):
        p=PACKAGE/'reproducibility/paper_b0/train_baseline.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
            '6d87aa2748b981d0257df64587f3ab264c3be4c8eb730fe6d85c794d5e820d29')
        import train_t2a_noaug as t2a
        self.assertEqual(t2a.EXPERIMENT,'T2-A'); self.assertFalse(t2a.AUGMENT_TRAIN)
        self.assertNotEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
                            hashlib.sha256((CODE/'train_baseline.py').read_bytes()).hexdigest())

    def test_path_configuration_and_mapping(self):
        import paths
        p=paths.get_audit_paths()
        self.assertEqual(p.root,CODE.parent)
        self.assertEqual(p.dicom_root,CODE.parent/'data/raw/RSNA Pneumonia/raw')
        self.assertEqual(p.mappings.name,'pneumonia-challenge-dataset-mappings_2018.json')
        self.assertEqual(p.annotations.name,'pneumonia-challenge-annotations-adjudicated-kaggle_2018.json')
        self.assertEqual(paths.migrated_dicom_relpath('RSNA Pneumonia/raw/study/series/a.dcm'),
                         Path('data/raw/RSNA Pneumonia/raw/study/series/a.dcm'))
        for bad in ('../a.dcm','C:/a.dcm','/a.dcm','RSNA Pneumonia/raw/../series/a.dcm','RSNA Pneumonia/raw/s/t/a.png'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paths.migrated_dicom_relpath(bad)

    def test_explicit_config_overrides(self):
        import paths
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            r=paths.get_audit_paths(p,p/'m.json',p/'a.json')
            self.assertEqual(r.dicom_root,p.resolve())
            self.assertEqual(r.mappings,(p/'m.json').resolve())

    def test_roi_analytic_fractional_area(self):
        from regions import fractional_roi_pool
        # ROI spans one full left cell and half a right cell: (1 + 2*0.5)/1.5.
        f=torch.tensor([[[[1.,2.],[3.,4.]]]],requires_grad=True)
        b=torch.tensor([[0.,0.,1.5,1.]],dtype=torch.float32)
        result=fractional_roi_pool(f,b,torch.tensor([0]),input_size=(2,2))
        self.assertAlmostEqual(float(result.detach()[0,0]),4/3,places=6)
        result.sum().backward()
        torch.testing.assert_close(f.grad,torch.tensor([[[[2/3,1/3],[0.,0.]]]]))

    def test_os16_and_shared_head(self):
        from model import MobileNetV2OS16
        from regions import shared_region_scores
        m=MobileNetV2OS16(pretrained=False).eval()
        with torch.no_grad():
            f=m.forward_features(torch.zeros(1,3,224,224))
            self.assertEqual(tuple(f.shape),(1,1280,14,14))
            z=shared_region_scores(f,m.classifier,torch.tensor([[0.,0.,224.,224.]]),torch.tensor([0]))
            torch.testing.assert_close(z,m.classifier(f.mean((2,3))).flatten(),rtol=1e-5,atol=1e-7)

    def test_lambda_schedule(self):
        from p6_preflight import lambda_at_epoch
        self.assertEqual([round(lambda_at_epoch(i,.1),6) for i in (1,2,3,4)],[.02,.06,.1,.1])
        for i in (0,-1,True):
            with self.assertRaises(ValueError):lambda_at_epoch(i,.1)

    def test_listwise_stable_and_k0_bce(self):
        from losses import listwise_rel_loss,bce_rel_loss
        a=torch.tensor(1000.,requires_grad=True); c=torch.tensor([999.,998.],requires_grad=True)
        loss=listwise_rel_loss(a,c);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(c.grad).all())
        z=torch.tensor([.4,-.2],requires_grad=True);y=torch.tensor([1.,0.])
        p=bce_rel_loss(z,y,[[(torch.tensor(.1),torch.empty(0))],[]],lambda_value=.1)
        self.assertEqual(p.eligible_images,0);self.assertEqual(float(p.rel.detach()),0)
        p.total.backward();self.assertTrue((z.grad!=0).all())

    def test_sampler_determinism_and_constraints(self):
        from sample_controls import Lung,Rect,sample_lesion,intersect,iou
        left=np.zeros((1024,1024),bool);left[100:900,60:470]=True
        right=np.zeros_like(left);right[100:900,550:960]=True
        lungs={'left':Lung(left),'right':Lung(right)};lesion=Rect(150.25,300.75,200.25,350.75)
        a=sample_lesion('im','ann',lesion,[lesion],lungs)
        b=sample_lesion('im','ann',lesion,[lesion],lungs)
        self.assertEqual(a,b);self.assertIn(a['k'],(1,2))
        for r in a['accepted']:
            self.assertEqual(intersect(r['rect'],lesion),0);self.assertGreaterEqual(r['occupancy'],.8-1e-6)
            self.assertAlmostEqual(r['rect'].w,lesion.w)
        if a['k']==2:self.assertLessEqual(iou(*[r['rect'] for r in a['accepted']]),.2+1e-6)
        self.assertEqual(sample_lesion('im','ann',lesion,[lesion],None)['k'],0)

    def test_segmentation_channels_and_mask_roundtrip(self):
        import generate_lung_masks as g
        from mask_pilots import save_mask
        from sample_controls import load_lungs
        self.assertEqual((g.LEFT_INDEX,g.RIGHT_INDEX,g.CANDIDATE_THRESHOLD),(4,5,.5))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mask.npz';l=np.zeros((1024,1024),bool);r=l.copy()
            l[100:700,100:400]=True;r[100:700,600:900]=True
            h=save_mask(p,l,r);v=load_lungs(p,h)
            np.testing.assert_array_equal(v['left'].mask,l)
            with self.assertRaises(FileExistsError):save_mask(p,l,r)

    def test_intervention_same_pixels_and_rie(self):
        import evaluate_regions as e
        from sample_controls import Rect
        raw=np.linspace(0,1,1024*1024,dtype=np.float32).reshape(1024,1024)
        rect=Rect(5.2,6.1,7.3,9.4);blur=np.zeros_like(raw)
        mask=e.region_mask(rect);self.assertEqual(int(mask.sum()),12)
        for mode in ('blur','fill'):
            out=e.modify_image(raw,rect,mode,blur)
            np.testing.assert_array_equal(out[~mask],raw[~mask])
        # Mock predictions to verify Delta-L - mean Delta-C, including negative Delta-C.
        from unittest.mock import patch
        with patch.object(e,'probabilities',return_value=[.4,.7,.8]):
            v=e.intervention_metrics(None,raw,blur,rect,[rect,rect],.6,torch.device('cpu'))
        self.assertAlmostEqual(v['rie_blur'],.35)

class RefactorParity(unittest.TestCase):
    def test_extracted_objective_ast_unchanged(self):
        before=ast.parse((OLD/'train_ablations.py').read_text())
        after=ast.parse((CODE/'training_objectives.py').read_text())
        for name in ('collect_region_scores','compute_training_loss','candidate_axis','geometry_candidates','sample_geometry_controls','geometry_csv_text','text_sha256'):
            a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))

    def test_objective_values_gradients_and_optimizer_step(self):
        old=load_old('train_ablations');new=importlib.import_module('train_ablations')
        from sample_controls import Rect
        for variant in ('b1','a2'):
            for lam in (0.,.02,.06,.1):
                with self.subTest(variant=variant,lam=lam):
                    torch.manual_seed(72);f=torch.randn(3,5,4,4,requires_grad=True);head=nn.Linear(5,1)
                    rect=Rect(50.1,75.2,190.3,250.4);control=Rect(600.1,75.2,740.3,250.4)
                    rect2=Rect(60.,500.,160.,700.)
                    exp={'variant':variant,'boxes':{'i1':(('a1',rect),('a2',rect2))},
                         'geometry_index':{'i1':(('a1',rect,(control,)),('a2',rect2,(control,rect)))}}
                    y=torch.tensor([1.,1.,0.]);ids=['i1','k0','neg']
                    results=[]
                    for module in (old,new):
                        x=f.detach().clone().requires_grad_();h=copy.deepcopy(head)
                        opt=torch.optim.AdamW(h.parameters(),lr=3e-5,weight_decay=1e-3)
                        v=module.compute_training_loss(exp,ids,y,x,h,lambda_value=lam)
                        v[0].backward();grads=[x.grad.clone(),h.weight.grad.clone(),h.bias.grad.clone()]
                        opt.step();results.append((v,grads,h.state_dict()))
                    for a,b in zip(results[0][0][:3],results[1][0][:3]):self.assertTrue(torch.equal(a,b))
                    self.assertEqual(results[0][0][3:],results[1][0][3:])
                    for a,b in zip(results[0][1],results[1][1]):self.assertTrue(torch.equal(a,b))
                    for k in results[0][2]:self.assertTrue(torch.equal(results[0][2][k],results[1][2][k]))

    def test_geometry_candidates_and_serialization_parity(self):
        old=load_old('train_ablations');new=importlib.import_module('train_ablations')
        from sample_controls import Rect
        for rect in (Rect(50.2,100.3,100.7,220.8),Rect(650.,400.,800.,600.)):
            for tier in (1,2):
                self.assertEqual(old.geometry_candidates(rect,'im','ann',tier),
                                 new.geometry_candidates(rect,'im','ann',tier))
            self.assertEqual(old.sample_geometry_controls('im','ann',rect,[rect]),
                             new.sample_geometry_controls('im','ann',rect,[rect]))
        row={k:i for i,k in enumerate(new.GEOMETRY_FIELDS)}
        self.assertEqual(old.geometry_csv_text([row]),new.geometry_csv_text([row]))
        self.assertEqual(old.text_sha256(old.geometry_csv_text([row])),new.text_sha256(new.geometry_csv_text([row])))

    def test_p0_rebuild_refuses_frozen_project(self):
        import audit_data
        from paths import AuditPaths
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);processed=p/'processed';processed.mkdir();(processed/'split.sha256').write_text('sealed')
            paths=AuditPaths(p,p,p,p,processed,p,p)
            with self.assertRaisesRegex(RuntimeError,'frozen split seal'):
                audit_data.audit(paths,None,p/'report.json')
            self.assertFalse((p/'report.json').exists())

    def test_csv_bytes_and_interrupted_write_policy(self):
        for name,columns in (('train_ablations','METRIC_FIELDS'),('train_bgrel','METRIC_COLS')):
            old=load_old(name);new=importlib.import_module(name)
            fields=getattr(new,columns);row={k:(None if i==2 else i/10) for i,k in enumerate(fields)}
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);a=p/'old.csv';b=p/'new.csv'
                old.write_metrics(a,[row]);new.write_metrics(b,[row]);self.assertEqual(a.read_bytes(),b.read_bytes())
                tmp=b.with_name('new.csv.tmp');tmp.write_text('INTERRUPTED')
                with self.assertRaises(RuntimeError):new.write_metrics(b,[row])
                self.assertEqual(tmp.read_text(),'INTERRUPTED')

    def test_canonical_b0_paper_guard_still_rejects_t2a(self):
        from t3_train_integration import check_b0_recipe
        with self.assertRaisesRegex(ValueError,'B0 recipe mismatch'):
            check_b0_recipe({'run_id':'b0_t2_noaug_s42'},{},dataset_sha='x',model_sha='y')

    def test_p0_cli_refuses_frozen_project_without_logs_or_report(self):
        import audit_data
        from paths import AuditPaths
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);processed=p/'processed';processed.mkdir();(processed/'split.sha256').write_text('sealed')
            paths=AuditPaths(p,p,p,p,processed,p,p)
            args=SimpleNamespace(dicom_root=None,mappings=None,annotations=None,overwrite=True)
            with patch.object(audit_data,'parse_arguments',return_value=args), \
                 patch.object(audit_data,'get_audit_paths',return_value=paths), \
                 patch.object(audit_data,'setup_logger') as logger:
                self.assertEqual(audit_data.main(),2)
                logger.assert_not_called()
            self.assertEqual(set(x.name for x in p.iterdir()),{'processed'})

    def test_no_stale_settings_file_references(self):
        for p in CODE.glob('*.py'):
            self.assertNotIn('from settings import',p.read_text())
            self.assertNotIn('with_name("settings.py")',p.read_text())

    def test_parse_all_and_internal_imports(self):
        for p in CODE.glob('*.py'):
            ast.parse(p.read_text());importlib.import_module(p.stem)

if __name__=='__main__':unittest.main()
