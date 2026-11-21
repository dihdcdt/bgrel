"""BG-REL V3.2 P5/T2-A BCE baseline, augmentation-off controlled run.

Only this experiment writes b0_t2_noaug_s42; original B0 remains untouched.
Run --preflight, --smoke, then --train in order. Never loads TEST pixels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import RSNAImageDataset, load_frozen, sha256, decode_dicom, prepare_image
from model import MobileNetV2OS16,imagenet_weight_path
from metrics import binary_metrics

PROJECT_ROOT=Path(__file__).resolve().parents[1]
RUN_ID='b0_t2_noaug_s42'
SMOKE_FILENAME='bce_smoke_t2_noaug_s42.json'
EXPERIMENT='T2-A'
CONTROL_RUN_ID='b0_dev_s42'
AUGMENT_TRAIN=False
SEED=42
BATCH=4
LR=3e-5
WEIGHT_DECAY=1e-3
MAX_EPOCHS=20
PATIENCE=4
MIN_IMPROVEMENT=0.001
METRIC_COLUMNS=['epoch','train_bce','val_bce','val_auroc','val_ap','lr_last','optimizer_steps','best_epoch','best_val_auroc','patience_counter','seconds']
CODE_FILES=['dataset.py','model.py','metrics.py','train_baseline.py']


def code_hashes():return {name:sha256(PROJECT_ROOT/'code'/name) for name in CODE_FILES}


def init_logger(path:Path|None):
    logger=logging.getLogger('b0');logger.handlers.clear();logger.setLevel(logging.INFO);logger.propagate=False
    format=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    stream=logging.StreamHandler(sys.stdout);stream.setFormatter(format);logger.addHandler(stream)
    if path is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        handler=logging.FileHandler(path,mode='a',encoding='utf-8');handler.setFormatter(format);logger.addHandler(handler)
    return logger


def set_seed(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def ensure_precision_supported(device:torch.device)->str:
    """Require BF16 CUDA training to avoid FP16 dynamic-loss-scale overflow.

    BF16 is supported by the project target GPU (RTX 4050). Do not silently
    fall back to FP16 or change numerical precision without a recorded run.
    """
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 unavailable; abort rather than silently use FP16')
        return 'bf16'
    if device.type=='cpu':
        return 'fp32'
    raise ValueError(f'Unsupported device: {device}')


def validate_metadata(root:Path):
    train,val=load_frozen(root)
    if {r['image_id'] for r in train}&{r['image_id'] for r in val}:
        raise ValueError('Train/validation image overlap')
    if len({r['patient_id'] for r in train}&{r['patient_id'] for r in val}):
        raise ValueError('Train/validation patient overlap')
    for name,rows in [('train',train),('val',val)]:
        counts={target:sum(r['target']==target for r in rows) for target in ('0','1')}
        if min(counts.values())<=0:raise ValueError(f'{name} has only one class')
    return train,val


def validate_sample_pixels(root,train,val,*,n=2):
    for part,rows in [('train',train),('val',val)]:
        # Frozen lexicographic sample, tests actual DICOM/UID, no TEST pixels.
        for r in rows[:n]:
            im=decode_dicom(root,r)
            processed=prepare_image(im)
            if processed.shape!=(3,224,224) or not torch.isfinite(processed).all():
                raise ValueError(f'{part} preprocess failed: {r["image_id"]}')


def make_loader(root,rows,*,train:bool,epoch:int,augment_train:bool=AUGMENT_TRAIN):
    """Keep train shuffling, but disable photometric transforms for T2-A.

    The dataset's train flag controls only augmentation. DataLoader's train
    flag separately controls shuffling, preserving the original B0 order.
    """
    ds=RSNAImageDataset(root,rows,train=train and augment_train,seed=SEED,epoch=epoch)
    generator=torch.Generator().manual_seed(SEED+epoch*1009+int(train))
    return DataLoader(ds,batch_size=BATCH,shuffle=train,num_workers=0,
                      pin_memory=torch.cuda.is_available(),drop_last=False,generator=generator)


def smoke(root:Path,device:torch.device):
    train,val=validate_metadata(root)
    validate_sample_pixels(root,train,val,n=2)
    precision=ensure_precision_supported(device)
    set_seed(SEED)
    model=MobileNetV2OS16(pretrained=True).to(device)
    x,y=next(iter(make_loader(root,train[:4],train=True,epoch=1,augment_train=AUGMENT_TRAIN)))
    x=x.to(device);y=y.to(device)
    model.train()
    # Exercise the same BF16 autocast path used by full training (not FP32).
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=(precision=='bf16')):
        out=model(x)
        if out.shape!=(4,) or not torch.isfinite(out).all():raise ValueError('Unexpected logits')
        loss=nn.functional.binary_cross_entropy_with_logits(out,y)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is not None and not torch.isfinite(g).all() for g in grads):
        raise ValueError('Non-finite gradients')
    if model.classifier.weight.grad is None or model.classifier.weight.grad.abs().sum().item()==0:
        raise ValueError('Head has no gradient')
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0,error_if_nonfinite=True)
    if not torch.isfinite(grad_norm):raise FloatingPointError('Nonfinite BF16 smoke gradient norm')
    model.eval()
    with torch.inference_mode():
        vx,vy=next(iter(make_loader(root,val[:4],train=False,epoch=0)))
        vout=model(vx.to(device))
        if not torch.isfinite(vout).all():raise ValueError('Invalid validation logits')
    return {'status':'PASS','source_sha256':code_hashes(),'split_sha256':sha256(root/'data/processed/split.csv'),
            'master_sha256':sha256(root/'data/processed/master_manifest.csv'),
            'pretrained_sha256':model.pretrained_sha256,'device':str(device),
            'train_items':len(train),'val_items':len(val),'smoke_batch':4,
            'smoke_bce':float(loss.item()),'smoke_grad_norm':float(grad_norm.item()),
            'amp_precision':precision,'grad_scaler_enabled':False,
            'experiment':EXPERIMENT,'run_id':RUN_ID,'control_run_id':CONTROL_RUN_ID,
            'augment_train':'none' if not AUGMENT_TRAIN else 'brightness_contrast_[0.9,1.1]'}


def save_json_exclusive(path:Path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as fp:
        json.dump(record,fp,indent=2,ensure_ascii=False);fp.write('\n')


def atomic_pt(path:Path,record):
    temp=path.with_name(path.name+'.tmp')
    if temp.exists():raise RuntimeError(f'Existing interrupted checkpoint: {temp}')
    try:
        torch.save(record,temp)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def atomic_metrics(path:Path,rows:list[dict]):
    temp=path.with_name(path.name+'.tmp')
    if temp.exists():raise RuntimeError(f'Existing interrupted metrics: {temp}')
    with temp.open('x',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=METRIC_COLUMNS,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    os.replace(temp,path)


def train_epoch(model,loader,optim,sched,scaler,device,logger,epoch):
    """Train without skipped samples. BF16 is not loss-scaled.

    Every loss/gradient must be finite; a genuine nonfinite gradient is fatal.
    Scheduler advances only after a successful optimizer update.
    """
    model.train();total=0.;n=0;steps=0
    precision=ensure_precision_supported(device)
    if scaler.is_enabled():
        raise RuntimeError('BF16/FP32 training must not use FP16 GradScaler')
    for i,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=(precision=='bf16')):
            z=model(x)
            if not torch.isfinite(z).all():
                raise FloatingPointError(f'Nonfinite train logits at epoch={epoch}, batch={i}')
            loss=nn.functional.binary_cross_entropy_with_logits(z,y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite train loss at epoch={epoch}, batch={i}')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0,error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient norm at epoch={epoch}, batch={i}')
        optim.step()
        sched.step();steps+=1
        total+=float(loss.item())*len(y);n+=len(y)
        if i%500==0 or i==len(loader):
            logger.info('Epoch %d train %d/%d BCE=%.6f LR=%.8g',epoch,i,len(loader),total/n,optim.param_groups[0]['lr'])
    return total/n,steps


@torch.inference_mode()
def validate_epoch(model,loader,device):
    model.eval();loss_sum=0.;n=0;ys=[];zs=[]
    for x,y in loader:
        z=model(x.to(device)).float().cpu()
        if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite validation logits')
        loss_sum+=float(nn.functional.binary_cross_entropy_with_logits(z,y,reduction='sum').item())
        n+=len(y);ys.extend(y.tolist());zs.extend(z.tolist())
    roc,ap=binary_metrics(ys,zs)
    return loss_sum/n,roc,ap


def run_training(root:Path,device:torch.device,*,resume:bool):
    train,val=validate_metadata(root)
    precision=ensure_precision_supported(device)
    smoke_path=root/'results/qa'/SMOKE_FILENAME
    if not smoke_path.is_file():raise RuntimeError('Required BCE CUDA smoke artifact missing. Run --smoke first.')
    proof=json.loads(smoke_path.read_text(encoding='utf-8'))
    if (proof.get('status')!='PASS' or proof.get('source_sha256')!=code_hashes()
            or proof.get('split_sha256')!=sha256(root/'data/processed/split.csv')
            or proof.get('master_sha256')!=sha256(root/'data/processed/master_manifest.csv')
            or proof.get('device')!=str(device)
            or proof.get('experiment')!=EXPERIMENT or proof.get('run_id')!=RUN_ID
            or proof.get('control_run_id')!=CONTROL_RUN_ID
            or proof.get('augment_train')!=('none' if not AUGMENT_TRAIN else 'brightness_contrast_[0.9,1.1]')
            or proof.get('amp_precision')!=precision or proof.get('grad_scaler_enabled') is not False):
        raise RuntimeError('Stale smoke proof or changed source; rerun smoke with new artifact version')
    set_seed(SEED)
    out=root/'results/runs'/RUN_ID
    if resume:
        if not out.is_dir():raise RuntimeError('Cannot resume: run directory missing')
    else:
        out.mkdir(parents=True,exist_ok=False)
    logger=init_logger(out/'train.log')
    logger.info('B0 START id=%s experiment=%s control=%s augmentation=%s device=%s resume=%s',
                RUN_ID,EXPERIMENT,CONTROL_RUN_ID,AUGMENT_TRAIN,device,resume)
    model=MobileNetV2OS16(pretrained=True).to(device)
    if model.pretrained_sha256!=proof['pretrained_sha256']:
        raise RuntimeError('Pretrained source changed')
    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    steps_per_epoch=math.ceil(len(train)/BATCH)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=steps_per_epoch*MAX_EPOCHS)
    # BF16 has FP32-like exponent range and needs no dynamic loss scaling.
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    config={'phase':'P5','experiment':EXPERIMENT,'control_run_id':CONTROL_RUN_ID,
            'model':'MobileNetV2_OS16_BCE','run_id':RUN_ID,'seed':SEED,'batch':BATCH,'epochs_cap':MAX_EPOCHS,
            'patience':PATIENCE,'min_improvement':MIN_IMPROVEMENT,'lr':LR,'weight_decay':WEIGHT_DECAY,
            'optimizer':'AdamW','scheduler':'CosineAnnealingLR_per_optimizer_step','amp':device.type=='cuda',
            'amp_precision':precision,'grad_scaler_enabled':False,
            'augment_train':'none' if not AUGMENT_TRAIN else 'brightness_contrast_[0.9,1.1]',
            'validation_augment':False,'pretrained_sha256':model.pretrained_sha256,
            'source_sha256':code_hashes(),'split_sha256':proof['split_sha256'],'master_sha256':proof['master_sha256'],
            'test_images_opened':False,'train_samples':len(train),'val_samples':len(val)}
    conf=out/'config.json'
    if not resume:save_json_exclusive(conf,config)
    elif json.loads(conf.read_text(encoding='utf-8'))!=config:raise RuntimeError('Resume config mismatch')
    rows=[];start=1;best=-float('inf');best_epoch=0;patience=0;best_patience_auc=-float('inf')
    if resume:
        last=out/'last.pt'
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state.get('config')!=config:raise RuntimeError('Checkpoint provenance changed')
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start=int(state['epoch'])+1;best=float(state['best_auc']);best_epoch=int(state['best_epoch']);patience=int(state['patience'])
        best_patience_auc=float(state['best_patience_auc'])
        rows=state['metrics']
        if start>MAX_EPOCHS or patience>=PATIENCE:raise RuntimeError('Run already completed; no resume necessary')
        if not (out/'metrics.csv').is_file():raise RuntimeError('Missing metrics on resume')
        with (out/'metrics.csv').open(newline='',encoding='utf-8') as f:real=list(csv.DictReader(f))
        if len(real)!=len(rows) or int(real[-1]['epoch'])!=start-1:
            raise RuntimeError('Metrics/checkpoint mismatch: preserve for audit')
        logger.info('RESUME epoch=%d best_epoch=%d',start,best_epoch)
    for epoch in range(start,MAX_EPOCHS+1):
        t=time.monotonic()
        loader=make_loader(root,train,train=True,epoch=epoch,augment_train=AUGMENT_TRAIN)
        train_bce,steps=train_epoch(model,loader,optim,sched,scaler,device,logger,epoch)
        val_bce,auc,ap=validate_epoch(model,make_loader(root,val,train=False,epoch=0),device)
        if auc>best+1e-12:
            best=auc;best_epoch=epoch
            atomic_pt(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'val_auc':auc,'val_ap':ap,'config':config})
        if auc>best_patience_auc+MIN_IMPROVEMENT:
            best_patience_auc=auc;patience=0
        else:patience+=1
        r={'epoch':epoch,'train_bce':f'{train_bce:.8f}','val_bce':f'{val_bce:.8f}','val_auroc':f'{auc:.8f}',
           'val_ap':f'{ap:.8f}','lr_last':f'{optim.param_groups[0]["lr"]:.12g}','optimizer_steps':steps,
           'best_epoch':best_epoch,'best_val_auroc':f'{best:.8f}','patience_counter':patience,
           'seconds':f'{time.monotonic()-t:.2f}'}
        rows.append(r)
        state={'model':model.state_dict(),'optimizer':optim.state_dict(),'scheduler':sched.state_dict(),
               'scaler':scaler.state_dict(),'epoch':epoch,'best_auc':best,'best_epoch':best_epoch,
               'patience':patience,'best_patience_auc':best_patience_auc,'metrics':rows,'config':config}
        atomic_pt(out/'last.pt',state)
        atomic_metrics(out/'metrics.csv',rows)
        logger.info('EPOCH %d train_BCE=%.5f val_BCE=%.5f AUC=%.6f AP=%.6f best=%d patience=%d time=%.1fs',
                    epoch,train_bce,val_bce,auc,ap,best_epoch,patience,time.monotonic()-t)
        if patience>=PATIENCE:
            logger.info('EARLY STOP at epoch=%d',epoch);break
    result={'status':'FINISHED','best_epoch':best_epoch,'best_val_auroc':best,
            'last_epoch':epoch,'checkpoint_sha256':sha256(out/'best.pt'),
            'metrics_sha256':sha256(out/'metrics.csv'),'source_sha256':code_hashes(),'test_images_opened':False}
    final=out/'summary.json'
    if final.exists():raise RuntimeError('Final summary already exists')
    save_json_exclusive(final,result)
    logger.info('B0 COMPLETE best_epoch=%d best_AUC=%.6f',best_epoch,best)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--smoke',action='store_true');mode.add_argument('--train',action='store_true')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.resume and not args.train:ap.error('--resume only valid with --train')
    if RUN_ID==CONTROL_RUN_ID:
        raise RuntimeError('T2-A must not reuse the frozen B0 control run ID')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    root=PROJECT_ROOT;device=torch.device(args.device)
    if args.preflight:
        precision=ensure_precision_supported(device)
        train,val=validate_metadata(root)
        validate_sample_pixels(root,train,val,n=2)
        print('B0 T2-A PREFLIGHT PASS',{'run_id':RUN_ID,'experiment':EXPERIMENT,
                                        'control_run_id':CONTROL_RUN_ID,
                                        'augment_train':AUGMENT_TRAIN,
                                        'train':len(train),'val':len(val),
                                        'pretrained_available':imagenet_weight_path().is_file(),
                                        'pretrained_path':str(imagenet_weight_path()),'amp_precision':precision})
    elif args.smoke:
        artifact=root/'results/qa'/SMOKE_FILENAME
        if artifact.exists():raise RuntimeError('Smoke proof exists; stop, do not overwrite')
        record=smoke(root,device)
        save_json_exclusive(artifact,record)
        print('B0 T2-A SMOKE PASS',record)
    else:run_training(root,device,resume=args.resume)

if __name__=='__main__':
    main()
