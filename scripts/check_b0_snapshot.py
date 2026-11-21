#!/usr/bin/env python3
r"""Read-only B0 artifact/source provenance inspection; no imports of ML code.

Usage: python scripts/check_b0_snapshot.py --project-root <repo-root>
No writes, no train, no imports of pydicom/torch, no TEST access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HISTORICAL_B0_SHA = '6d87aa2748b981d0257df64587f3ab264c3be4c8eb730fe6d85c794d5e820d29'

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            digest.update(b)
    return digest.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--historic-source',type=Path,default=Path(__file__).resolve().parents[1]/'reproducibility/baseline_originals/b0_dev_s42/train_baseline.py')
    args=ap.parse_args();project=args.project_root
    good=True
    if not args.historic_source.is_file():
        print('FAIL missing historical B0 source:',args.historic_source);raise SystemExit(1)
    historical_hash=sha256(args.historic_source)
    print('HISTORICAL_B0_SOURCE',historical_hash,'PASS' if historical_hash==HISTORICAL_B0_SHA else 'FAIL')
    good &= historical_hash==HISTORICAL_B0_SHA
    run=project/'results/runs/b0_dev_s42'
    config_path,summary_path,checkpoint=run/'config.json',run/'summary.json',run/'best.pt'
    for f in (config_path,summary_path,checkpoint):
        if not f.is_file(): print('MISSING',f);return 1
    config=json.loads(config_path.read_text(encoding='utf-8'))
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    recorded=config.get('source_sha256',{}).get('train_baseline.py')
    print('B0_CONFIG_TRAINER_HASH',recorded,'PASS' if recorded==HISTORICAL_B0_SHA else 'FAIL')
    good &= recorded==HISTORICAL_B0_SHA
    saved_checkpoint_hash=summary.get('checkpoint_sha256')
    actual_checkpoint_hash=sha256(checkpoint)
    print('B0_CHECKPOINT_HASH',actual_checkpoint_hash,'PASS' if actual_checkpoint_hash==saved_checkpoint_hash else 'FAIL')
    good &= actual_checkpoint_hash==saved_checkpoint_hash
    print('B0_RUN_ID',config.get('run_id'),'PASS' if config.get('run_id')=='b0_dev_s42' else 'FAIL')
    good &= config.get('run_id')=='b0_dev_s42'
    print('B0_RECIPE_AUGMENT',config.get('augment_train'),'PASS' if config.get('augment_train')=='brightness_contrast_[0.9,1.1]' else 'FAIL')
    good &= config.get('augment_train')=='brightness_contrast_[0.9,1.1]'
    # Source drift is NOT a failure of the frozen experiment; do not confuse active with historical bytes.
    active=project/'code/train_baseline.py'
    if active.is_file():
        current=sha256(active)
        print('ACTIVE_TRAIN_BASELINE',current,'MATCHES_HISTORICAL' if current==HISTORICAL_B0_SHA else 'DIFFERS_FROM_HISTORICAL_EXPECTED_FOR_R3')
    data=project/'code/dataset.py'
    if data.is_file():
        current=sha256(data)
        recorded_ds=config.get('source_sha256',{}).get('dataset.py')
        print('DATASET_SOURCE',current,'RECORD',recorded_ds,'MATCH' if current==recorded_ds else 'DRIFT_AFTER_R3_DO_NOT_REWRITE_HISTORY')
    print('READ_ONLY_B0_SNAPSHOT_CHECK', 'PASS' if good else 'FAIL')
    return 0 if good else 1

if __name__=='__main__':
    raise SystemExit(main())
