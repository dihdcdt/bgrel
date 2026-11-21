#!/usr/bin/env bash
# BG-REL V3.2 / R5 DEEP MERGE — verified workflow guide and safe read-only checks.
# Version: 2026-11-07.  No training, P0/P1 rebuild, or TEST inference is run automatically.
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

usage() {
cat <<'HELP'
BG-REL V3.2 — R5 DEPLOYED (29 active Python modules)

  bash run_pipeline.sh status              # Structure, file presence, live version
  bash run_pipeline.sh plan                # P0–P9 ordered commands; DISPLAY ONLY
  bash run_pipeline.sh assets              # Google Drive upload checklist, local presence
  bash run_pipeline.sh verify              # Frozen SHA, split, archived paper metrics
  bash run_pipeline.sh tests               # R5 local unit tests; no training or TEST pixels
  bash run_pipeline.sh check               # Run status + verify + tests once
  bash run_pipeline.sh preflight P2|P3|P5|P6|P7|P8|P9
  bash run_pipeline.sh help

Run from the BG-REL project root in Git Bash (Windows) or bash.
Optional: BGREL_ROOT=/path/to/bgrel bash run_pipeline.sh status

WARNING: preflight invokes existing program CLIs. Some can append QA/log files.
         The other modes do not decode DICOM/TEST pixels, train or overwrite data.
         The 'plan' mode DOES NOT RUN the printed historical commands.
HELP
}
err() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
message() { printf '%s\n' "$*"; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${BGREL_ROOT:-$SCRIPT_DIR}"
[[ -d "$ROOT/code" ]] || err "code/ not found: $ROOT"
ROOT="$(cd -- "$ROOT" && pwd -P)"
PY=""
find_python() {
    local p
    for p in "$ROOT/.venvs/bgrel/Scripts/python.exe" \
             "$ROOT/.venvs/bgrel/bin/python"; do
        if [[ -f "$p" ]]; then PY="$p"; return 0; fi
    done
    err "Missing Python venv at .venvs/bgrel. Refer to README, Environment. Do NOT copy a venv from someone else's PC."
}

status() {
    message "PROJECT_ROOT=$ROOT"
    if [[ -f "$ROOT/code/mask_qa.py" && -f "$ROOT/code/mask_pilots.py" && -f "$ROOT/code/train_t2a_noaug.py" ]]; then
        message 'SOURCE_LAYOUT=R5_STYLE (source checksum not checked by status)'
    else
        message 'SOURCE_LAYOUT=UNKNOWN_OR_INCOMPLETE'
    fi
    local files=(
      'code' 'tests' 'scripts' 'docs' 'reproducibility' 'run_pipeline.sh'
      'data/raw/RSNA Pneumonia/raw'
      'data/raw/pneumonia-challenge-dataset-mappings_2018.json'
      'data/raw/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json'
      'data/processed/master_manifest.csv' 'data/processed/boxes.csv'
      'data/processed/split.csv' 'data/processed/split.sha256'
      'data/lung_masks/train' 'models/pretrained/pspnet_chestxray_best_model_4.pth'
      'results/qa/sampler_manifest.json'
      'results/runs/b0_dev_s42/best.pt'
      'results/runs/b1_boxpos_s42/best.pt'
      'results/runs/a3_l100_s42/best.pt'
      'results/evaluation/final_test/protocol_freeze.json'
      'results/evaluation/final_test/manifest.json'
    )
    local path
    for path in "${files[@]}"; do
        if [[ -e "$ROOT/$path" ]]; then printf 'PRESENT  %s\n' "$path";
        else printf 'MISSING  %s\n' "$path"; fi
    done
    find_python
    message "PYTHON=$PY"
    "$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
files = sorted((root / 'code').glob('*.py'))
print('ACTIVE_PYTHON_FILES=' + str(len(files)))
print('B0_RUNNER=' + ('separate_from_T2A' if (root/'code/train_baseline.py').is_file() and (root/'code/train_t2a_noaug.py').is_file() else 'UNVERIFIED'))
print('STATUS_ONLY=TRUE; NO_DICOM_PIXEL_READ=TRUE; NO_TRAINING=TRUE')
PY
    message 'NOTE: This is a presence check, not a deployment certificate.'
}

plan() {
cat <<'PLAN'
BG-REL V3.2 — R5 RESEARCH SEQUENCE (DISPLAY ONLY, NOT EXECUTED)

IMPORTANT: The published study is complete and frozen. Commands with --run,
--generate, --train, --freeze must NEVER be copied into the current project.
For fresh research, create a separate workspace, output/run IDs, approvals,
fully installed dependencies and a separately sealed TEST protocol. The R5
historical run guards may block fresh training: a fresh-machine full run has
NOT been validated and is NOT enabled by this script.

P0  Data audit (DICOM + 2 adjudicated JSON; writes processed CSV)
    python code/audit_data.py
P1  Patient-disjoint split (preview -> review -> freeze; seed 42)
    python code/split_patients.py --preview --seed 42
    python code/split_patients.py --freeze --seed 42

P2  PSPNet provisional lung masks and QA (TRAIN only, human QA gate)
    python code/generate_lung_masks.py --smoke --count 8 --device cuda
    python code/mask_pilots.py --lung-pilot --device cuda
    python code/mask_qa.py --prepare
    python code/mask_qa.py --qa --device cuda
    python code/mask_policy.py --qa --device cuda
    python code/mask_policy.py --audit-qa200
    python code/mask_pilots.py --policy-pilot --device cuda
    [REVIEW QA / RESEARCHER APPROVAL]
    python code/build_train_lung_masks.py --preflight
    python code/build_train_lung_masks.py --generate --accept-provisional --device cuda

P3  Deterministic within-image matching (K=0/1/2)
    python code/sample_controls.py --preflight
    python code/sample_controls.py --run

P4/P5  B0 BCE baseline (R5: separate from T2-A no augmentation)
    python code/train_baseline.py --preflight --device cuda
    python code/train_baseline.py --smoke --device cuda
    python code/train_baseline.py --train --device cuda
    [OPTIONAL, NOT PAPER B0: python code/train_t2a_noaug.py --preflight --device cuda]

P6  A3 BG-REL (requires documented approval; lambda max 0.10)
    python code/p6_preflight.py --preflight
    python code/p6_preflight.py --two-batch-smoke
    python code/train_bgrel.py --preflight --lambda-max 0.10 --seed 42
    python code/train_bgrel.py --dry-run --lambda-max 0.10 --seed 42
    python code/train_bgrel.py --train --lambda-max 0.10 --seed 42 --approval PATH_TO_NEW_APPROVAL.json

P7  B1 positive-box BCE (A2 optional, not reported in paper)
    python code/train_ablations.py --variant b1 --preflight
    python code/train_ablations.py --variant b1 --smoke
    python code/train_ablations.py --variant b1 --train

P8a VAL image metrics / patient-cluster bootstrap (2,969 images)
    python code/evaluate_p8a.py --preflight --device cuda
    python code/evaluate_p8a.py --run --device cuda
    python code/evaluate_p8a.py --bootstrap-preflight --bootstrap 2000 --seed 42
    python code/evaluate_p8a.py --bootstrap-run --bootstrap 2000 --seed 42
    python code/evaluate_b1_val.py --preflight --device cuda
    python code/evaluate_b1_val.py --run --device cuda

P8b VAL regional evaluation (344 eligible lesions, 168 patients)
    python code/evaluate_regions.py --preflight --device cuda
    python code/evaluate_regions.py --run --device cuda
    python code/evaluate_b1_regions.py --preflight --device cuda
    python code/evaluate_b1_regions.py --run --device cuda
    python code/evaluate_lesion.py --preflight --device cuda
    python code/evaluate_lesion.py --run --device cuda

P9  Frozen internal TEST (2,967 images, FP32, pre-registered, ONE-TIME)
    python code/evaluate_final_test.py --preflight --device cuda
    python code/evaluate_final_test.py --run --device cuda
    [CURRENT FROZEN STUDY: use ONLY]
    python code/evaluate_final_test.py --verify-output

CURRENT PROJECT: use 'status', 'verify', 'assets', 'tests' or 'check'.
No mode in run_pipeline.sh performs P0/P1 or training/TEST inference.
PLAN
}

assets() {
cat <<'ASSET'
GOOGLE DRIVE CHECKLIST (source tree first; all paths relative to BG-REL root)

01_SOURCE: code/ scripts/ tests/ reproducibility/ docs/ README.md
           run_pipeline.sh requirements.txt .gitignore; optional paper PDF.
02_FROZEN: data/processed/{master_manifest.csv,split.csv,boxes.csv,split.sha256}
           data/raw/{two original adjudicated JSON files} (check redistribution).
03_RAW:    data/raw/RSNA Pneumonia/raw/ (30,000 DICOM source images).
           Do not publicly redistribute without verifying the original license/terms.
04_MASKS:  data/lung_masks/train/ (derived TRAIN lung masks, *.npz).
05_MODELS: models/pretrained/pspnet_chestxray_best_model_4.pth
           results/runs/{b0_dev_s42,b1_boxpos_s42,a3_l100_s42}/
           Include each best.pt + config/summary/metrics/log.
06_RESULTS:results/qa/ results/evaluation/ results/reports/ results/logs/
           Include evaluation predictions only for appropriate authorized sharing.

NEVER upload: .venvs/  __pycache__/  *.pyc  temporary stages, credentials,
             unrelated backups or local personal paths.
Keep exact relative paths; use ZIP chunks if uploading large trees. See DRIVE_UPLOAD_GUIDE.md.
ASSET
    local path
    for path in \
       'code' 'scripts' 'tests' 'reproducibility' 'docs' 'README.md' \
       'run_pipeline.sh' 'requirements.txt' \
       'data/processed/master_manifest.csv' 'data/processed/split.csv' \
       'data/processed/boxes.csv' 'data/processed/split.sha256' \
       'data/raw/pneumonia-challenge-dataset-mappings_2018.json' \
       'data/raw/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json' \
       'data/raw/RSNA Pneumonia/raw' 'data/lung_masks/train' \
       'models/pretrained/pspnet_chestxray_best_model_4.pth' \
       'results/runs/b0_dev_s42/best.pt' \
       'results/runs/b1_boxpos_s42/best.pt' \
       'results/runs/a3_l100_s42/best.pt' \
       'results/qa' 'results/evaluation' 'results/reports' 'results/logs'; do
        if [[ -e "$ROOT/$path" ]]; then printf 'PRESENT  %s\n' "$path";
        else printf 'MISSING  %s\n' "$path"; fi
    done
    message 'ASSET_CHECK_ONLY=TRUE; NO_COPY=TRUE; NO_DICOM_OPEN=TRUE'
}

verify() {
    find_python
    "$PY" - "$ROOT" <<'PY'
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
root=Path(sys.argv[1]); failures=[]

def record(ok, label, detail=''):
    print(('PASS' if ok else 'FAIL')+' | '+label+((' | '+str(detail)) if detail else ''))
    if not ok: failures.append(label)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

expected={
 'data/processed/master_manifest.csv':'4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b',
 'data/processed/split.csv':'ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7',
 'data/processed/boxes.csv':'6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45',
 'models/pretrained/pspnet_chestxray_best_model_4.pth':'019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc',
 'results/runs/b0_dev_s42/best.pt':'f6bda8a74c5d72b948a9ee4ba35cf53c1c0176e7d15dbb457ecc78a4d79f2929',
 'results/runs/b1_boxpos_s42/best.pt':'f1b2ab19211d22dc4ebe53cb978d36e15fa436b58e16f389954f5d2da0e688f1',
 'results/runs/a3_l100_s42/best.pt':'7bd87ba3841f0b8c8287a859dc3c942027e6fb318b2d5122dbfe1321bca2743f',
}
for rel, digest in expected.items():
    p=root/rel
    record(p.is_file() and sha256(p)==digest, 'SHA256 '+rel)

for name in ['pneumonia-challenge-dataset-mappings_2018.json',
             'pneumonia-challenge-annotations-adjudicated-kaggle_2018.json']:
    record((root/'data/raw'/name).is_file(), 'RAW JSON '+name)
record((root/'data/raw/RSNA Pneumonia/raw').is_dir(), 'DICOM tree exists (pixels not opened)')
seal=root/'data/processed/split.sha256'
record(seal.is_file() and seal.read_text(encoding='utf-8').split()[0]==expected['data/processed/split.csv'], 'frozen split seal')

sp=root/'data/processed/split.csv'
if sp.is_file():
    try:
        with sp.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        counts=Counter(r['split'].lower() for r in rows)
        patients=defaultdict(set)
        ids=[]
        for r in rows:
            patients[r['patient_id']].add(r['split'].lower())
            ids.append(r['image_id'])
        record(counts=={'train':23748,'val':2969,'test':2967},'split image counts',str(dict(counts)))
        record(len(ids)==29684 and len(set(ids))==29684,'unique image IDs')
        record(len(patients)==12233 and all(len(s)==1 for s in patients.values()),'patient-disjoint',len(patients))
    except (OSError,KeyError,ValueError,UnicodeError) as exc:
        record(False,'split CSV schema',type(exc).__name__)

metrics={
 'p8a_val':{'b0_dev_s42':(.880948,.703314),'a3_l100_s42':(.881102,.700887)},
 'b1_val':{'b1_boxpos_s42':(.881258,.706462)},
 'final_test':{'b0_dev_s42':(.878775,.717998),'b1_boxpos_s42':(.877135,.709789),'a3_l100_s42':(.878496,.712560)}
}
for group, runs in metrics.items():
    for run,(auc,ap) in runs.items():
        p=root/'results/evaluation'/group/run/'metrics.json'
        try:
            m=json.loads(p.read_text(encoding='utf-8'))
            ok=(abs(float(m['auroc'])-auc)<5e-7 and abs(float(m['average_precision'])-ap)<5e-7
                and m.get('n_images')==(2967 if group=='final_test' else 2969))
        except (OSError,KeyError,TypeError,ValueError): ok=False
        record(ok,'archived metrics '+group+'/'+run)
for rel in ['results/qa/sampler_manifest.json','results/evaluation/p8a_val/manifest.json',
            'results/evaluation/b1_val/manifest.json','results/evaluation/final_test/manifest.json',
            'results/evaluation/final_test/protocol_freeze.json']:
    p=root/rel
    try: ok=isinstance(json.loads(p.read_text(encoding='utf-8')),dict)
    except (OSError,ValueError,UnicodeError): ok=False
    record(ok,'archived JSON '+rel)

print('VERIFY_SUMMARY failures='+str(len(failures))+' test_pixels_opened=False training_started=False')
raise SystemExit(bool(failures))
PY
}

tests() {
    find_python
    message 'Running R5 source tests (no DICOM/TEST inference, no training).'
    ( cd "$ROOT" && PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" -m unittest discover -s tests -v )
}

preflight() {
    [[ $# -eq 1 ]] || err 'Specify exactly one phase: P2 P3 P5 P6 P7 P8 P9'
    find_python
    local phase="${1^^}"
    message 'ATTENTION: existing preflight CLIs may append project QA/log artifacts.'
    case "$phase" in
        P2) "$PY" "$ROOT/code/build_train_lung_masks.py" --preflight ;;
        P3) "$PY" "$ROOT/code/sample_controls.py" --preflight ;;
        P5) "$PY" "$ROOT/code/train_baseline.py" --preflight --device cpu ;;
        P6) "$PY" "$ROOT/code/p6_preflight.py" --preflight ;;
        P7) "$PY" "$ROOT/code/train_ablations.py" --variant b1 --preflight ;;
        P8) "$PY" "$ROOT/code/evaluate_p8a.py" --preflight --device cpu ;;
        P9) "$PY" "$ROOT/code/evaluate_final_test.py" --verify-output ;;
        *) err 'Allowed: P2 P3 P5 P6 P7 P8 P9. P0/P1 frozen and blocked.' ;;
    esac
}

case "${1:-help}" in
    help|-h|--help) usage ;;
    status) status ;;
    plan) plan ;;
    assets) assets ;;
    verify) verify ;;
    tests) tests ;;
    check) status; verify; tests; message 'LOCAL_R5_CHECK_COMPLETE=TRUE' ;;
    preflight) shift; preflight "$@" ;;
    *) usage; err "Unknown command: $1" ;;
esac
