# R4 clean candidate change log

## Source behavior and line consolidation

1. **One BCE training engine**: `code/train_baseline.py` retains the B0 paper recipe, with explicit immutable variant config and common methods for both B0 and T2-A. `code/train_t2a_noaug.py` retains its historical experiment name and no-augmentation policy but delegates implementation to the same engine. Original paper B0 and R3 T2-A full trainer sources are archived byte-exact under `reproducibility/baseline_originals`.
2. **R4 V2 improvements preserved**: `settings.py` merged into `paths.py`; `training_objectives.py` extracted 7 B1/A2 objective/geometry functions; atomic metrics writer common to B1/A2 and A3; frozen P0 rebuild guard; baseline source identity versioned in `refactor_helpers.py`.
3. **Historical source recovery**: SHA-exact originals for 13 recorded paper/provenance source variants stored as immutable named records; every hash verified. Never relabel an original checkpoint with candidate source hashes.
4. **Documentation and CLI sequence**: updated comprehensive `README.md`, safe `run_pipeline.sh`, Windows regression/deploy/rollback wrappers, complete path mapping and results inventory.
5. **Crash-safe workspace source transaction**: backup of code+docs+tests+reproducibility outside the project, old-path rollback on errors. No modifications to frozen data, test pixels, checkpoints, predictions or paper output files.

## Preserved original behavior

- Core decoder, MobileNetV2-OS16, fractional ROI, REL, PSPNet mask policy, frozen sampler and P8/P9 evaluators are **byte-identical** to the R3 source unless listed as changed in `FILE_MAPPING.csv`.
- Three reported variants B0/B1/A3 and frozen patient split, lambda schedule, checkpoints, archived evaluation statistics are unchanged.
- `p8a_bootstrap.py` and `evaluate_lesion.py` remain available as independent CLIs; no unverified removal.

## Not claimed

- All producer modules have been merged into ~20 files.
- Source-level changes prove identical CUDA training trajectory without a real training replay.
- Historical SHA-protected evaluators will accept modified R4 files without an isolated replay environment.
- Old full training or TEST inference was executed during cleanup.
