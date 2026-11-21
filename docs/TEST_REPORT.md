# R4 clean refactor — verification summary (2026-11-07)

## Inputs directly inspected

- Windows `BGREL_ONE_SHOT_EVIDENCE_20261107_055446_166280.zip`: 171 collected files, zero collector errors; 32 active Python files, complete archive predictions, run metadata, processed CSVs, QA summary and project tree. Does not contain DICOM pixel data or checkpoint bytes.
- `code(3).zip` / R3 snapshot and `BGREL_R4_COMPLETE_V2.zip`: byte and dependency comparison.
- `BGREL_R4_REVIEW_20261107_053814_6466b6.zip`: previous genuine Windows V2 review, 42 tests/23 CLI checks, real TRAIN/VAL and three-checkpoint inference parity, 13 SHA-identical sources recovered.
- The six-page final `AI2007_Group_6.pdf`, Tables I–VI and Methods/Results (not a source of executable test evidence).

## Verified in the CURRENT packaging session

| Item | Result | Evidence/limits |
|---|---|---|
| Python AST parse 33 active modules | PASS | Syntax verified; not a full runtime import under Windows |
| Synthetic CPU unit suite | **51/51 PASS** | Linux CPU; **pydicom replaced with an import-only stub because unavailable in this container**; no real DICOM decoded |
| Shared BCE train_epoch and validate_epoch AST | PASS | Unmodified mathematical bodies vs recovered original paper B0 source |
| Synthetic single-batch AdamW optimizer parity | PASS | Exact loss, model parameters and optimizer step as original B0 |
| B0 augmentation / T2-A no augmentation + TRAIN shuffle | PASS | Controlled dataset/loader tests; both legacy run IDs remain write-protected |
| B1/A2 objective + geometry parity | PASS | AST/gradients, fixed seed synthetic cases inherited and re-tested |
| Historical sources | **13/13 SHA match** | Source identity only; historical runtime replay **NOT_RUN** |
| Paper Tables II–VI from original archived predictions | **PASS** | AUROC/AP, matched VAL REM/Top1/RIE, paired regional CIs and 2,000 TEST bootstrap replicates; no TEST pixel access |
| New full workspace apply/rollback | PASS | Linux test on cloned **all 32 actual R3 sources**; backup restores exact original code, README and protects results sentinel |
| Failure injection | PASS | 3 interruption positions verified; automatic old-path restoration |
| B0 and T2-A new CLI `--help` | PASS | Linux CPU direct help checks; **all 23 new CLI checks on Windows still PENDING** |
| New R4 clean package real Windows TRAIN/VAL + checkpoints | **PENDING** | Installer will check; cannot claim from CPU synthetic tests or older V2 review |
| Historical training/evaluator full replay | **NOT_RUN** | Never claimed by modern source-only deployment; original source SHA and archived outputs preserved |
| R4 clean deployed on laptop | **NO** | Installer is user-controlled and refuses a failed or drifted project |

**Numeric evidence:** `docs/ARCHIVED_METRIC_PARITY_LOCAL.json` records all recomputed figures from archived CSVs, not medical pixels. The original archived prediction data and frozen SHA remain untouched.

## Source surface after refactor

- 32 source files, 12,052 lines in active R3 snapshot.
- 33 source files, 12,201 lines in pre-existing R4 V2 candidate.
- **33 source files, 11,962 lines** in this R4 clean candidate.
- Main additional improvement: T2-A wrapper is 55 lines and delegates to the B0 engine (the former T2-A duplicated ~350 lines); the B0 engine adds explicit recipe parameters and source hashing.
- No artificial reduction of P2/P3/P8/P9 producer entrypoints solely to hit a file quota.

## Safety and deploy gate

Use `scripts/INSTALL.ps1` only after reading `docs/DEPLOYMENT_SCOPE.md`; it runs one consolidated Windows verification BEFORE transactional cutover. Windows acceptance requires R3 hashes, frozen files, real TRAIN/VAL and checkpoint parity, archived numeric metrics, and 13 historical SHA sources. It does **not** invent a historical runtime replay. Protected `data/`, `models/`, `results/`, `.venvs/` are never altered by the transaction; frozen file SHA and results inventory are checked before and after.

All source and package checksums are in `PACKAGE_SHA256.json` (self-excluding). Do not modify files in the extracted candidate after packaging; changes cause the installer to refuse them.
