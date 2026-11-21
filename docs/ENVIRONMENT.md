# Environment snapshot and setup caution

Source: ONE_SHOT evidence collected on Windows 11, 07 Nov 2026 (not reinstalled or changed by this package).

- Python 3.12.0, `.venvs/bgrel/Scripts/python.exe`
- torch 2.9.1+cu128; torchvision 0.24.1+cu128; CUDA 12.8 observed in older logs
- numpy 1.26.4; scipy 1.17.1; scikit-learn 1.7.2
- pydicom 3.0.1; TorchXRayVision 1.5.7; pandas 2.3.3; Pillow 11.3.0

`requirements.txt` in the LIVE project was **P0-only** (numpy/pydicom/Pillow), not a complete training environment lock. The installer leaves it untouched and uses the current working venv, avoiding unrequested package upgrades. Future environment rebuild must record CUDA compatibility and its own exact requirements lock; it is a separate operation from source refactor.

Data sources are at `data/raw/RSNA Pneumonia/raw` and two original 2018 JSON filenames in `data/raw`. `master_manifest.csv` keeps its legacy `RSNA Pneumonia/raw/...` relative values and the `paths.py` read-time resolver maps them to the current physical directory. Never rewrite frozen CSV paths to accommodate a cleanup.
