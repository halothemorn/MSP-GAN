# MSP-GAN

MSP-GAN reconstructs 3D coating microstructures from segmented 2D SEM images. Train one model per coating using your own phase labels. This repository provides source code only.

## Data

Split by original SEM field before tiling. Use integer labels `0 = crack`, `1 = pore`, and `2 = matrix`. Keep the pixel spacing consistent and the coating thickness along image rows.

```text
fields/train/A.npy             Full training field
patches/train/A/patch_001.npy  Non-overlapping patch from field A
fields/validation/B.npy        Full held-out field
```

Match each training patch folder to its field ID. Use square patches of a common power-of-two width of at least 16 pixels. Training samples fixed patches and computes reference statistics from the full training fields.

## Run

From the repository directory with Python 3.11:

```bash
python -m pip install -r requirements.txt
python train.py --data patches --fields fields --output runs/generator.pt
python validate.py --fields fields --checkpoint runs/generator.pt --output runs/evaluation
```

Use a new evaluation directory for each run. It contains the generated 3D label volumes (`volume_*.npy`), phase fractions (`phase_fractions.csv`), directional pore-crack correlations (`s2_void.csv`), and a held-out comparison summary (`summary.json`).
