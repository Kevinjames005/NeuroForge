# NeuroForge – Dataset & Preprocessing Module (Person 1)

## Overview
This module loads raw EEG recordings from the **BCI Competition IV-2a** dataset and converts them into a clean, standardized format ready for the Synthetic EEG Generation module (Person 2).

## Dataset
- **Source:** BCI Competition IV, Dataset 2a (official site: bbci.de/competition/iv/)
- **Subjects:** 9
- **Task:** Motor imagery — subjects imagine one of 4 movements
- **Classes:** left hand, right hand, feet, tongue
- **Channels used:** 22 EEG channels (3 EOG channels removed — used only for eye-artifact detection, not analysis)
- **Sampling rate:** 250 Hz
- **Files used:** Training (`*T.gdf`) files only, since Evaluation (`*E.gdf`) files do not contain embedded class labels in this dataset release.

## Pipeline Steps

1. **Load** — Raw `.gdf` file is loaded using MNE-Python.
2. **Channel selection** — The 3 EOG (eye movement) channels are removed by name, keeping only the 22 EEG channels.
3. **Band-pass filtering** — A 4–38 Hz filter is applied. This range covers the brain-wave frequencies relevant to motor imagery (theta, alpha, beta bands) and removes slow drift and high-frequency noise.
4. **Epoching** — The continuous recording is cut into individual trials. Each trial spans -0.5s to +4.0s around the movement cue (4.5 seconds total, 1126 samples at 250 Hz). The pre-cue period (-0.5s to 0s) is used for baseline correction.
5. **Baseline correction** — Applied automatically during epoching, using the pre-cue period as the baseline for each trial.
6. **Normalization** — Each trial is Z-score normalized per channel (mean subtracted, divided by standard deviation), so amplitude differences between subjects/sessions don't bias downstream models.
7. **Export** — Each subject's processed data is saved as a `.npz` file containing:
   - `X`: EEG data, shape `(288, 22, 1126)` = (trials, channels, time samples)
   - `y`: numeric class labels
   - `label_names`: readable class labels (left_hand, right_hand, feet, tongue)
   - `channel_names`: names of the 22 EEG channels
   - `sfreq`: sampling frequency (250 Hz)

## Output
9 files in `processed_data/`, one per subject:


Each subject has 288 trials (72 per class × 4 classes).

## How to Reproduce
1. Install dependencies: `pip install -r requirements.txt`
2. Place the 9 training `.gdf` files in a `data/` folder
3. Run: `python preprocess.py`

## How to Load the Output (for Person 2)
```python
import numpy as np

data = np.load("processed_data/A01T_processed.npz")
X = data["X"]                  # shape (288, 22, 1126)
y = data["y"]                  # numeric labels
label_names = data["label_names"]  # e.g. 'left_hand', 'right_hand', ...
```

## Notes / Known Limitations
- Evaluation (`E`) files were not included in this version, since they require a separate true-labels file not bundled with the main dataset download. Can be added later if needed.
- No trials were rejected in this run — the dataset's built-in rejection marker (event code 1023) was not present in the Training files used.

