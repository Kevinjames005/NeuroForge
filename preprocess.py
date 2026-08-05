"""
NeuroForge - Person 1 Module
Dataset & Preprocessing Pipeline for BCI Competition IV-2a

Processes all 9 subjects' Training (T) files into standardized,
normalized epoch arrays ready for Person 2's synthetic generation module.
"""

import mne
import numpy as np
import os
import glob

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
DATA_FOLDER = "data"
OUTPUT_FOLDER = "processed_data"
LOW_FREQ = 4.0
HIGH_FREQ = 38.0
TMIN = -0.5
TMAX = 4.0

EVENT_ID = {
    "left_hand": 769,
    "right_hand": 770,
    "feet": 771,
    "tongue": 772,
}


def process_file(filepath):
    """Runs the full pipeline on one .gdf file and returns (X, y, label_names, ch_names, sfreq)."""
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)

    eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    raw.drop_channels(eog_channels)

    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    selected_event_id = {}
    for name, code in EVENT_ID.items():
        key = str(code)
        if key in event_dict:
            selected_event_id[name] = event_dict[key]

    if len(selected_event_id) != 4:
        raise ValueError(
            f"Expected 4 motor imagery events in {filepath}, "
            f"found {len(selected_event_id)}: {selected_event_id}"
        )

    epochs = mne.Epochs(
        raw, events, event_id=selected_event_id,
        tmin=TMIN, tmax=TMAX, baseline=(TMIN, 0),
        preload=True, verbose=False,
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    X_mean = X.mean(axis=2, keepdims=True)
    X_std = X.std(axis=2, keepdims=True)
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    code_to_name = {v: k for k, v in selected_event_id.items()}
    label_names = np.array([code_to_name[label] for label in y])

    return X_normalized, y, label_names, epochs.ch_names, epochs.info["sfreq"]


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    training_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*T.gdf")))

    if not training_files:
        print(f"No training .gdf files found in '{DATA_FOLDER}/'. Check your data folder.")
        return

    print(f"Found {len(training_files)} training files to process.\n")

    summary = []

    for i, filepath in enumerate(training_files, start=1):
        subject_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"[{i}/{len(training_files)}] Processing {subject_name} ...")

        try:
            X, y, label_names, ch_names, sfreq = process_file(filepath)
        except Exception as e:
            print(f"   FAILED: {e}\n")
            summary.append((subject_name, "FAILED", str(e)))
            continue

        output_path = os.path.join(OUTPUT_FOLDER, f"{subject_name}_processed.npz")
        np.savez(
            output_path,
            X=X,
            y=y,
            label_names=label_names,
            channel_names=np.array(ch_names),
            sfreq=sfreq,
        )
        print(f"   -> Saved {X.shape} to {output_path}\n")
        summary.append((subject_name, "OK", str(X.shape)))

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for subject_name, status, info in summary:
        print(f"{subject_name}: {status} - {info}")


if __name__ == "__main__":
    main()