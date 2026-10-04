import numpy as np
import mne


def validate_eeg(filepath: str) -> dict:
    reasons = []

    try:
        raw = mne.io.read_raw_edf(filepath, preload=True)
    except Exception as e:
        return {
            "valid": False,
            "reasons": [f"unreadable: {e}"]
        }

    if len(raw.ch_names) == 0:
        reasons.append("no channels found")

    data = raw.get_data()

    if np.isnan(data).any():
        reasons.append("contains NaNs")

    if raw.info["sfreq"] <= 0:
        reasons.append("invalid sampling rate")

    try:
        events, _ = mne.events_from_annotations(raw)

        if len(events) == 0:
            reasons.append("no event markers found")

    except Exception:
        reasons.append("no event markers found")

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons
    }