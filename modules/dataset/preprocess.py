import mne
import numpy as np


def preprocess(raw, l_freq=1.0, h_freq=40.0, tmin=0.0, tmax=2.0):
    raw = raw.copy()

    raw.filter(l_freq=l_freq, h_freq=h_freq)

    events, event_id = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )

    data = epochs.get_data()

    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)

    data = (data - mean) / (std + 1e-8)

    labels = epochs.events[:, 2]

    return data, labels, event_id