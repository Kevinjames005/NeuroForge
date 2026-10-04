import numpy as np
import mne

from modules.dataset.preprocess import preprocess
from modules.dataset.export import export_npz, load_npz


SAMPLE = r"datasets\physionet.org\files\eegmmidb\1.0.0\S001\S001R04.edf"

OUT = r"data\processed\S001R04.npz"


def test_export_roundtrip():
    raw = mne.io.read_raw_edf(SAMPLE, preload=True)

    data, labels, event_id = preprocess(raw)

    export_npz(
        data,
        labels,
        {"event_id": event_id},
        OUT
    )

    data2, labels2, metadata2 = load_npz(OUT)

    assert np.array_equal(data, data2)
    assert labels.tolist() == labels2.tolist()
    assert metadata2["event_id"] == event_id