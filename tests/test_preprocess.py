import mne

from modules.dataset.preprocess import preprocess


SAMPLE = r"datasets\physionet.org\files\eegmmidb\1.0.0\S001\S001R04.edf"


def test_preprocess_shape():
    raw = mne.io.read_raw_edf(SAMPLE, preload=True)

    data, labels, event_id = preprocess(raw)

    n_channels = len(raw.ch_names)

    assert data.shape[0] > 0
    assert data.shape[1] == n_channels
    assert data.shape[2] == int(2.0 * raw.info["sfreq"]) + 1
    assert len(labels) == data.shape[0]