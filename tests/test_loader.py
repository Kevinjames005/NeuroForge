from modules.dataset.loader import load_and_register


SAMPLE = r"datasets\physionet.org\files\eegmmidb\1.0.0\S001\S001R04.edf"


def test_load_and_register():
    metadata = load_and_register(SAMPLE, "S001R04.edf")

    assert metadata["n_channels"] == 64
    assert metadata["sfreq"] == 160.0
    assert metadata["duration"] > 0
    assert "id" in metadata