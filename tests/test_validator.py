from modules.dataset.validator import validate_eeg


GOOD = r"datasets\physionet.org\files\eegmmidb\1.0.0\S001\S001R04.edf"

BAD = r"data\raw\test_fixtures\broken.edf"


def test_valid_file():
    result = validate_eeg(GOOD)

    assert result["valid"] is True
    assert result["reasons"] == []


def test_invalid_file():
    result = validate_eeg(BAD)

    assert result["valid"] is False
    assert len(result["reasons"]) > 0