import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE = r"datasets\physionet.org\files\eegmmidb\1.0.0\S001\S001R04.edf"


def test_upload_dataset():
    with open(SAMPLE, "rb") as f:
        response = client.post(
            "/datasets/upload",
            files={"file": ("S001R04.edf", f, "application/octet-stream")}
        )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"

    metadata = body["metadata"]

    assert metadata["n_channels"] == 64
    assert metadata["sfreq"] == 160.0
    assert "id" in metadata

    processed_path = metadata["processed_path"]

    assert os.path.exists(processed_path)