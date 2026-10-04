from fastapi import APIRouter, UploadFile
import shutil
import os
import mne

from modules.dataset.loader import load_and_register
from modules.dataset.validator import validate_eeg
from modules.dataset.preprocess import preprocess
from modules.dataset.export import export_npz


router = APIRouter()


@router.post("/datasets/upload")
def upload_dataset(file: UploadFile):
    dest = f"data/raw/{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    validation = validate_eeg(dest)

    if not validation["valid"]:
        return {
            "status": "rejected",
            "reasons": validation["reasons"]
        }

    metadata = load_and_register(
        dest,
        file.filename
    )

    raw = mne.io.read_raw_edf(
        dest,
        preload=True
    )

    data, labels, event_id = preprocess(raw)

    out_path = (
        f"data/processed/"
        f"{os.path.splitext(file.filename)[0]}.npz"
    )

    export_npz(
        data,
        labels,
        {"event_id": event_id},
        out_path
    )

    metadata["processed_path"] = out_path

    return {
        "status": "ok",
        "metadata": metadata
    }