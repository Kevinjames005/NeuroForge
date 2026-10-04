import mne

from core.db import SessionLocal, Dataset


def load_and_register(filepath: str, filename: str) -> dict:
    raw = mne.io.read_raw_edf(filepath, preload=True)

    metadata = {
        "n_channels": len(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
        "duration": raw.n_times / raw.info["sfreq"],
    }

    session = SessionLocal()

    record = Dataset(
        filename=filename,
        **metadata
    )

    session.add(record)
    session.commit()

    metadata["id"] = record.id

    session.close()

    return metadata