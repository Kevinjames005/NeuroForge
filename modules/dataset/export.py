import numpy as np
import json


def export_npz(data, labels, metadata: dict, out_path: str):
    np.savez(
        out_path,
        data=data,
        labels=labels,
        metadata=json.dumps(metadata)
    )


def load_npz(path: str):
    npz = np.load(path, allow_pickle=True)

    metadata = json.loads(str(npz["metadata"]))

    return npz["data"], npz["labels"], metadata