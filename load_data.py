from moabb.datasets import BNCI2014_001  # renamed from BNCI2014001
from moabb.paradigms import MotorImagery
import numpy as np

def load_bci_iv_2a():
    dataset = BNCI2014_001()
    paradigm = MotorImagery()

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

    return X, y, metadata

if __name__ == "__main__":
    X, y, metadata = load_bci_iv_2a()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First 5 labels:", y[:5])
    print("Metadata columns:", metadata.columns.tolist())