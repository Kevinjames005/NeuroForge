import numpy as np
from load_data import load_bci_iv_2a

def inspect(X, y, metadata):
    print("="*50)
    print("SHAPE & TYPE")
    print("="*50)
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    print("\n" + "="*50)
    print("VALUE RANGE / SCALE")
    print("="*50)
    print("Min:", X.min())
    print("Max:", X.max())
    print("Mean:", X.mean())
    print("Std:", X.std())

    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    print("Any NaNs?", np.isnan(X).any())
    print("Any Infs?", np.isinf(X).any())

    print("\n" + "="*50)
    print("CLASS DISTRIBUTION")
    print("="*50)
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")

    print("\n" + "="*50)
    print("METADATA PREVIEW")
    print("="*50)
    print(metadata.head())

if __name__ == "__main__":
    X, y, metadata = load_bci_iv_2a()
    inspect(X, y, metadata)