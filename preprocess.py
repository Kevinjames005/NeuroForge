import numpy as np

def normalize(X):
    """Z-score normalize EEG data. Returns normalized data + stats for later reversal."""
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), {"mean": mean, "std": std}

def denormalize(X_norm, stats):
    """Reverse normalization to get back to original EEG scale."""
    return X_norm * stats["std"] + stats["mean"]

def encode_labels(y):
    """Convert string labels to integer class indices."""
    classes = sorted(np.unique(y))  # consistent ordering
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int64)
    return y_encoded, label_to_idx

if __name__ == "__main__":
    from load_data import load_bci_iv_2a

    X, y, metadata = load_bci_iv_2a()

    X_norm, stats = normalize(X)
    y_encoded, label_map = encode_labels(y)

    print("Normalized mean/std:", X_norm.mean(), X_norm.std())
    print("Label map:", label_map)
    print("First 5 encoded labels:", y_encoded[:5])