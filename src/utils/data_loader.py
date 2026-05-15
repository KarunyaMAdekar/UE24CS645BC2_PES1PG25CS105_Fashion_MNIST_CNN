import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_fashion_mnist():
    print("Loading Fashion MNIST...")
    data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X, y = data.data, data.target.astype(int)

    # Normalize to [0, 1] and reshape to (N, 1, 28, 28)
    X = X.astype(np.float32) / 255.0
    X = X.reshape(-1, 1, 28, 28)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test