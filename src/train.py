import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data_loader import load_fashion_mnist
from model import CNN

CLASSES = ['T-shirt','Trouser','Pullover','Dress','Coat',
           'Sandal','Shirt','Sneaker','Bag','Ankle boot']

def get_batches(X, y, batch_size):
    N = X.shape[0]
    idx = np.random.permutation(N)
    for start in range(0, N, batch_size):
        b = idx[start:start+batch_size]
        yield X[b], y[b]

def evaluate(model, X, y, batch_size=256):
    correct, total = 0, 0
    for Xb, yb in get_batches(X, y, batch_size):
        probs = model.forward(Xb)
        correct += (probs.argmax(axis=1) == yb).sum()
        total   += len(yb)
    return correct / total

def train(epochs=5, batch_size=64, lr=0.01):
    X_train, X_test, y_train, y_test = load_fashion_mnist()
    model = CNN(lr=lr)

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(1, epochs+1):
        epoch_loss = []
        for Xb, yb in tqdm(get_batches(X_train, y_train, batch_size),
                           desc=f"Epoch {epoch}/{epochs}"):
            probs = model.forward(Xb)
            loss  = model.cross_entropy_loss(probs, yb)
            model.backward(probs, yb)
            model.update()
            epoch_loss.append(loss)

        avg_loss  = np.mean(epoch_loss)
        train_acc = evaluate(model, X_train[:5000], y_train[:5000])
        test_acc  = evaluate(model, X_test,  y_test)
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    # ── Plots ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, marker='o'); axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(train_accs,  marker='o', label='Train')
    axes[1].plot(test_accs,   marker='s', label='Test')
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout(); plt.savefig("results.png"); plt.show()
    print("Plot saved to results.png")
    return model

if __name__ == "__main__":
    train(epochs=5, batch_size=64, lr=0.01)