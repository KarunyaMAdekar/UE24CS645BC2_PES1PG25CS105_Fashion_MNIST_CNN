import numpy as np
from layers.conv_layer    import ConvLayer
from layers.pool_layer    import MaxPoolLayer
from layers.flatten_layer import FlattenLayer
from layers.fc_layer      import FCLayer

class CNN:
    """
    Architecture:
      Conv(1→8, k=3, p=1) → ReLU → MaxPool(2)   → [N, 8, 14, 14]
      Conv(8→16, k=3, p=1)→ ReLU → MaxPool(2)   → [N, 16, 7, 7]
      Flatten                                     → [N, 784]
      FC(784→128, ReLU)
      FC(128→10, Softmax)
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.conv1   = ConvLayer(1,  8,  kernel_size=3, padding=1)
        self.pool1   = MaxPoolLayer(2, 2)
        self.conv2   = ConvLayer(8,  16, kernel_size=3, padding=1)
        self.pool2   = MaxPoolLayer(2, 2)
        self.flatten = FlattenLayer()
        self.fc1     = FCLayer(16*7*7, 128, activation='relu')
        self.fc2     = FCLayer(128,    10,  activation='softmax')

    def _relu(self, X): return np.maximum(0, X)
    def _relu_back(self, dA, Z): return dA * (Z > 0)

    def forward(self, X):
        Z1 = self.conv1.forward(X);  A1 = self._relu(Z1);  self.Z1 = Z1
        P1 = self.pool1.forward(A1)
        Z2 = self.conv2.forward(P1); A2 = self._relu(Z2);  self.Z2 = Z2
        P2 = self.pool2.forward(A2)
        F  = self.flatten.forward(P2)
        A3 = self.fc1.forward(F)
        A4 = self.fc2.forward(A3)
        return A4   # probabilities

    def cross_entropy_loss(self, probs, y):
        N = y.shape[0]
        log_p = -np.log(probs[np.arange(N), y] + 1e-9)
        return log_p.mean()

    def backward(self, probs, y):
        N = y.shape[0]
        # Gradient of softmax + cross-entropy
        dA4 = probs.copy()
        dA4[np.arange(N), y] -= 1
        dA4 /= N

        dF  = self.fc2.backward(dA4)
        dA3 = self.fc1.backward(dF)
        dP2 = self.flatten.backward(dA3)
        dA2 = self.pool2.backward(dP2)
        dZ2 = self._relu_back(dA2, self.Z2)
        dP1 = self.conv2.backward(dZ2)
        dA1 = self.pool1.backward(dP1)
        dZ1 = self._relu_back(dA1, self.Z1)
        self.conv1.backward(dZ1)

    def update(self):
        self.conv1.update(self.lr)
        self.conv2.update(self.lr)
        self.fc1.update(self.lr)
        self.fc2.update(self.lr)