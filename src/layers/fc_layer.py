import numpy as np

class FCLayer:
    """
    Dense layer with optional ReLU or Softmax activation.
    activation: 'relu' | 'softmax' | None
    """
    def __init__(self, in_features, out_features, activation=None):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.activation = activation

    def _relu(self, Z):        return np.maximum(0, Z)
    def _relu_back(self, dA):  return dA * (self.Z > 0)

    def _softmax(self, Z):
        eZ = np.exp(Z - Z.max(axis=1, keepdims=True))
        return eZ / eZ.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        if   self.activation == 'relu':    return self._relu(self.Z)
        elif self.activation == 'softmax': return self._softmax(self.Z)
        return self.Z

    def backward(self, dout):
        if self.activation == 'relu':
            dout = self._relu_back(dout)
        # For softmax, dout is already (probs - one_hot) / N from loss
        self.dW = self.X.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db