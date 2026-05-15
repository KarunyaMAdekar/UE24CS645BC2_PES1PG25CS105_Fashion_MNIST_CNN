import numpy as np

class FlattenLayer:
    """Flattens (N, C, H, W) → (N, C*H*W)"""
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)