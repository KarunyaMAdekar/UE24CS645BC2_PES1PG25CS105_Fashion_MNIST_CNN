import numpy as np

class MaxPoolLayer:
    """
    2D Max Pooling
    Input:  (N, C, H, W)
    Output: (N, C, H//pool_size, W//pool_size)
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride    = stride

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        p, s = self.pool_size, self.stride
        H_out, W_out = (H - p) // s + 1, (W - p) // s + 1
        out = np.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                patch = X[:, :, i*s:i*s+p, j*s:j*s+p]
                out[:, :, i, j] = patch.max(axis=(2, 3))
        return out

    def backward(self, dout):
        X, p, s = self.X, self.pool_size, self.stride
        N, C, H, W = X.shape
        H_out, W_out = dout.shape[2], dout.shape[3]
        dX = np.zeros_like(X)

        for i in range(H_out):
            for j in range(W_out):
                patch = X[:, :, i*s:i*s+p, j*s:j*s+p]
                # Mask: 1 where the max value is
                max_val = patch.max(axis=(2,3), keepdims=True)
                mask    = (patch == max_val)
                # Distribute gradient to max positions
                dX[:, :, i*s:i*s+p, j*s:j*s+p] += \
                    mask * dout[:, :, i, j][:, :, np.newaxis, np.newaxis]
        return dX