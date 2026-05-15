import numpy as np

class ConvLayer:
    """
    2D Convolution Layer
    Input shape:  (N, C_in, H, W)
    Output shape: (N, C_out, H_out, W_out)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _pad(self, X):
        if self.padding > 0:
            return np.pad(X, ((0,0),(0,0),(self.padding, self.padding),(self.padding, self.padding)))
        return X

    def forward(self, X):
        self.X_input = X
        N, C, H, W = X.shape
        p, s, k   = self.padding, self.stride, self.kernel_size
        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1

        X_pad = self._pad(X)
        out   = np.zeros((N, self.out_channels, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i*s, j*s
                patch = X_pad[:, :, h_start:h_start+k, w_start:w_start+k]  # (N, C, k, k)
                # out[:, f, i, j] = sum over (C, k, k) of patch * W[f]
                out[:, :, i, j] = np.tensordot(patch, self.W, axes=([1,2,3],[1,2,3])) + self.b
        self.X_pad = X_pad
        return out

    def backward(self, dout):
        N, C, H, W = self.X_input.shape
        p, s, k   = self.padding, self.stride, self.kernel_size
        H_out     = (H + 2*p - k) // s + 1
        W_out     = (W + 2*p - k) // s + 1

        dX_pad    = np.zeros_like(self.X_pad)
        self.dW   = np.zeros_like(self.W)
        self.db   = np.zeros_like(self.b)

        # db: sum over N, H_out, W_out
        self.db   = dout.sum(axis=(0, 2, 3))

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i*s, j*s
                patch = self.X_pad[:, :, h_start:h_start+k, w_start:w_start+k]
                # dW: accumulate
                self.dW += np.tensordot(dout[:, :, i, j], patch, axes=([0],[0]))
                # dX
                dX_pad[:, :, h_start:h_start+k, w_start:w_start+k] += \
                    np.tensordot(dout[:, :, i, j], self.W, axes=([1],[0]))

        if self.padding > 0:
            return dX_pad[:, :, p:-p, p:-p]
        return dX_pad

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db