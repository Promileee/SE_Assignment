import numpy as np
import matplotlib.pyplot as plt


def g(x1, x2):
    return np.sin(5 * x1 / 2) + 2 - ((x1**2 + 4) * (x2 - 1)) / 20


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_count = int(np.ceil(n_samples * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class NumpyMLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes=(64, 32, 16),
        max_iter=1000,
        learning_rate=0.01,
        random_state=None,
    ):
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.params_initialized = False

    def _init_params(self, n_features):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        layer_sizes = [n_features] + self.hidden_layer_sizes + [1]
        self.weights = []
        self.biases = []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if i < len(layer_sizes) - 2:
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(1.0 / fan_in)
            W = std * np.random.randn(fan_in, fan_out)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(W))
            self.v_w.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))
        self.t = 0
        self.params_initialized = True

    @staticmethod
    def _relu(Z):
        return np.maximum(0.0, Z)

    @staticmethod
    def _relu_grad(Z):
        return (Z > 0).astype(Z.dtype)

    def _forward(self, X):
        A = X
        caches = []
        num_layers = len(self.weights)
        for i in range(num_layers):
            W, b = self.weights[i], self.biases[i]
            Z = A @ W + b
            if i < num_layers - 1:
                A = self._relu(Z)
            else:
                A = Z
            caches.append((Z, A))
        return A, caches

    def _backward(self, X, y, caches):
        n = X.shape[0]
        y = y.reshape(-1, 1)
        y_pred = caches[-1][1]
        dA = (2.0 / n) * (y_pred - y)
        grads_W = [None] * len(self.weights)
        grads_B = [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            Z_i, _ = caches[i]
            A_prev = X if i == 0 else caches[i - 1][1]
            if i == len(self.weights) - 1:
                dZ = dA
            else:
                dZ = dA * self._relu_grad(Z_i)
            grads_W[i] = A_prev.T @ dZ
            grads_B[i] = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ self.weights[i].T
        return grads_W, grads_B

    def _adam_update(self, grads_W, grads_B, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        lr = self.learning_rate
        for i in range(len(self.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grads_W[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (grads_W[i] ** 2)
            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grads_B[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (grads_B[i] ** 2)
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def fit(self, X, y):
        if not self.params_initialized:
            self._init_params(X.shape[1])
        for _ in range(self.max_iter):
            y_pred, caches = self._forward(X)
            grads_W, grads_B = self._backward(X, y, caches)
            self._adam_update(grads_W, grads_B)
        return self

    def predict(self, X):
        y_pred, _ = self._forward(X)
        return y_pred.ravel()

    def score(self, X, y):
        y_true = y
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


if __name__ == "__main__":
    n_samples = 100
    x1_samples = np.random.uniform(-2, 5, n_samples)
    x2_samples = np.random.uniform(-1, 6, n_samples)
    y_samples = g(x1_samples, x2_samples)
    X_samples = np.vstack([x1_samples, x2_samples]).T

    X_train, X_test, y_train, y_test = train_test_split(
        X_samples, y_samples, test_size=0.2, random_state=42
    )

    model = NumpyMLPRegressor(
        hidden_layer_sizes=(64, 32, 16), max_iter=1000, learning_rate=0.01, random_state=42
    )
    model.fit(X_train, y_train)

    x1 = np.linspace(-2, 5, 100)
    x2 = np.linspace(-1, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)

    Z_real = g(X1, X2)
    Z_pred = model.predict(np.vstack([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

    contour_levels = [-4, -2, 0, 2, 4]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    cp_real = plt.contour(X1, X2, Z_real, levels=contour_levels, cmap='viridis')
    plt.contour(X1, X2, Z_real, levels=[0], colors='red', linewidths=2)
    plt.title('Real Function $g(x_1, x_2)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.colorbar(cp_real)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    cp_pred = plt.contour(X1, X2, Z_pred, levels=contour_levels, cmap='viridis')
    plt.contour(X1, X2, Z_pred, levels=[0], colors='red', linewidths=2)
    plt.title('Neural Network Approximation')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.colorbar(cp_pred)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    score = model.score(X_test, y_test)
    print(f"R^2 score on test set: {score:.4f}")


