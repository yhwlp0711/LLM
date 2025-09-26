import numpy as np

np.random.seed(42)
n_samples = 100
in_features = 6
out_features = 1
X = 2 * np.random.rand(n_samples, in_features)

true_W = np.random.randn(in_features, out_features)
true_b = np.random.randn(1, out_features)

y = X @ true_W + true_b

W = np.random.randn(in_features, out_features)
b = np.random.randn(1, out_features)

lr = 0.05
epochs = 2000

for epoch in range(epochs):
    y_pred = X @ W + b
    loss = np.mean((y_pred - y) ** 2)

    dW = (2/n_samples) * X.T @ (y_pred - y)
    db = (2/n_samples) * np.sum(y_pred - y)

    W = W - lr * dW
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    
print(W - true_W)
print(b - true_b)