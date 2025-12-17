import numpy as np
import matplotlib.pyplot as plt

def softmax(u):
    u_shifted = u - np.max(u, axis=0, keepdims=True)
    exp_u = np.exp(u_shifted)
    y = exp_u / np.sum(exp_u, axis=0, keepdims=True)
    return y

def computeThetaGD_DEBUG(x, y, alpha=0.01, iterations=1000):
    loss = np.zeros(iterations)

    print(f"Input y shape BEFORE flatten: {y.shape}")
    print(f"Input y sample: {y[:5]}")

    # CHECK IF FLATTEN HAPPENS
    y = y.flatten()

    print(f"Input y shape AFTER flatten: {y.shape}")
    print(f"Input y sample after flatten: {y[:5]}")

    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_classes = np.max(y) + 1

    _x = np.c_[np.ones(n_samples), x]
    theta = np.random.randn(n_features + 1, n_classes) * 0.01

    # One-hot encode
    _y = np.zeros((n_samples, n_classes))
    _y[np.arange(n_samples), y] = 1

    print(f"One-hot _y shape: {_y.shape}")
    print(f"One-hot _y first 5 rows:\n{_y[:5]}")
    print(f"Row sums (should be 1): {_y.sum(axis=1)[:5]}")
    print()

    for i in range(min(5, iterations)):  # Just 5 iterations for debug
        u = _x @ theta
        prediction = softmax(u.T).T
        gradient = _x.T @ (prediction - _y)
        theta = theta - alpha * gradient
        loss[i] = -np.sum(_y * np.log(prediction + 1e-15)) / n_samples
        print(f"Iteration {i}: loss = {loss[i]}")

    return theta, loss

# Load data
npzfile = np.load('data/data_07_3cl_ez.npz')
data = npzfile['data']
ref = npzfile['ref']

print("="*60)
print("Testing with actual 3-class data")
print("="*60)
theta, loss = computeThetaGD_DEBUG(data, ref, alpha=0.01, iterations=5)

