import numpy as np
import matplotlib.pyplot as plt

def softmax(u):
    u_shifted = u - np.max(u, axis=0, keepdims=True)
    exp_u = np.exp(u_shifted)
    y = exp_u / np.sum(exp_u, axis=0, keepdims=True)
    return y

def computeThetaGD(x, y, alpha=0.01, iterations=1000, plot_loss=False):
    loss = np.zeros(iterations)

    y = y.flatten()
    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_classes = np.max(y) + 1

    _x = np.c_[np.ones(n_samples), x]
    theta = np.random.randn(n_features + 1, n_classes) * 0.01

    _y = np.zeros((n_samples, n_classes))
    _y[np.arange(n_samples), y] = 1

    for i in range(iterations):
        u = _x @ theta
        prediction = softmax(u.T).T
        gradient = _x.T @ (prediction - _y)
        theta = theta - alpha * gradient
        loss[i] = -np.sum(_y * np.log(prediction + 1e-15)) / n_samples

    if plot_loss:
        plt.figure(figsize=(12, 4))

        # Plot 1: Full loss curve
        plt.subplot(1, 3, 1)
        plt.plot(np.arange(iterations), loss)
        plt.xlabel("Iterace")
        plt.ylabel("Loss")
        plt.title("Full Loss Curve")
        plt.grid(True)

        # Plot 2: Loss after first 10 iterations (to see detail)
        plt.subplot(1, 3, 2)
        plt.plot(np.arange(10, iterations), loss[10:])
        plt.xlabel("Iterace")
        plt.ylabel("Loss")
        plt.title("Loss (skipping first 10 iterations)")
        plt.grid(True)

        # Plot 3: Log scale
        plt.subplot(1, 3, 3)
        plt.semilogy(np.arange(iterations), loss)
        plt.xlabel("Iterace")
        plt.ylabel("Loss (log scale)")
        plt.title("Loss (log scale)")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('loss_debug.png', dpi=100)
        plt.show()

    print(f"Loss at iteration 0: {loss[0]}")
    print(f"Loss at iteration 10: {loss[10]}")
    print(f"Loss at iteration 100: {loss[100]}")
    print(f"Loss at final iteration: {loss[-1]}")
    print(f"Min loss: {np.min(loss)}")
    print(f"Max loss: {np.max(loss)}")

    return theta

# Load data
npzfile = np.load('data/data_07_3cl_ez.npz')
data = npzfile['data']
ref = npzfile['ref']

print("Data shape:", data.shape)
print("Ref shape:", ref.shape)
print("Unique classes:", np.unique(ref))
print()

theta = computeThetaGD(data, ref, alpha=0.01, iterations=1000, plot_loss=True)
print("\nFinal theta:")
print(theta)

