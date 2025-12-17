import numpy as np

# Simulate the issue
ref = np.array([[0], [0], [1], [2], [1]])  # Shape (5, 1)

print("Original ref shape:", ref.shape)
print("Original ref:\n", ref)
print()

# What happens with flatten
y = ref.flatten()
print("After flatten, y shape:", y.shape)
print("After flatten, y:", y)
print()

# One-hot encoding
n_samples = len(y)
n_classes = np.max(y) + 1
_y = np.zeros((n_samples, n_classes))
_y[np.arange(n_samples), y] = 1

print("One-hot encoded _y:\n", _y)
print()

# Now test with 2D ref that's not flattened properly
ref_2d = np.array([[0], [0], [1], [2], [1]])
print("If we DON'T flatten and y is still 2D:", ref_2d.shape)
try:
    _y_bad = np.zeros((n_samples, n_classes))
    _y_bad[np.arange(n_samples), ref_2d] = 1  # This might fail or give wrong result
    print("One-hot with 2D ref:\n", _y_bad)
except Exception as e:
    print("ERROR:", e)

