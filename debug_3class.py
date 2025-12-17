import numpy as np

def softmax(u):
    """
    vstupem muze byt skalar, vektor, nebo matice
    """
    # Numerical stability: subtract max value to prevent overflow
    u_shifted = u - np.max(u, axis=0, keepdims=True)
    exp_u = np.exp(u_shifted)
    y = exp_u / np.sum(exp_u, axis=0, keepdims=True)
    return y

# Load the 3-class data
npzfile = np.load('data/data_07_3cl_ez.npz')
data = npzfile['data']
ref = npzfile['ref']

print('Data shape:', data.shape)
print('Ref shape:', ref.shape)
print('Ref dtype:', ref.dtype)
print('Unique classes:', np.unique(ref))
print('First 20 ref values:', ref[:20])
print()

# Try the training process
y = ref.flatten()
n_samples = data.shape[0]
n_features = data.shape[1]
n_classes = np.max(y) + 1

print(f'n_samples: {n_samples}')
print(f'n_features: {n_features}')
print(f'n_classes: {n_classes}')
print()

# Add bias
_x = np.c_[np.ones(n_samples), data]
print(f'_x shape: {_x.shape}')

# Initialize theta
np.random.seed(42)
theta = np.random.randn(n_features + 1, n_classes) * 0.01
print(f'theta shape: {theta.shape}')
print(f'theta initial:\n{theta}')
print()

# One-hot encode
_y = np.zeros((n_samples, n_classes))
_y[np.arange(n_samples), y] = 1
print(f'_y shape: {_y.shape}')
print(f'First 10 rows of _y:\n{_y[:10]}')
print(f'Sum of _y per row (should all be 1): {_y.sum(axis=1)[:10]}')
print()

# First forward pass
u = _x @ theta
print(f'u shape: {u.shape}')
print(f'u sample:\n{u[:5]}')
print()

prediction = softmax(u.T).T
print(f'prediction shape: {prediction.shape}')
print(f'prediction sample:\n{prediction[:5]}')
print(f'Sum per row (should be 1): {prediction.sum(axis=1)[:5]}')
print()

# Calculate loss
loss = -np.sum(_y * np.log(prediction + 1e-15)) / n_samples
print(f'Initial loss: {loss}')

