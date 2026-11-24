
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Subtract the maximum for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def initialize_parameters(width):
    W1 = np.random.normal(0, 0.01, (64, width))
    b1 = np.zeros(width)

    W2 = np.random.normal(0, 0.01, (width, width))
    b2 = np.zeros(width)

    W3 = np.random.normal(0, 0.01, (width, 10))
    b3 = np.zeros(10)

    return W1, b1, W2, b2, W3, b3

def forward_pass(X, parameters):
    W1, b1, W2, b2, W3, b3 = parameters

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    probabilities = softmax(Z3)

    return probabilities

# Load and prepare the dataset
X, y = load_digits(return_X_y=True)
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

np.random.seed(42)

# Test both required hidden-layer widths
for width in [16, 64]:
    parameters = initialize_parameters(width)
    probabilities = forward_pass(X_train[:32], parameters)

    print(f"\nNetwork width: {width}")
    print("Output shape:", probabilities.shape)
    print("First row probability sum:", probabilities[0].sum())
    print("All probabilities between 0 and 1:",
          np.all((probabilities >= 0) & (probabilities <= 1)))
    print("All row sums equal 1:",
          np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))
