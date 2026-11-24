
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def relu(x):
    return np.maximum(0, x)

def softmax(x):
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

    return softmax(Z3)

def cross_entropy_loss(probabilities, Y):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -np.mean(np.sum(Y * np.log(probabilities), axis=1))

# Load and prepare the dataset
X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42, stratify=y
)

np.random.seed(42)

for width in [16, 64]:
    parameters = initialize_parameters(width)
    probabilities = forward_pass(X_train, parameters)
    loss = cross_entropy_loss(probabilities, Y_train)

    print(f"Network width: {width}")
    print(f"Initial loss: {loss:.6f}")
    print(f"Expected loss approximately: {np.log(10):.6f}")
