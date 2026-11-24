
import numpy as np


# ReLU activation function
def relu(x):
    return np.maximum(0, x)


# Softmax activation function
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Initialize weights and biases
def initialize_parameters(width):
    W1 = np.random.normal(0, 0.01, (64, width))
    b1 = np.zeros(width)

    W2 = np.random.normal(0, 0.01, (width, width))
    b2 = np.zeros(width)

    W3 = np.random.normal(0, 0.01, (width, 10))
    b3 = np.zeros(10)

    return [W1, b1, W2, b2, W3, b3]


# Forward propagation
def forward_pass(X, parameters):
    W1, b1, W2, b2, W3, b3 = parameters

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    P = softmax(Z3)

    cache = (X, Z1, A1, Z2, A2, Z3, P)

    return P, cache


# Cross-entropy loss
def cross_entropy_loss(P, Y):
    P = np.clip(P, 1e-12, 1.0)
    return -np.mean(np.sum(Y * np.log(P), axis=1))


# Numerically stable loss calculated directly from logits
def stable_loss_from_logits(logits, Y):
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits

    log_sum_exp = (
        np.log(np.sum(np.exp(shifted), axis=1))
        + max_logits.ravel()
    )

    correct_scores = np.sum(Y * logits, axis=1)

    return np.mean(log_sum_exp - correct_scores)


# Backpropagation
def backward_pass(Y, parameters, cache):
    X, Z1, A1, Z2, A2, Z3, P = cache
    W1, b1, W2, b2, W3, b3 = parameters

    N = X.shape[0]

    # Output layer
    dZ3 = (P - Y) / N
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # Second hidden layer
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # First hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]


# Numerical gradient checking
def gradient_check(X, Y, parameters, gradients):
    epsilon = 1e-5
    rng = np.random.default_rng(42)

    passed = 0
    total = 10

    # Check weights in W1, W2 and W3
    weight_indices = [0, 2, 4]
    weight_names = {0: "W1", 2: "W2", 4: "W3"}

    for check in range(total):
        parameter_index = int(rng.choice(weight_indices))
        parameter = parameters[parameter_index]
        analytical_gradient = gradients[parameter_index]

        index = tuple(
            int(rng.integers(0, size))
            for size in parameter.shape
        )

        original_value = parameter[index]

        # Calculate loss at weight + epsilon
        parameter[index] = original_value + epsilon
        _, cache_plus = forward_pass(X, parameters)
        loss_plus = stable_loss_from_logits(cache_plus[5], Y)

        # Calculate loss at weight - epsilon
        parameter[index] = original_value - epsilon
        _, cache_minus = forward_pass(X, parameters)
        loss_minus = stable_loss_from_logits(cache_minus[5], Y)

        # Restore the original weight
        parameter[index] = original_value

        # Numerical gradient
        numerical_gradient = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        # Analytical gradient
        analytical_value = analytical_gradient[index]

        denominator = (
            abs(analytical_value) + abs(numerical_gradient)
        )

        if denominator < 1e-12:
            relative_error = abs(
                analytical_value - numerical_gradient
            )
        else:
            relative_error = (
                abs(analytical_value - numerical_gradient)
                / denominator
            )

        if relative_error < 1e-6:
            passed += 1

        print(
            f"Check {check + 1}: {weight_names[parameter_index]}, "
            f"analytical={analytical_value:.10e}, "
            f"numerical={numerical_gradient:.10e}, "
            f"relative error={relative_error:.3e}"
        )

    print(f"\nPassed: {passed}/{total}")
    print("Required: at least 9/10 with relative error < 1e-6")
    print("Gradient check:", "PASS" if passed >= 9 else "FAIL")


# Main program
if __name__ == "__main__":
    np.random.seed(42)

    # Create a sample batch
    X = np.random.rand(32, 64)
    labels = np.random.randint(0, 10, size=32)
    Y = np.eye(10)[labels]

    # Initialize network with hidden width 16
    parameters = initialize_parameters(16)

    # Test forward propagation and backpropagation
    P, cache = forward_pass(X, parameters)
    loss = cross_entropy_loss(P, Y)
    gradients = backward_pass(Y, parameters, cache)

    print("Initial loss:", loss)
    print("\nParameter and gradient shapes:")

    names = ["W1", "b1", "W2", "b2", "W3", "b3"]

    for name, parameter, gradient in zip(
        names, parameters, gradients
    ):
        print(
            f"{name}: parameter={parameter.shape}, "
            f"gradient={gradient.shape}"
        )

    print("\nGradient checking:")
    gradient_check(X, Y, parameters, gradients)
