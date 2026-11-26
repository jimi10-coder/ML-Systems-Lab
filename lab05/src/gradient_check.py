
import torch
from autograd_engine import Value


def numerical_gradient(function, x, epsilon=1e-6):
    x_plus = x + epsilon
    x_minus = x - epsilon

    return (function(x_plus) - function(x_minus)) / (2 * epsilon)


# --------------------------------------------------
# 1. Our automatic differentiation engine
# --------------------------------------------------

a = Value(2.0)
b = Value(3.0)
c = Value(1.0)

output = ((a * b + c) ** 2).relu()
output.backward()

our_gradients = [a.grad, b.grad, c.grad]

print("Our engine:")
print("Output:", output.data)
print("Gradients:", our_gradients)


# --------------------------------------------------
# 2. PyTorch automatic differentiation
# --------------------------------------------------

ta = torch.tensor(2.0, requires_grad=True)
tb = torch.tensor(3.0, requires_grad=True)
tc = torch.tensor(1.0, requires_grad=True)

torch_output = torch.relu((ta * tb + tc) ** 2)
torch_output.backward()

torch_gradients = [
    ta.grad.item(),
    tb.grad.item(),
    tc.grad.item()
]

print("\nPyTorch:")
print("Output:", torch_output.item())
print("Gradients:", torch_gradients)


# --------------------------------------------------
# 3. Numerical gradients
# --------------------------------------------------

def calculate_output(a, b, c):
    return max(0, (a * b + c) ** 2)


numerical_gradients = [
    numerical_gradient(lambda x: calculate_output(x, 3.0, 1.0), 2.0),
    numerical_gradient(lambda x: calculate_output(2.0, x, 1.0), 3.0),
    numerical_gradient(lambda x: calculate_output(2.0, 3.0, x), 1.0)
]

print("\nNumerical gradients:")
print(numerical_gradients)


# --------------------------------------------------
# 4. Compare the results
# --------------------------------------------------

torch_difference = max(
    abs(x - y)
    for x, y in zip(our_gradients, torch_gradients)
)

numerical_difference = max(
    abs(x - y)
    for x, y in zip(our_gradients, numerical_gradients)
)

max_difference = max(torch_difference, numerical_difference)

print("\n========== GRADIENT VERIFICATION ==========")
print("Difference with PyTorch:", torch_difference)
print("Difference with numerical gradients:", numerical_difference)
print("Maximum absolute difference:", max_difference)

if max_difference < 1e-6:
    print("PASS: Gradient verification successful!")
else:
    print("FAIL: Gradient difference is too large.")
