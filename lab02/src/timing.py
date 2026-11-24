
import time
import statistics
import numpy as np
from sklearn.datasets import load_digits

from backpropagation import (
    initialize_parameters,
    forward_pass,
    backward_pass,
)

# Load a batch of real digit images
X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]

X_batch = X[:32]
Y_batch = Y[:32]

# Initialize the network
np.random.seed(42)
parameters = initialize_parameters(64)

# Prepare the forward pass and cache
P, cache = forward_pass(X_batch, parameters)

# Warm-up runs, not included in the measurements
for _ in range(10):
    P, cache = forward_pass(X_batch, parameters)
    gradients = backward_pass(Y_batch, parameters, cache)

# Measure forward pass separately
forward_times = []

for _ in range(100):
    start = time.perf_counter()
    P, cache = forward_pass(X_batch, parameters)
    end = time.perf_counter()
    forward_times.append((end - start) * 1000)

# Measure backward pass separately
P, cache = forward_pass(X_batch, parameters)
backward_times = []

for _ in range(100):
    start = time.perf_counter()
    gradients = backward_pass(Y_batch, parameters, cache)
    end = time.perf_counter()
    backward_times.append((end - start) * 1000)

# Calculate median times in milliseconds
forward_median = statistics.median(forward_times)
backward_median = statistics.median(backward_times)

print("Network width: 64")
print("Batch size: 32")
print("Measurements: 100 runs")
print(f"Forward pass median:  {forward_median:.6f} ms")
print(f"Backward pass median: {backward_median:.6f} ms")

if forward_median > backward_median:
    print("The forward pass was slower in this measurement.")
elif backward_median > forward_median:
    print("The backward pass was slower in this measurement.")
else:
    print("Both passes had the same median time.")

print(
    "\nBoth passes use matrix multiplication. "
    "The backward pass also calculates gradients and propagates "
    "errors through the network, so it may take longer. "
    "The actual result depends on the hardware and matrix sizes."
)
