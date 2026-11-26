import torch
import torch.nn as nn
import time
import statistics

# Create a small neural network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Create an input tensor
x = torch.tensor([[1.0, 2.0]], requires_grad=True)

# Forward pass
output = model(x)

# Display computation graph information
print("Output:", output)
print("Output shape:", output.shape)
print("Output gradient function:", output.grad_fn)

# Display previous operations in the graph
print("\nPrevious operations:")
for function in output.grad_fn.next_functions:
    print(function[0])
    # Set the model to evaluation mode
model.eval()

# Create an example input
example_input = torch.randn(1, 2)

# Create a traced version of the model
traced_model = torch.jit.trace(model, example_input)

# Warm-up both models
with torch.no_grad():
    for _ in range(20):
        model(example_input)
        traced_model(example_input)

# Measure eager execution
eager_times = []

with torch.no_grad():
    for _ in range(100):
        start = time.perf_counter()
        model(example_input)
        eager_times.append(time.perf_counter() - start)

# Measure traced execution
traced_times = []

with torch.no_grad():
    for _ in range(100):
        start = time.perf_counter()
        traced_model(example_input)
        traced_times.append(time.perf_counter() - start)

# Display median times in milliseconds
eager_median = statistics.median(eager_times) * 1000
traced_median = statistics.median(traced_times) * 1000

print("\nExecution comparison:")
print(f"Eager execution:  {eager_median:.6f} ms")
print(f"Traced execution: {traced_median:.6f} ms")
# Create a tensor
base = torch.randn(512, 512)

# Create a transposed, non-contiguous view
non_contiguous = base.t()

# Create a contiguous copy
contiguous = non_contiguous.contiguous()

print("\nTensor inspection:")
print("Base shape:", base.shape)
print("Base dtype:", base.dtype)
print("Base contiguous:", base.is_contiguous())

print("\nTransposed tensor:")
print("Shape:", non_contiguous.shape)
print("Dtype:", non_contiguous.dtype)
print("Contiguous:", non_contiguous.is_contiguous())
print("Strides:", non_contiguous.stride())

print("\nContiguous copy:")
print("Contiguous:", contiguous.is_contiguous())

# Measure transpose view creation
view_times = []

for _ in range(100):
    start = time.perf_counter()
    view = base.t()
    view_times.append(time.perf_counter() - start)

# Measure making the tensor contiguous
copy_times = []

for _ in range(100):
    start = time.perf_counter()
    copy = non_contiguous.contiguous()
    copy_times.append(time.perf_counter() - start)

print("\nNon-contiguous operation timing:")
print(
    f"Transpose view creation: "
    f"{statistics.median(view_times) * 1000:.6f} ms"
)
print(
    f"Making tensor contiguous: "
    f"{statistics.median(copy_times) * 1000:.6f} ms"
)