import time
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from autograd_engine import Value

# 1. Create the dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset loaded successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input features:", X_train.shape[1])

# Create one neuron
def neuron(inputs, weights, bias):
    result = sum((x * w for x, w in zip(inputs, weights)), bias)
    return result.relu()


# Create the neural network: 2 -> 8 -> 1
class TinyNeuralNetwork:

    def __init__(self):
        self.weights1 = [
            [Value(np.random.uniform(-0.5, 0.5)) for _ in range(2)]
            for _ in range(8)
        ]
        self.biases1 = [Value(0.0) for _ in range(8)]

        self.weights2 = [
            Value(np.random.uniform(-0.5, 0.5))
            for _ in range(8)
        ]
        self.bias2 = Value(0.0)

    def forward(self, inputs):
        hidden = [
            neuron(inputs, weights, bias)
            for weights, bias in zip(self.weights1, self.biases1)
        ]

        output = sum(
            (h * w for h, w in zip(hidden, self.weights2)),
            self.bias2
        )

        return output.sigmoid()

    def parameters(self):
        params = []
        for weights in self.weights1:
            params.extend(weights)
        params.extend(self.biases1)
        params.extend(self.weights2)
        params.append(self.bias2)
        return params

def binary_cross_entropy(prediction, target):
    epsilon = 1e-7

    return -(
        target * (prediction + epsilon).log()
        + (1 - target) * (1 - prediction + epsilon).log()
    )


def train_custom_model(epochs=20, learning_rate=0.05):
    model = TinyNeuralNetwork()

    start_time = time.perf_counter()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, label in zip(X_train, y_train):
            inputs = [
                Value(float(x[0])),
                Value(float(x[1]))
            ]

            prediction = model.forward(inputs)
            target = Value(float(label))

            loss = binary_cross_entropy(prediction, target)
            total_loss += loss.data

            # Calculate gradients
            loss.backward()

            # Update parameters
            for parameter in model.parameters():
                parameter.data -= learning_rate * parameter.grad
                parameter.grad = 0.0

        if (epoch + 1) % 5 == 0:
            print(
                f"Custom model - Epoch {epoch + 1}, "
                f"Loss: {total_loss / len(X_train):.4f}"
            )

    elapsed_time = time.perf_counter() - start_time

    return model, elapsed_time

class PyTorchNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch_model(epochs=20, learning_rate=0.05):
    torch.manual_seed(42)

    model = PyTorchNeuralNetwork()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(
        y_train, dtype=torch.float32
    ).view(-1, 1)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    start_time = time.perf_counter()

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(len(X_tensor)):
            inputs = X_tensor[i].view(1, 2)
            target = y_tensor[i].view(1, 1)

            prediction = model(inputs)
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"PyTorch - Epoch {epoch + 1}, "
                f"Loss: {total_loss / len(X_tensor):.4f}"
            )

    elapsed_time = time.perf_counter() - start_time

    return model, elapsed_time


# Test the network and loss

# Benchmark both models over 5 runs
custom_times = []
pytorch_times = []

for run in range(5):
    print(f"\nBenchmark run {run + 1}/5")

    _, custom_time = train_custom_model()
    _, torch_time = train_pytorch_model()

    custom_times.append(custom_time)
    pytorch_times.append(torch_time)

# Calculate median times
custom_median = statistics.median(custom_times)
pytorch_median = statistics.median(pytorch_times)

print("\n========== FINAL BENCHMARK ==========")

print("Custom engine times:", custom_times)
print("PyTorch times:", pytorch_times)

print("Custom engine median:", round(custom_median, 4), "seconds")
print("PyTorch median:", round(pytorch_median, 4), "seconds")

if custom_median < pytorch_median:
    print("Custom engine was faster in this benchmark.")
else:
    print("PyTorch was faster in this benchmark.")

    