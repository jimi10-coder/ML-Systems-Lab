
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from backpropagation import (
    initialize_parameters,
    forward_pass,
    backward_pass,
    cross_entropy_loss,
)

# Paths for saving results
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load and prepare the dataset
X, y = load_digits(return_X_y=True)
X = X / 16.0
Y = np.eye(10)[y]

X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
    X, y, Y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 2. Experiment settings
widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32

all_losses = {}
results = []

# 3. Train all six configurations
for width in widths:
    for learning_rate in learning_rates:
        print(
            f"\nTraining: width={width}, "
            f"learning_rate={learning_rate}"
        )

        # Use a reproducible initialization for each experiment
        np.random.seed(42)
        parameters = initialize_parameters(width)
        epoch_losses = []

        for epoch in range(epochs):
            # Shuffle the training examples
            rng = np.random.default_rng(42 + epoch)
            indices = rng.permutation(len(X_train))
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            batch_losses = []

            # Mini-batch gradient descent
            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                P, cache = forward_pass(X_batch, parameters)
                loss = cross_entropy_loss(P, Y_batch)
                gradients = backward_pass(Y_batch, parameters, cache)

                # Update all weights and biases
                for parameter, gradient in zip(parameters, gradients):
                    parameter -= learning_rate * gradient

                batch_losses.append(loss)

            # Calculate loss on the complete training set
            P_train, _ = forward_pass(X_train, parameters)
            train_loss = cross_entropy_loss(P_train, Y_train)
            epoch_losses.append(train_loss)

            print(
                f"Epoch {epoch + 1:02d}/{epochs}, "
                f"loss={train_loss:.6f}"
            )

        # 4. Evaluate on the test set
        P_test, _ = forward_pass(X_test, parameters)
        predictions = np.argmax(P_test, axis=1)
        accuracy = np.mean(predictions == y_test) * 100

        key = f"W={width}, lr={learning_rate}"
        all_losses[key] = epoch_losses

        results.append({
            "width": width,
            "learning_rate": learning_rate,
            "final_train_loss": epoch_losses[-1],
            "test_accuracy_percent": accuracy,
        })

        print(f"Final training loss: {epoch_losses[-1]:.6f}")
        print(f"Test accuracy: {accuracy:.2f}%")

# 5. Plot all six loss curves
plt.figure(figsize=(10, 6))

for label, losses in all_losses.items():
    plt.plot(range(1, epochs + 1), losses, label=label)

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training Loss for Different Network Configurations")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = RESULTS_DIR / "loss_curves.png"
plt.savefig(plot_path, dpi=150)
plt.close()

# 6. Save the results in a CSV file
csv_path = RESULTS_DIR / "training_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "width",
            "learning_rate",
            "final_train_loss",
            "test_accuracy_percent",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("\n========== FINAL RESULTS ==========")

for result in results:
    print(
        f"Width={result['width']:2d}, "
        f"LR={result['learning_rate']:.3f}, "
        f"loss={result['final_train_loss']:.6f}, "
        f"test accuracy={result['test_accuracy_percent']:.2f}%"
    )

print(f"\nLoss plot saved to: {plot_path}")
print(f"Results saved to: {csv_path}")
