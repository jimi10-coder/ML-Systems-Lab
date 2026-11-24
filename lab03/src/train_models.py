
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import time
import os

from models import MLP, CNN, AttentionModel
from prepare_data import train_loader, val_loader, test_loader

device = torch.device("cpu")

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


results = []

for name, model in models.items():
    print("\n" + "=" * 50)
    print(f"Training {name}")
    print("=" * 50)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.perf_counter()

    for epoch in range(5):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        validation_accuracy = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.2f}%"
        )

    training_time = time.perf_counter() - start_time

    test_accuracy = evaluate(model, test_loader)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"\nTest accuracy: {test_accuracy:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Parameters: {parameter_count:,}")

    results.append({
        "Model": name,
        "Parameters": parameter_count,
        "Test Accuracy (%)": round(test_accuracy, 2),
        "Training Time (seconds)": round(training_time, 2)
    })

# Save results to CSV

results_folder = os.path.join(
    os.path.dirname(__file__), "..", "results"
)

os.makedirs(results_folder, exist_ok=True)

results_file = os.path.join(
    results_folder, "training_results.csv"
)

with open(results_file, "w", newline="") as file:
    fieldnames = [
        "Model",
        "Parameters",
        "Test Accuracy (%)",
        "Training Time (seconds)"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nTraining results saved to: {results_file}")

print("\nTraining results saved to lab03/results/training_results.csv")
