
import torch
from torchinfo import summary

from models import MLP, CNN, AttentionModel

# Create the three models
models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

# Check the number of parameters
for name, model in models.items():
    print("\n" + "=" * 50)
    print(f"{name} MODEL")
    print("=" * 50)

    model_summary = summary(
        model,
        input_size=(1, 1, 28, 28),
        verbose=0
    )

    print(model_summary)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_parameters:,}")
