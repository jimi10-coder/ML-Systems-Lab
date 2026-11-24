
import torch
from thop import profile
from models import MLP, CNN, AttentionModel

# Create the three models
models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

# One input image: 1 channel, 28x28 pixels
sample = torch.randn(1, 1, 28, 28)

print("FLOPs Analysis")
print("=" * 60)

# Hand calculations for representative layers
mlp_flops = 2 * 784 * 256

cnn_flops = 2 * 3 * 3 * 1 * 16 * 28 * 28

attention_flops = 2 * 16 * 64 * (3 * 64)

print("\nHand-calculated FLOPs (one image):")
print(f"MLP first Linear layer: {mlp_flops:,}")
print(f"CNN first Conv2d layer: {cnn_flops:,}")
print(f"Attention QKV projections: {attention_flops:,}")

print("\nTHOP profiling (complete models):")

for name, model in models.items():
    model.eval()

    macs, parameters = profile(
        model,
        inputs=(sample,),
        verbose=False
    )

    print(f"\n{name}:")
    print(f"THOP MACs: {macs:,.0f}")
    print(f"THOP parameters: {parameters:,.0f}")
