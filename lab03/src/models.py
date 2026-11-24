
import torch
import torch.nn as nn


# 1. MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# 2. CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# 3. Attention Model
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Divide 28x28 image into sixteen 7x7 patches
        self.patch_projection = nn.Linear(49, 64)

        # Two Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Input: batch x 1 x 28 x 28
        batch_size = x.shape[0]

        # Split into 4x4 patches, each 7x7
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)

        # Shape: batch x 4 x 4 x 1 x 7 x 7
        x = x.contiguous().view(batch_size, 16, 49)

        # Project patches into 64 dimensions
        x = self.patch_projection(x)

        # Transformer encoder
        x = self.transformer(x)

        # Mean pooling
        x = x.mean(dim=1)

        # Classification
        return self.classifier(x)


# Test all three models
if __name__ == "__main__":
    mlp = MLP()
    cnn = CNN()
    attention = AttentionModel()

    sample = torch.randn(4, 1, 28, 28)

    print("MLP output:", mlp(sample).shape)
    print("CNN output:", cnn(sample).shape)
    print("Attention output:", attention(sample).shape)
