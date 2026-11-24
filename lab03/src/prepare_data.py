
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 1. Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# 2. Download and load the official training dataset
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 3. Split into 55,000 training and 5,000 validation images
train_data, val_data = random_split(
    train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

# 4. Load the official test dataset (10,000 images)
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 5. Create DataLoaders with batch size 128
batch_size = 128

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 6. Display dataset information
print("Fashion-MNIST Dataset Preparation")
print("----------------------------------")
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_dataset))

# 7. Check one batch
images, labels = next(iter(train_loader))

print("\nBatch information:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Image data type:", images.dtype)
print("Image minimum:", images.min().item())
print("Image maximum:", images.max().item())
