
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load the handwritten digits dataset
X, y = load_digits(return_X_y=True)

# Scale pixel values from 0-16 to 0-1
X = X / 16.0

# One-hot encode the labels
Y = np.eye(10)[y]

# Split into 70% training and 30% testing data
X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
    X,
    y,
    Y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Display information about the dataset
print("Original dataset shape:", X.shape)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training labels shape:", Y_train.shape)
print("Testing labels shape:", Y_test.shape)
print("Minimum pixel value:", X.min())
print("Maximum pixel value:", X.max())
print("First one-hot label:", Y_train[0])
print("Sum of first one-hot label:", Y_train[0].sum())
