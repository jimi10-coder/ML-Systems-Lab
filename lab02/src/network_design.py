
import numpy as np

def initialize_parameters(width):
    # Initialize weights from a normal distribution
    W1 = np.random.normal(0, 0.01, (64, width))
    b1 = np.zeros(width)

    W2 = np.random.normal(0, 0.01, (width, width))
    b2 = np.zeros(width)

    W3 = np.random.normal(0, 0.01, (width, 10))
    b3 = np.zeros(10)

    return W1, b1, W2, b2, W3, b3


np.random.seed(42)

for width in [16, 64]:
    W1, b1, W2, b2, W3, b3 = initialize_parameters(width)

    print(f"\nNetwork width: {width}")
    print("W1:", W1.shape)
    print("b1:", b1.shape)
    print("W2:", W2.shape)
    print("b2:", b2.shape)
    print("W3:", W3.shape)
    print("b3:", b3.shape)
