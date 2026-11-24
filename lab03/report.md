# Lab 3: Architectures and Their Costs

## 1. Goal

The goal of this laboratory was to implement and compare three neural network architectures: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and an Attention-based model.

The experiment focused on classification accuracy, model parameters, computational cost, and inference latency using the Fashion-MNIST dataset.

## 2. Method

### Dataset

Fashion-MNIST was loaded using PyTorch and torchvision.

- Training dataset: 55,000 images
- Validation dataset: 5,000 images
- Test dataset: 10,000 images
- Image dimensions: 28 × 28 pixels
- Batch size: 128
- Normalization: `Normalize((0.2860,), (0.3530,))`

### Model architectures

Three models were implemented:

- **MLP:** Flatten layer followed by Linear layers of 256, 128, and 10 neurons.
- **CNN:** Two convolutional layers with 16 and 32 channels, followed by max pooling and a classification layer.
- **Attention:** Images divided into 16 patches of 7 × 7 pixels, projected to 64 dimensions, followed by two Transformer encoder layers and a classification layer.

### Training configuration

All models were trained using the same settings:

- Epochs: 5
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Loss function: Cross-Entropy Loss

### Inference latency

Latency was measured on a CPU using:

- Batch size 1: 100 measurements
- Batch size 128: 20 measurements
- Median latency was reported.

## 3. Results

### 3.1 Parameter count

| Model | Parameters |
|---|---:|
| MLP | 235,146 |
| CNN | 20,490 |
| Attention | 70,794 |

The CNN has the smallest number of parameters. The MLP has the largest number.

The architectures were implemented according to the specified layer configurations. However, their resulting parameter counts do not fall within the laboratory's suggested target range of 100,000–150,000 parameters.

### 3.2 FLOPs and computational cost

The following representative operations were calculated manually:

| Architecture | Representative operation | Hand-calculated FLOPs |
|---|---|---:|
| MLP | Linear 784 → 256 | 401,408 |
| CNN | First convolutional layer | 225,792 |
| Attention | QKV projections | 393,216 |

THOP profiling of the complete models produced:

| Model | THOP MACs | THOP parameters |
|---|---:|---:|
| MLP | 234,752 | 235,146 |
| CNN | 1,031,744 | 20,490 |
| Attention | 591,488 | 37,514 |

The hand calculations and THOP results are not directly equivalent because the hand calculations cover individual operations, while THOP profiles complete models. Also, MACs and FLOPs use different counting conventions: one multiply-accumulate is commonly counted as one MAC or approximately two FLOPs.

For the Attention model, THOP did not count all Transformer parameters and operations. Therefore, the parameter count from `torchinfo.summary` (70,794) is used as the complete model parameter count.

### 3.3 Inference latency

| Model | Batch size 1 (ms) | Batch size 128 (ms) |
|---|---:|---:|
| MLP | 0.2266 | 0.7608 |
| CNN | 0.4311 | 6.6082 |
| Attention | 0.5915 | 3.3671 |

The MLP had the lowest inference latency for both batch sizes in this experiment. The CNN was the slowest for batch size 128, while Attention was slower than MLP for batch size 1.

### 3.4 Training accuracy and time

| Model | Test accuracy | Training time |
|---|---:|---:|
| MLP | 87.57% | 75.01 s |
| CNN | 89.90% | 183.40 s |
| Attention | 84.13% | 218.83 s |

The CNN achieved the highest test accuracy of 89.90%. The Attention model achieved 84.13%, while the MLP achieved 87.57%.

### 3.5 Comparison graphs

The following graphs were generated:

- `results/accuracy_vs_parameters.png`
- `results/accuracy_vs_latency.png`

They compare the test accuracy of the three architectures against their parameter counts and measured inference latency.

## 4. Discussion

The CNN achieved the best classification accuracy while having the smallest number of parameters. This suggests that convolutional layers are effective for image classification.

One reason is parameter sharing. A convolutional filter is applied across different image locations instead of learning separate weights for every location. CNNs also use local connections, which help them identify local image patterns such as edges and shapes.

The MLP was the fastest model in this experiment. However, it had the largest number of parameters. It processes flattened images using fully connected layers, which do not directly preserve the spatial structure of the image.

The Attention model had more computational and training costs than the MLP in this experiment. Its attention mechanism compares information between image patches. This can help capture relationships between different parts of an image, but it also introduces additional computation.

The results show that having more parameters does not automatically produce higher accuracy. The CNN achieved the best accuracy with far fewer parameters than the MLP.

The latency results also show that performance depends on batch size and architecture. The CNN was slower for batch size 128 in our CPU measurements, even though it achieved the highest accuracy.

Overall, the CNN was the most cost-effective architecture when considering test accuracy and parameter count. The MLP was the best choice when inference speed was the main priority.

## 5. Conclusions

1. The CNN achieved the highest test accuracy (89.90%) and had the smallest number of parameters (20,490).
2. The MLP provided the fastest inference for both tested batch sizes, but it required the most parameters (235,146).
3. The Attention model had the lowest test accuracy (84.13%) and the longest training time (218.83 seconds) in this experiment.

## 6. Limitations

The experiments were performed on a CPU, so the latency results may be different on a GPU or another computer.

All models were trained for only five epochs. More training or hyperparameter tuning could change their final accuracy.

THOP did not fully account for the Attention model's Transformer operations and parameters, so its profiling results should be interpreted with caution.

The suggested parameter target of 100,000–150,000 was not achieved by all three prescribed architectures. Their specified layer configurations were kept unchanged for this comparison.