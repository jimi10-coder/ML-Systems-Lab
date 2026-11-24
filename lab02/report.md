# Lab 2 — Neural Network from Scratch

## 1. Dataset Preparation

I used the Digits dataset from scikit-learn. The dataset contains 1,797 images, and each image is represented by 64 input features. I scaled the pixel values by dividing them by 16 and converted the labels into one-hot encoded vectors. I split the data into 70% training data and 30% testing data.

## 2. Network Design and Forward Pass

I implemented a neural network with two hidden layers using ReLU activation and an output layer using softmax. I tested two hidden-layer widths: 16 and 64 neurons. I checked that the output probabilities summed to 1 for each input example.

## 3. Loss Function

I used cross-entropy loss to measure the difference between the predicted probabilities and the correct labels. Before training, the loss was approximately 2.3026, which is close to ln(10), as expected for a network with ten classes and nearly uniform initial predictions.

## 4. Backpropagation and Gradient Checking

I implemented backpropagation to calculate gradients for the network's weights and biases. I compared analytical gradients with numerical gradients using small changes to selected weights. The gradient check passed 9 out of 10 checks, meeting the lab requirement.

## 5. Training Experiments

## 5. Training Experiments and Results

I trained the neural network using two hidden-layer widths, 16 and 64 neurons, and three learning rates: 0.001, 0.01 and 0.1. Each configuration was trained for 50 epochs using mini-batches of 32 examples.

The results are shown below.

| Hidden width | Learning rate | Final training loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.3025 | 10.19% |
| 16 | 0.01 | 2.3025 | 10.19% |
| 16 | 0.1 | 1.0616 | 58.70% |
| 64 | 0.001 | 2.3025 | 10.19% |
| 64 | 0.01 | 2.3023 | 10.19% |
| 64 | 0.1 | 0.1561 | 93.15% |

The learning rate had a strong effect on the results. With learning rates of 0.001 and 0.01, the networks did not learn effectively during the 50 epochs. With a learning rate of 0.1, both networks improved. The network with 64 hidden neurons achieved the best result, with 93.15% test accuracy and a final training loss of 0.1561.

I saved the loss curves in `results/loss_curves.png` and the numerical results in `results/training_results.csv`.

## 6. Timing Analysis

I measured the forward and backward passes using a network with 64 hidden neurons and a batch size of 32. I performed 100 measurements and calculated the median execution time.

The median forward-pass time was 0.056200 ms, while the median backward-pass time was 0.071900 ms. The backward pass was approximately 28% slower. Both passes use matrix multiplication, but the backward pass also calculates gradients and propagates errors through the network. These additional operations can explain why it took longer in my experiment.


## 7. Conclusion

In this lab, I implemented a neural network from scratch using NumPy. I learned how to prepare data, perform forward propagation, calculate cross-entropy loss and update weights using backpropagation. I also checked the gradients numerically, and 9 out of 10 checks passed.

The experiments showed that the learning rate and network width affected the results. The best configuration was a network with 64 hidden neurons and a learning rate of 0.1, which achieved 93.15% test accuracy. The timing experiment also showed that the backward pass was slower than the forward pass in my measurements.

Overall, this lab helped me understand the basic operations of a neural network and how its configuration affects training performance.
