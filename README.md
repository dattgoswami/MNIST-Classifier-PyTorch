# MNIST Classification with PyTorch

A simple convolutional neural network (CNN) using PyTorch to classify the MNIST dataset of handwritten digits.

## Requirements:

- Python 3
- PyTorch
- torchvision

```
pip install torch torchvision
```

## Model Architecture:

- Convolutional layer with 32 filters, kernel size 3x3
- ReLU activation function
- Convolutional layer with 64 filters, kernel size 3x3
- ReLU activation function
- Max pooling with 2x2 kernel size
- Dropout (25%)
- Fully connected layer with 9216 input features and 128 output features
- ReLU activation function
- Dropout (50%)
- Fully connected layer with 128 input features and 10 output features (one for each digit 0-9)

## Dataset:

Uses the MNIST dataset which consists of handwritten digits. The dataset is automatically downloaded using torchvision's datasets module.

## Training:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Cross-Entropy Loss
- Number of epochs: 5
- Training batch size: 32

## Testing:

- Test batch size: 1000

## Usage:

Run the script using the following command:

```
python3 mnist_pytorch.py
```

## Expected Output:

After training and testing the model on the MNIST dataset, the script will output the average loss and accuracy on the test set. For example:

```
Test set: Average loss: 0.0001, Accuracy: 9829/10000 (98%)
```

This indicates that the model achieved an accuracy of 98% on the MNIST test dataset.

## Notes:

- The script uses two different types of dropout (25% and 50%) to reduce overfitting.
- The script uses the Adam optimizer for faster convergence.
- The model is trained for 5 epochs. This value can be adjusted for more or less training as needed.

## Modifications:

To experiment with the model, you can modify various parameters including the number of filters in the convolutional layers, the dropout rates, the optimizer settings, and the number of training epochs.
