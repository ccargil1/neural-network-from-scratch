# Neural Network from Scratch
A fully connected neural network built from scratch using **Python and NumPy** to recognize handwritten digits from the MNIST dataset.
This project implements the fundamental mathematics behind neural networks, including forward propagation, backpropagation, activation functions, cross-entropy loss, and gradient descent.
**No machine learning frameworks such as TensorFlow or PyTorch were used.**

## Results
The neural network achieved **97.71% accuracy** on the MNIST test dataset, correctly classifying **9,771 out of 10,000 handwritten digits.**
| Metric | Result |
| --- | --- |
| Dataset | MNIST |
| Training Samples | 50,000 |
| Validation Samples | 10,000 |
| Test Samples | 10,000 |
| Best Validation Accuracy | 97.69% |
| Test Accuracy | 97.71% |

## Network Architecture
The network consists of an input layer, two hidden layers, and an output layer.
| Layer | Neurons | Activation |
| --- | --- | --- |
| Input | 784 | None |
| Hidden 1 | 128 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 10 | Softmax |

Each input is a **28 x 28 grayscale image,** flattened into a vector of 784 normalized pixel values.
The output layer produces a probability distribution across the ten possible digits (0-9).

## Training Configuration
| Hyperparameter | Value |
| --- | --- |
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 0.1 |
| Optimizer | Mini-batch Gradient Descent |
| Loss Function | Cross-Entropy |
| Weight Initialization | He Initialization |

## Implementation
The neural network was implemented using NumPy for matrix operations, without relying on automatic differentitation or high-level machine learning frameworks.

## Dense Layers
Each fully connected layer performs the transformation:
```python
Z = WA + b
```
Where:
- W represents the weight matrix
- A represents the input activations
- b represents the bias vector.
- Z represents the resulting weighted sums.

## Activation Functions
**ReLU (Rectified Linear Unit)** is used in the hidden layers to introduce nonlinearity.
```python
ReLU(x) = max(0, x)
```
**Softmax** is used in the output layer to convert the final outputs into class probabilities.

## Loss Function
Cross-entropy loss measures the difference between the network's predicted probabilities and the correct digit labels.

## Backpropagation
Backpropagation calculates the gradients of the loss with respect to each layer's weights and biases using the chain rule.
These gradients are used during gradient descent to update the network's parameters and improve its predictions.

## Additional Features
- **Mini-batch training:** Processes multiple samples simultaneously using vectorized NumPy operations.
- **He initialization:** Initializes weights based on the number of inputs to each layer.
- **Numerically stable Softmax:** Improves numerical stability when calculating output probabilities.
- **Data shuffling:** Randomizes training samples between epochs.
- **Validation evaluation:** Measures performance on a separate validation dataset after each epoch.
- **Model checkpointing:** Preserves the weights and biases from the epoch with the highest validation accuracy.
- **Model serialization:** Saves trained parameters to a `.npz` file for later evaluation.

## Getting Started
### Requirements
- Python 3
- NumPy
- python-mnist

### Installation
Clone the repository:
```bash
git clone https://github.com/ccargil1/neural-network.git cd neural-network
```
Create a virtual environment:
```bash
python -m venv .venv
```
Activate the virtual environment.
**Windows:**
```bash
.venv\Scripts\activate
```
**macOS / Linux:**
```bash
source .venv/bin/activate
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
### MNIST Dataset
Download the MNIST dataset and extract the following files into the directory expected by `src/data.py`:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## Usage
### Training
To train the neural network:
```bash
python train.py
```
The training script:
1. Loads and normalizes the MNIST dataset.
2. Splits the training dataset into training and validation subsets.
3. Initializes the neural network.
4. Trains the network using mini-batch gradient descent.
5. Evaluates validation accuracy after each epoch.
6. Restores the parameters from the best-performing epoch.
7. Saves the trained model parameters.

### Evaluation
To evaluate the trained model:
```bash
python evaluate.py
```
The evaluation script loads the saved model parameters and calculates classification accuracy on the MNIST test dataset.

### File Descriptions
| File | Description |
| --- | --- |
| layers.py | Dense layer implementation, gradient calculations, and parameter updates |
| activations.py | ReLU, ReLU derivative, and Softmax |
| losses.py | Cross-entropy loss calculation |
| network.py | Neural network forward propagation, backpropagation, and training |
| data.py | MNIST loading, normalization, and shuffling |
| train.py | Training pipeline, validation, and model checkpointing |
| evaluate.py | Model loading and test accuracy evaluation |

## What I Learned
Building this project strengthened my understanding of:
- Linear algebra and matrix operations in machine learning.
- The mathematical foundations of neural networks.
- Forward propagation and backpropagation.
- Gradient-based optimization.
- Training, validation, and test dataset separation.
- Numerical stability and vectorized computation.
- Saving and restoring trained model parameters.

Implementing the network from scratch provided a deeper understanding of how neural networks learn, rather than relying on abstractions provided by existing machine learning frameworks.