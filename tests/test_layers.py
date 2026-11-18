from src.activations import relu
from src.activations import softmax
from src.layers import Dense
import numpy as np

# layer = Dense(784, 128)
# layer2 = Dense(128, 64)
# layer3 = Dense(64, 10)
# x = np.random.rand(784, 1)

# result = relu(layer.forward(x))
# result2 = layer2.forward(result)
# result3 = layer3.forward(result2)

layer = Dense(3, 2)

x = np.array([[1.0], [2.0], [3.0]])
grad_output = np.array([[0.5], [-0.3]])

forward = layer.forward(x)
backward = layer.backward(grad_output)

og_weights = layer.weights.copy()
og_biases = layer.biases.copy()
layer.update(0.01)

print(og_weights, layer.weights)
print(og_biases, layer.biases)