from src.activations import relu
from src.activations import softmax
from src.layers import Dense
import numpy as np

layer = Dense(784, 128)
layer2 = Dense(128, 64)
layer3 = Dense(64, 10)
x = np.random.rand(784, 1)

result = relu(layer.forward(x))
result2 = layer2.forward(result)
result3 = layer3.forward(result2)

print(softmax(result3))