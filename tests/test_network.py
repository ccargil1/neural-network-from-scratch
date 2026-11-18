from src.network import NeuralNetwork
from src.layers import Dense
import numpy as np

network = NeuralNetwork()
network.add_layer(Dense(784, 128))
network.add_layer(Dense(128, 64))
network.add_layer(Dense(64, 10))

x = np.random.rand(784, 1)
predictions = network.forward(x)

print(predictions.shape)
print(predictions.sum())
print(predictions)