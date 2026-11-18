from src.network import NeuralNetwork
from src.layers import Dense
import numpy as np

network = NeuralNetwork()
network.add_layer(Dense(784, 128))
network.add_layer(Dense(128, 64))
network.add_layer(Dense(64, 10))

x = np.random.rand(784, 1)
predictions = network.forward(x)

for i in range(100):
    loss = network.train_step(x, 3, 0.001)
    if i == 0 or i == 99:
        print(loss)
        
network.forward(x)
print(network.predictions[3])