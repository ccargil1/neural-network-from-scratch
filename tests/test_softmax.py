from src.activations import softmax
import numpy as np

x = np.array([[2.0], [1.0], [0.1]])
print(softmax(x))