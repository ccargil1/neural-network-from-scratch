from src.activations import relu
from src.activations import relu_derivative
import numpy as np

x = np.array([-3, -1, 0, 2, 5])

print(relu_derivative(x))