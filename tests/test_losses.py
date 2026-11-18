from src.losses import cross_entropy
import numpy as np

predictions = np.array([[0.1], [0.8], [0.1]])

print(cross_entropy(predictions, 0))