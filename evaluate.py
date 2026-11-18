from pathlib import Path
from src.network import NeuralNetwork
from src.layers import Dense
from src.data import load_mnist
import numpy as np

model_path = Path("models/best_model.npz")
if not model_path.is_file():
    raise FileNotFoundError(f"Saved model not found: {model_path}")

_, _, test_images, test_labels = load_mnist()

test_batch = test_images.T

network = NeuralNetwork()
network.add_layer(Dense(784, 128))
network.add_layer(Dense(128, 64))
network.add_layer(Dense(64, 10))

saved_data = np.load(model_path)
for i, layer in enumerate(network.layers):
    layer.weights = saved_data[f"weights_{i}"]
    layer.biases = saved_data[f"biases_{i}"]
    
test_predictions = network.forward(test_batch)
predicted_digits = np.argmax(test_predictions, axis=0)

correct_predictions: np.ndarray = predicted_digits == test_labels
accuracy: float = np.mean(correct_predictions)

print(f"Test accuracy: {accuracy * 100:.2f}%")