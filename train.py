from neural_network.data import load_mnist, shuffle_mnist
from neural_network.network import NeuralNetwork
from neural_network.layers import Dense
import numpy as np

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.1

train_images, train_labels, _, _ = load_mnist()

validation_images = train_images[:10_000]
validation_labels = train_labels[:10_000]

train_images = train_images[10_000:]
train_labels = train_labels[10_000:]
imagesSize = train_images.shape[0]

validation_batch = validation_images.T

network = NeuralNetwork()
network.add_layer(Dense(784, 128))
network.add_layer(Dense(128, 64))
network.add_layer(Dense(64, 10))

best_accuracy = 0.0
best_weights = [layer.weights.copy() for layer in network.layers]
best_biases = [layer.biases.copy() for layer in network.layers]

for i in range(EPOCHS):
    total_loss = 0.0
    
    train_images, train_labels = shuffle_mnist(train_images, train_labels)
    
    for j in range((imagesSize + BATCH_SIZE - 1) // BATCH_SIZE):
        start = j * BATCH_SIZE
        end = (j + 1) * BATCH_SIZE
        
        start = min(imagesSize, start)
        end = min(imagesSize, end)
        
        batch = train_images[start : end].T
        batch_labels = train_labels[start : end]
        
        total_loss += network.train_step(batch, batch_labels, LEARNING_RATE) * (end - start)

    print(f"Epoch {i + 1}/{EPOCHS}", end=" | ")

    average_loss = total_loss / imagesSize
    print(f"Training average loss: {average_loss:.4f}", end=" | ")

    validation_predictions = network.forward(validation_batch)
    predicted_digits = np.argmax(validation_predictions, axis=0)
    
    correct_predictions: np.ndarray = predicted_digits == validation_labels
    accuracy: float = np.mean(correct_predictions)
    
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        
        best_weights = [layer.weights.copy() for layer in network.layers]
        best_biases = [layer.biases.copy() for layer in network.layers]
    
model_data: dict[str, np.ndarray] = {}
for i, layer in enumerate(network.layers):
    layer.weights = best_weights[i]
    layer.biases = best_biases[i]
    
    model_data[f"weights_{i}"] = layer.weights
    model_data[f"biases_{i}"] = layer.biases
    
validation_predictions = network.forward(validation_batch)
predicted_digits = np.argmax(validation_predictions, axis=0)

correct_predictions: np.ndarray = predicted_digits == validation_labels
accuracy: float = np.mean(correct_predictions)

print(f"Best accuracy: {best_accuracy * 100:.2f}%")
print(f"Restored accuracy: {accuracy * 100:.2f}%")

np.savez("models/best_model.npz", **model_data)  # pyright: ignore[reportArgumentType]