from src.data import load_mnist
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = load_mnist()

plt.title(f"Label: {train_labels[0]}")
plt.imshow(train_images[0].reshape(28, 28), cmap="gray")
plt.show()