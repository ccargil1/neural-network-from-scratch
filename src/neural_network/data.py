from mnist import MNIST
from typing import cast
import numpy as np

SEED = 42

def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist = MNIST("data/mnist")
    
    train_images, train_labels = cast(
        tuple[list[list[int]], list[int]],
        mnist.load_training()
    )

    test_images, test_labels = cast(
        tuple[list[list[int]], list[int]],
        mnist.load_testing()
    )
    
    train_images = np.array(train_images) / 255.0
    train_labels = np.array(train_labels)
    test_images = np.array(test_images) / 255.0
    test_labels = np.array(test_labels)
    
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(train_images))
    
    train_images = train_images[indices]
    train_labels = train_labels[indices]
        
    return train_images, train_labels, test_images, test_labels

def shuffle_mnist(train_images: np.ndarray, train_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.random.permutation(len(train_images))
        
    train_images = train_images[indices]
    train_labels = train_labels[indices]
    
    return train_images, train_labels