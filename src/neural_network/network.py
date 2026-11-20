from src.neural_network.layers import Dense
from src.neural_network.activations import relu, relu_derivative, softmax
from src.neural_network.losses import cross_entropy
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers: list[Dense] = []
        self.pre_activations: list[np.ndarray] = []
        
    def add_layer(self, layer: Dense) -> None:
        self.layers.append(layer)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        length = len(self.layers)
        
        self.pre_activations.clear()
        
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            self.pre_activations.append(x)
            
            if i + 1 != length:
                x = relu(x)
        
        self.predictions = softmax(x)
        return self.predictions
    
    def backward(self, target: np.ndarray) -> None:
        length = len(self.layers)
        
        batchSize: int = self.predictions.shape[1]
        indices = np.arange(batchSize)
        
        target_vector = np.zeros_like(self.predictions)
        target_vector[target, indices] = 1
        
        grad = (self.predictions - target_vector) / batchSize
        
        for i, layer in enumerate(reversed(self.layers)):
            index = length - 1 - i
            grad = layer.backward(grad)
            
            if index > 0:
                grad *= relu_derivative(self.pre_activations[index - 1])
                
    def update(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update(learning_rate)
            
    def train_step(self, x: np.ndarray, target: np.ndarray, learning_rate: float) -> float:
        predictions = self.forward(x)
        loss = cross_entropy(predictions, target)
        
        self.backward(target)
        self.update(learning_rate)
        
        return loss