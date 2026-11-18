from src.layers import Dense
from src.activations import relu, softmax
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
                
        return softmax(x)