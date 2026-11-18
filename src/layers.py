import numpy as np

class Dense:
    def __init__(self, input_size: int, output_size: int):
        heInit = (2 / input_size) ** 0.5
        
        self.weights = np.random.randn(output_size, input_size) * heInit
        self.biases = np.zeros((output_size, 1))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.weights @ x + self.biases