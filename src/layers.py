import numpy as np

class Dense:
    def __init__(self, input_size: int, output_size: int):
        heInit = np.sqrt(2 / input_size)
        
        self.weights = np.random.randn(output_size, input_size) * heInit
        self.biases = np.zeros((output_size, 1))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return self.weights @ x + self.biases
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weights = grad_output @ self.input.T
        self.grad_biases = np.sum(grad_output, axis=1, keepdims=True)
        return self.weights.T @ grad_output
    
    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases