import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    shifted: np.ndarray = x - np.max(x)
    exp_values = np.exp(shifted)
    
    total = np.sum(exp_values)
    
    return exp_values / total