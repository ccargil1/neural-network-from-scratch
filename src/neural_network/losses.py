import numpy as np

def cross_entropy(predictions: np.ndarray, target: np.ndarray) -> float:
    batchSize: int = predictions.shape[1]
    indices = np.arange(batchSize)
    
    p_correct = predictions[target, indices]
    clipped = np.clip(p_correct, (10 ** -15), 1)
    
    return -np.mean(np.log(clipped))