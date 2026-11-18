import numpy as np

def cross_entropy(predictions: np.ndarray, target: int) -> float:
    p_correct: float = predictions[target][0]
    clipped = np.clip(p_correct, (10 ** -15), 1)
    
    return -np.log(clipped)