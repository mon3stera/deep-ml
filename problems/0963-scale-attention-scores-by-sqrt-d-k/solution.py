import numpy as np
import math 

def softmax(a: np.ndarray, dim: int) -> np.ndarray:
    maximum = np.max(a, axis=dim, keepdims=True)
    e_x = np.exp(a - maximum)
    return e_x / np.sum(e_x, axis=dim, keepdims=True)

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    _, d_k = Q.shape
    scores = Q @ K.T 
    scores = scores / math.sqrt(d_k) 
    scores = softmax(scores, dim=-1)
    return np.round(scores, 4).tolist()
    
