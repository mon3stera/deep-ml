# --- cell 1 ---
import numpy as np

def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * np.asarray(x)))

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # TODO: Implement your activation function
    
    gate = sigmoid(x)
    return gate * x * 1.5