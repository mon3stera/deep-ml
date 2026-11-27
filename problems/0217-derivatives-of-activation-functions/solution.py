import math 

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2

def relu(x):
    return 1.0 if x > 0.0 else 0.0  

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    return {
        "sigmoid": sigmoid_derivative(x),
        "tanh": tanh_derivative(x),
        "relu": relu(x),
    }
    