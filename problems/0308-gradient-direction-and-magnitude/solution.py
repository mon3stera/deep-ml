import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    # Your code here
    gradient = np.array(gradient)
    
    magnitude = np.sqrt((gradient ** 2).sum())
    direction = gradient / (magnitude + 1e-12) 
    descent_direction = -direction
    return {
        "magnitude": magnitude,
        "direction": direction,
        "descent_direction": descent_direction,
    }