import numpy as np

def poly2d(x, y):
    return (2 * x * y + y * y, x * x + 2 * x * y)

def exp_sum(x, y):
    return (np.exp(x + y), np.exp(x + y))

def product_sin(x, y):
    return (np.sin(y), x * np.cos(y))

def poly3d(x, y, z):
    return (2 * x * y, x * x + z * z, 2 * y * z)

def squared_error(x, y):
    return (2 * (x - y), -2 * (x - y))

f = {
    "poly2d": (2, poly2d),
    "exp_sum": (2, exp_sum),
    "product_sin": (2, product_sin),
    "poly3d": (3, poly3d),
    "squared_error": (2, squared_error),
}

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    # Your code here
    n_arguments, func = f[func_name]
    if n_arguments == 2:
        return func(point[0], point[1])
    else:
        return func(point[0], point[1], point[2])