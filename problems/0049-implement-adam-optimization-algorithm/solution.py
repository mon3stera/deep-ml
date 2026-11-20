import torch
import math

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    # Your code here
    m_t = 0
    v_t = 0

    for i in range(1, num_iterations + 1):
        m_t = beta1 * m_t + (1 - beta1) * grad(x0) 
        v_t = beta2 * v_t + (1 - beta2) * grad(x0) * grad(x0)
        hat_m_t = m_t / (1 - beta1 ** i)
        hat_v_t = v_t / (1 - beta2 ** i)
        x0 = x0 - learning_rate * hat_m_t / (torch.sqrt(hat_v_t) + epsilon) 

    return x0
