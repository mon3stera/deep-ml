import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    t = torch.tensor(x_value, requires_grad=True)
    y = t * t + 3 * t + 2
    y.backward()
    return t.grad.item()
