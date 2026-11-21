import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if not training:
        return x

    mask = torch.rand(x.shape) >= p 
    x = x * mask 
    x = x / (1 - p)
    return x
