import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply RMSNorm to the input tensor.

    Parameters:
        x   : torch.Tensor of shape (batch_size, features)
        g   : torch.Tensor of shape (features,) - gain parameter
        eps : float - small constant for numerical stability

    Returns:
        torch.Tensor of same shape as x
    """
    _, d = x.shape
    rms = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / d  + eps)
    return g * x / rms