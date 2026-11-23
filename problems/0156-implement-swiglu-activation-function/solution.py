import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    x1, x2 = x.chunk(2, dim=-1)
    gate = torch.nn.functional.sigmoid(x2)
    return x2 * gate * x1