import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # Implement your code here
    v1, v2 = v1.to(dtype=torch.float32), v2.to(dtype=torch.float32)
    dot_p = torch.dot(v1, v2)
    norm = torch.linalg.norm(v1) * torch.linalg.norm(v2)
    return (dot_p / norm).item()