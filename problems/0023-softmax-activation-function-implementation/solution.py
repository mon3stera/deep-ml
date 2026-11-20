import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores = torch.tensor(scores, dtype=torch.float32)
    max_value = torch.max(scores)
    scores = scores - max_value 
    exp_s = torch.exp(scores)
    scale = torch.sum(exp_s)
    scores = exp_s / scale 
    return scores.tolist()

