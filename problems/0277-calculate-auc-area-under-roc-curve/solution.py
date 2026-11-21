import torch

def calculate_auc(y_true, y_scores) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) using PyTorch.
    
    Args:
        y_true: List, array, or torch.Tensor of binary ground truth labels (0 or 1)
        y_scores: List, array, or torch.Tensor of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    pos_samples = y_scores[y_true == 1]
    neg_samples = y_scores[y_true == 0]
    pos_samples = pos_samples[:, None]
    neg_samples = neg_samples[None, :]
    delta = pos_samples - neg_samples
    win = (delta > 0).float() + 0.5 * (delta == 0).float()
    return win.mean().item()