import torch

def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # TODO
    n_elements = pred.shape
    delta = pred - target 
    MSE = (delta * delta).sum() / pred.numel() 
    return MSE.item()
