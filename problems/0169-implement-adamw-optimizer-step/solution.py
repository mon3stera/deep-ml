import torch
import math 

def adamw_update(w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, t: int, lr: float, beta1: float, beta2: float, epsilon: float, weight_decay: float) -> tuple:
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter tensor (torch.Tensor)
      g: gradient tensor (torch.Tensor)
      m: first moment tensor (torch.Tensor)
      v: second moment tensor (torch.Tensor)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # Your code here
    m_t = beta1 * m + (1 - beta1) * g 
    v_t = beta2 * v + (1 - beta2) * g * g 
    hat_m_t = m_t / (1 - math.pow(beta1, t))
    hat_v_t = v_t / (1 - math.pow(beta2, t))
    w_new = (1 - lr * weight_decay) * w - lr * hat_m_t / (torch.sqrt(hat_v_t) + epsilon) 
    return w_new, m_t, v_t 