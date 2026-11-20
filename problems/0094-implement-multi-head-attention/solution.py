import torch
import torch.nn.functional as F
import math
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    return (
        X @ W_q,
        X @ W_k,
        X @ W_v,
    )

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    seq_len, d_k = Q.shape
    
    qk_t = Q @ K.transpose(0, 1)
    scores = F.softmax(qk_t / math.sqrt(d_k), dim=1)
    return scores @ V 

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_k = Q.shape

    Q = Q.reshape((seq_len, n_heads, d_k // n_heads)).transpose(0, 1)
    K = K.reshape((seq_len, n_heads, d_k // n_heads)).transpose(0, 1)
    V = V.reshape((seq_len, n_heads, d_k // n_heads)).transpose(0, 1)
    
    qk_t = Q @ K.transpose(-1, -2)
    scores = F.softmax(qk_t / math.sqrt(d_k // n_heads), dim=-1)
    scores = scores @ V
    scores = scores.transpose(0, 1).contiguous().view((seq_len, d_k))
    return scores
    