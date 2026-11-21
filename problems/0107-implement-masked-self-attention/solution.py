import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # Your code here
    
    seq_len, d_k = Q.shape 

    qk_t = Q @ K.transpose(0, 1)
    qk_t = qk_t / d_k + mask  

    scores = torch.nn.functional.softmax(qk_t, dim=1)
    out = scores @ V 
    return out 