import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    # [..., 1]
    e = x.mean(dim=-1, keepdim=True)

    E_x_pow = (x ** 2).mean(dim=-1, keepdim=True)
    pow_E_x = e * e 

    # [..., 1]
    var = E_x_pow - pow_E_x
    return gamma * (x - e) / torch.sqrt(var + eps) + beta  