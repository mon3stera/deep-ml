import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    N, _ = logits.shape

    # [N, 1]
    max_logit = torch.max(logits, dim=1, keepdim=True).values

    # [N, ]
    target_part = -logits[torch.arange(0, N).to(device=logits.device), targets]

    # [N, 1]
    max_part = max_logit 

    # [N, ]
    log_part = torch.log(torch.sum(torch.exp(logits - max_logit), dim=1))
    
    max_part = max_part.unsqueeze(-1)
    l = target_part + max_part + log_part 
    return l.mean()

