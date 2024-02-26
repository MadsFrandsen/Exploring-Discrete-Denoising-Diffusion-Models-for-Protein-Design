import torch
import torch.nn.functional as F




def discrete_kl_logits(logits1, logits2, eps=1.e-6):
    out = (
        F.softmax(logits1 + eps, dim=-1) *
        (F.log_softmax(logits1 + eps, dim=-1) -
         F.log_softmax(logits2 + eps, dim=-1)))
    return torch.sum(out, dim=-1)

def discrete_kl_probs(probs1, probs2, eps=1.e-6):
    pass