import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from utils import Tokenizer
from constants import PROTEIN_ALPHABET

def sample_prior(a, b, len=len(PROTEIN_ALPHABET)):
    """
    Returns prior for KL at T -> inf with same shape as q over total possible values
    Prior is a uniform distribution over number of values
    """
    prior = torch.empty(a, b)
    prior = torch.ones_like(prior) / len
    return prior


class D3PMLVBLoss(KLDivLoss):
    """
    Shape:
        Inputs:
            - src_one_hot: (B, L, K) original seq one hot encoded
            - q: (B, L, K) forward prob dist
            - predictions: (B, L, K) model predictions
            - tgt: (B, L) corrupted sequence tokenized
            - tgt_one_hot: (B, L, K) corrupted sequence one hot encoded
            - timestep (B)
            - Q (K, K) transition matrix
            - Q_bar (K, K) transition matrix accounting for time
        
        Returns:
            - lvb: lower var bound loss as defined in Structured Denoising Diffusion, Austin et. al
    """
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, tokenizer=Tokenizer()):
        self.tmax = tmax
        self.tokenizer = Tokenizer
        self.K = self.tokenizer.K
        super().__init__(reduction=reduction, log_target=log_target)
    
    def forward(self, src_onehot, q, predicitons, tgt, tgt_onehot, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predicitons[:, :, :self.K], dim=2)
        losses = []
        for i in range(tgt.shape[0]): # loop over batch
            if timestep[i] == 1:
                # CE (L_t=0)
                r_loss = CrossEntropyLoss(predicitons[i].unsqueeze(0), tgt[i].unsqueeze(0))
                losses.append(r_loss)
            elif timestep[i] == self.tmax:
                # D KL (L_T)
                prior = sample_prior(q[i].shape[0], q[i].shape[1], len=self.K)
                prior = prior.to(tgt.device)
                kl_loss_i = super().forward(prior.log(), q[i])
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                pred = p[i]
                x_t = src_onehot[i]
                x_0 = tgt_onehot[i]

                # Calculate p_theta_marg, simplified for loops
                A = torch.mm(x_t, torch.t(Q[timestep[i]])) # [P x K]
                B = torch.mm(x_0, Q_bar[timestep[i] - 1]) # P x K
                Q_expand = Q_bar[timestep[i] - 1].unsqueeze(0).expand(A.shape[0], self.K, self.K) # [ P x K x K]
                B_pred = torch.mul(pred.unsqueeze(2), Q_expand)

                q_t = torch.mul(A.unsqueeze(1), B_pred)
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2), pred.unsqueeze(2)).squeeze() # mul and sum over model logits

                num = torch.mul(A, B)
                denom = torch.bmm(torch.mm(x_0, Q_bar[timestep[i]]).unsqueeze(1), x_t.unsqueeze(2))
                q_t_minus1 = num / denom.squeeze().unsqueeze(1)

                p_theta_marg = p_theta_marg / p_theta_marg.sum(dim=1, keepdim=True) # re-normalize probs per residue
                p_theta_marg = p_theta_marg.to(tgt.device)
                kl_loss_i = super().forward(p_theta_marg.log(), q_t_minus1) # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
                

        losses = torch.stack(losses) # loss per sequence in batch
        lvb = ((losses.sum() / tgt.shape[0])) # loss per batch, norm by batchsize
        return lvb


