import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from utils import Tokenizer
from constants import MSA_AAS

def sample_prior(a, b, len=len(MSA_AAS)):
    """
    Returns prior for KL at T -> inf with same shape as q over total possible values
    Prior is a uniform distribution over number of values
    """
    prior = torch.empty(a, b)
    prior = torch.ones_like(prior) / len
    return prior

def sample_priorMSA(a, b, c, len=(len(MSA_AAS))):
    """
    Returns prior for KL at T-> inf with same shape as q over total possible values (all_aas)
    Prior is a stationary distribution; uniform distribution over number of values
    """
    prior = torch.empty(a, b, c)
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
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, tokenizer=Tokenizer(), evaluation=None):
        self.tmax = tmax
        self.tokenizer = tokenizer
        self.K = self.tokenizer.K
        super().__init__(reduction=reduction, log_target=log_target)
        self.reconstruction_loss = CrossEntropyLoss(weight=None, reduction='mean')
        self.evaluation = evaluation
    
    def forward(self, src_onehot, q, predicitons, tgt, tgt_onehot, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predicitons[:, :, :self.K], dim=2)
        losses = []
        for i in range(tgt.shape[0]): # loop over batch
            if timestep[i] == 1:
                # CE (L_t=0)
                r_loss = self.reconstruction_loss(predicitons[i], tgt[i])
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
                pred = pred.to(torch.float64)
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
        if self.evaluation:
            return losses
        
        lvb = ((losses.sum() / tgt.shape[0])) # loss per batch, norm by batchsize
        return lvb
    

class D3PMLVBLossMSA(KLDivLoss):
    """
        Shape:
            Inputs:
                - src_one_hot: (B, D, L, K) original MSA one hot encoded
                - q: (B, D, L, K) forward prob dist
                - predictions: (B, D, L, K) model predictions
                - tgt: (B, D, L) corrupted MSA tokenized
                - tgt_one_hot: (B, D, L, K) corrupted MSA one hot encoded
                - input_mask: (B, D, L) bool mask indicating pad locations
                - timestep (B)
                - Q (K, K) transition matrix
                - Q_bar (K, K) transition matrix accounting for time

            Returns
                - lower var bound loss as defined in Structured Denoising Diffusion, Austin et. al
    """
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, tokenizer=Tokenizer()):
        self.tmax = tmax
        self.tokenizer = tokenizer
        self.K = tokenizer.K
        super().__init__(reduction=reduction, log_target=log_target)
        self.reconstruction_loss = CrossEntropyLoss(weight=None, reduction='mean')
    
    def forward(self, src_one_hot, q, predictions, tgt, tgt_one_hot, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predictions[:, :, :, :self.K], dim=3)
        losses = []
        for i in range(len(tgt)):
            if timestep[i] == 1:
                r_loss = self.reconstruction_loss(predictions[i].reshape(-1, predictions[i].shape[-1]), tgt[i].reshape(-1))
                losses.append(r_loss)
            elif timestep[i] == self.tmax:
                prior = sample_priorMSA(q[i].shape[0], q[i].shape[1], q[i].shape[2], len=self.tokenizer.K)
                prior = prior.to(tgt.device)
                kl_loss_i = super().forward(prior.log(), q[i])
                losses.append(kl_loss_i)
            else:
                pred = p[i, :, :, :self.K].flatten(start_dim=0, end_dim=1) # [pos x tokens]
                pred = pred.to(torch.float64) 
                x_t = src_one_hot[i, :, :, :self.K].flatten(start_dim=0, end_dim=1)
                x_0 = tgt_one_hot[i, :, :, :self.K].flatten(start_dim=0, end_dim=1)

                A = torch.mm(x_t, torch.t(Q[timestep[i]])) # [P x K]
                B = torch.mm(x_0, Q_bar[timestep[i] - 1]) # P x K
                Q_expand = Q_bar[timestep[i] - 1].unsqueeze(0).expand(A.shape[0], self.K, self.K) # [P x K x K]
                B_pred = torch.mul(pred.unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred) # batch mul (broadcast logit dim)
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2), pred.unsqueeze(2)).squeeze() # this marginalizes over logits
                p_theta_marg = p_theta_marg / p_theta_marg.sum(dim=1, keepdim=True)
                p_theta_marg = p_theta_marg.to(tgt.device)

                num = torch.mul(A, B)
                denom = torch.bmm(torch.mm(x_0, Q_bar[timestep[i] - 1]).unsqueeze(1), x_t.unsqueeze(2))
                q_t_minus1 = num / denom.squeeze().unsqueeze(1)

                kl_loss_i = super().forward(p_theta_marg.log(), q_t_minus1)
                losses.append(kl_loss_i)
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (tgt.shape[0]))
        return lvb



def KL_test(q, tokenizer):
    prior = sample_priorMSA(q.shape[0], q.shape[1], q.shape[2], len=tokenizer.K)
    kl_loss = KLDivLoss(reduction='batchmean', log_target=False)
    return kl_loss(prior.log(), q)

