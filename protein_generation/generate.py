import numpy as np
import torch
import tqdm
from utils import Tokenizer


def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, device, batch_size=3):
    """
    Generate a random start string from uniform dist and convert to predictions
    """

    model.eval()
    
    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
    sample = sample.to(torch.long)
    sample = sample.to(device)
    Q = Q.to(device)
    Q_bar = Q_bar.to(device)

    # iterate over reverse timesteps
    timesteps = torch.linspace(timesteps-1, 1, int((timesteps-1)/1), dtype=int)
    timesteps = timesteps.to(device)
    with torch.no_grad():
        for t in tqdm(timesteps):
            timesteps = torch.tensor([t] * batch_size).to(device)
            predictions = model(sample, timesteps)
            p = predictions[:, :, :tokenizer.K]
            p = torch.nn.functional.softmax(p, dim=-1) # softmax over categorical probs
            p = p.to(torch.float64)
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, torch.t[Q[t]]) # [P x K]
                Q_expand = Q_bar[t-1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K) # [P x K x K]
                B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                q_t = torch.mul(A.unsqueeze(1), B_pred) # [P x K x K]
                p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2), p[i].unsqueeze(2)).squeeze()
                p_theta_marg = p_theta_marg / p_theta_marg.sum(dim=1, keepdim=True)
                x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
            sample = x_tminus1

        untokenized = [tokenizer.untokenize(s) for s in sample]
        return sample, untokenized
                