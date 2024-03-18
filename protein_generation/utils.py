import torch
import numpy as np
from constants import PROTEIN_ALPHABET, GAP


def cumprod_matrix(a):
    """
    Takes a list of transition matrices and outputs a list
    of the cumulative products (Q_bar) at each timestep
    """
    a_bar = [a[0]]
    start = a[0]
    for i in range(len(a) - 1):
        a_prod_temp = torch.mm(start, a[i + 1])
        start = a_prod_temp
        a_bar.append(a_prod_temp)
    return a_bar


def softmax(x):
    """
    Compute softmax over x
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _beta_schedule(num_steps, start=1e-5, end=0.999, schedule='linear'):
    """Function to generate betas."""
    if schedule == 'linear':
        return torch.linspace(start, end, num_steps)
    elif schedule == 'cosine':
        betas = torch.linspace(np.pi / 2, 0, num_steps)
        betas = torch.cos(betas) * (end - start) + start
        return betas
    elif schedule == 'sohl-dickstein':
        betas = torch.linspace(0, num_steps-1, num_steps)
        betas = 1/(num_steps - betas + 1)
    else:
        raise NotImplementedError(type)


class Tokenizer(object):
    def __init__(self, protein_alphabet=PROTEIN_ALPHABET, gap=GAP):
        self.alphabet = list("".join(protein_alphabet))
        self.gap = gap
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)
        self.K = len(self.alphabet)
    

    def q_uniform_schedule(self, num_steps=500, schedule='sohl-dickstein'):
        """
        Takes a number of time steps and a schedule and
        generates Q_t and Q_bar_t matrices.
        """
        betas = _beta_schedule(num_steps=num_steps, schedule=schedule)
        Q_t = []
        for i in range(len(betas)):
            # each iteration of the loop computes matrix Q_t at time t=i
            # using the equations from the paper for uniform transition matrix
            q_non_diag = torch.ones((self.K, self.K)) / self.K * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(self.K)) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod)
        Q_t = torch.stack(Q_t)
        return Q_prod, Q_t
    
    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq])
    
    def untokenize(self, x):
        if torch.is_tensor(x):
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])
    
    def one_hot(self, tokenized):
        "seq -> one hot"
        x_onehot = torch.nn.functional.one_hot(tokenized, num_classes=self.K)
        return x_onehot.to(torch.double)
    
    def undo_one_hot(self, x_onehot):
        "one hot -> seq"
        tokenized = [np.where(r==1)[0] for r in x_onehot]
        return tokenized
