import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import ContextUnet
from torch.utils.data import DataLoader
from D3PM import D3PM

def generate_betas(T, beta_0=1e-5, beta_T=0.999, linear: bool = True):

    if linear:
        # linear schedule
        betas = torch.linspace(beta_0, beta_T, T)
    else:
        # consine schedule
        betas = torch.linspace(torch.pi / 2, 0, T)
        betas = torch.exp(betas) * (beta_T - beta_0) + beta_0

    return betas



def compute_transition_matrix(betas, T, num_bins=4):
    transition_matrices = torch.zeros((T, num_bins, num_bins))

    for t in range(T):
        beta = betas[t]
        transition_matrix = torch.zeros((num_bins, num_bins))

        for i in range(num_bins):
            for j in range(num_bins):
                if i == j:
                    transition_matrix[i, j] = 1 - (num_bins - 1)/num_bins * beta
                else:
                    transition_matrix[i, j] = 1/num_bins * beta
        
        transition_matrices[t] = transition_matrix

    return transition_matrices

def compute_acc_transition_matrices(T, transition_matrices):
    accumulated_transition_matrices = torch.zeros(
        (T, transition_matrices.shape[1], transition_matrices.shape[2])
    )

    accumulated_transition_matrices[0] = transition_matrices[0]

    for t in range(1, T):
        accumulated_transition_matrices[t] = torch.matmul(
            accumulated_transition_matrices[t - 1], transition_matrices[t]
        )
    
    return accumulated_transition_matrices


def loss(logits, data, init_data, t):
    
    # logits = model(data.unsqueeze(1), t.unsqueeze(1))
    
    loss_vb = nn.CrossEntropyLoss(logits, data)
    loss_init = 0.0 * nn.CrossEntropyLoss(logits, init_data)

    loss = loss_vb + loss_init
    
    return loss, loss_vb, loss_init


def train(model: D3PM, train_loader: DataLoader, epochs, optimizer):
    model.train()

    for e in range(1, epochs+1):
        train_loss = 0
        train_loss_vals = []
        for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = x.to(model.device)

            optimizer.zero_grad()
            logits, _ = model(x)
            l, l_vb, l_init = loss(logits, x, )
    pass

