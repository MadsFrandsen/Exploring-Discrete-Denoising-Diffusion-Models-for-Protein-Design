import torch
import torch.nn as nn
from model import ContextUnet

from utils import *



class D3PM(nn.Moduel):
    def __init__(self, device, model: ContextUnet, T, num_bins):
        super(D3PM, self).__init__()

        self.device = device
        self.model = model
        self.time_steps = T
        self.num_bins = num_bins

        self.beta_t = generate_betas(T)
        
        self.transition_matrices = compute_transition_matrix(
            self.beta_t, T, num_bins
        )

        self.cumulated_transition = compute_acc_transition_matrices(
            self.transition_matrices
        )
    
    def forward(self, data, t):
        data = data.unsqueeze(1)
        t = t.unsqueeze(1)

        logits = self.model(data, t)

        return logits
    

    


    





