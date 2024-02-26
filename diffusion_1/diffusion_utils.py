import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from model import ContextUnet

def loss(model: ContextUnet, logits, data, init_data, t):
    
    # logits = model(data.unsqueeze(1), t.unsqueeze(1))
    
    loss_vb = nn.CrossEntropyLoss(logits, data)
    loss_init = 0.0 * nn.CrossEntropyLoss(logits, init_data)

    loss = loss_vb + loss_init
    
    return loss, (loss_vb, loss_init)