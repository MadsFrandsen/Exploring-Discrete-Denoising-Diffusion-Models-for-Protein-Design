from typing import Union
import os
import string
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from constants import ALL_AAS, AMB_AAS, CAN_AAS
from utils import Tokenizer

# path to dataset
path = '/Users/madsfrandsen/Documents/BSc_project/protein_generation/alignments/BLAT_ECOLX_1_b0.5.a2m'


def parse_fasta(path):
    sequences = []
    names = []
    with open(path, 'r') as file:
        sequence = ''
        for line in file:
            line = line.rstrip()
            
            # If line is a new sequence we append the previous sequence,
            # and reset the sequence to the empty string.
            if line.startswith('>'):
                names.append(line)
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line.replace('.', '-').upper()
        
        # Add last line to the list
        if sequence:
            sequences.append(sequence)
    
    # Convert to Numpy array and return
    return np.array(sequences), np.array(names)


data, names = parse_fasta(path=path)

train_data, test_data, train_names, test_names = train_test_split(
    data, names, test_size=0.20, random_state=42)


def sample_transition_matrix(x_0, Q_bar):
    """
    Sample a markov transition according to next_step = x_0 * Q^time,
    where Q_bar = Q^t or cumprod of scheduled transition matrices
    returns sample and probabilities
    """
    p_next_step = torch.mm(x_0, Q_bar)
    next_step = torch.multinomial(p_next_step, num_samples=1)
    return next_step.squeeze(), p_next_step # sample and probabilities



class ProteinSequenceDataset(Dataset):
    def __init__(self, train=True):
        self.data = train_data if train else test_data
        self.names = train_names if train else test_names

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        name = self.names[idx]
        return data, name


class Collater(object):
    def __init__(self, tokenizer=Tokenizer(), num_steps=500, Q=None, Q_bar=None):
        self.tokenizer=tokenizer
        self.num_steps = num_steps
        self.K = tokenizer.K
        self.Q = Q
        self.Q_bar = Q_bar
    
    def __call__(self, sequences):
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        one_hot = torch.stack([self.tokenizer.one_hot(seq) for seq in tokenized])

        src = []
        timesteps = []
        q_x = []

        for i, t in enumerate(tokenized):
            x = one_hot[i]
            t = np.random.randint(1, self.num_steps)
            timesteps.append(t)

            x_t, q_x_t = sample_transition_matrix(x, self.Q_bar[t])
            src.append(x_t)
            q_x.append(q_x_t)
        
        src = torch.stack(src).to(torch.long)
        q_x = torch.stack(q_x).to(torch.double)
        timesteps = torch.tensor(timesteps, dtype=torch.long)

        return src, one_hot, timesteps, tokenized.to(torch.long), self.Q, self.Q_bar, q_x



# for seq in data:
#     if any(aa in AMB_AAS for aa in seq):
#         print('Found AA')


# for seq in data:
#     for aa in seq:
#         if aa == 'Z':
#             print(seq)


# for seq in data:
#     print(seq)

