from typing import Union
import os
import string
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from constants import ALL_AAS, AMB_AAS, CAN_AAS

path = '/Users/madsfrandsen/Documents/BSc_project/protein_generation/alignments/BLAT_ECOLX_1_b0.5.a2m'


def parse_fasta(path):
    sequences = []
    with open(path, 'r') as file:
        sequence = ''
        for line in file:
            line = line.rstrip()
            
            # If line is a new sequence we append the previous sequence,
            # and reset the sequence to the empty string.
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line.replace('.', '-').upper()
        
        # Add last line to the list
        if sequence:
            sequences.append(sequence)
    
    # Convert to Numpy array and return
    return np.array(sequences)


data = parse_fasta(path=path)

train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


class BlAT_ECOLX_1(Dataset):
    def __init__(self, train=True):
        self.data = parse_fasta(path=path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


for seq in data:
    if any(aa in AMB_AAS for aa in seq):
        print('Found AA')


for seq in data:
    for aa in seq:
        if aa == 'Z':
            print(seq)


for seq in data:
    print(seq)

