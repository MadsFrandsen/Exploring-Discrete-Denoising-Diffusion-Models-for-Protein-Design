from typing import Union
import os
import string
import json
import numpy as np
import torch
from torch.utils.data import Dataset

path = '/Users/madsfrandsen/Documents/BSc_project/protein_generation/alignments/BLAT_ECOLX_1_b0.5.a2m'


def parse_fasta(path):
    sequences = []
    with open(path, 'r') as file:
        sequence = ''
        for line in file:
            line = line.strip()
            
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



class BlAT_ECOLX_1(Dataset):
    def __init__(self, path):
        self.data = parse_fasta(path=path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


