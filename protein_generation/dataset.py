from typing import Union
from collections import defaultdict
import os
import string
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from constants import ALL_AAS, AMB_AAS, CAN_AAS
from utils import Tokenizer, sample_transition_matrix

# path to dataset
# path = '/Users/madsfrandsen/Documents/BSc_project/protein_generation/alignments/BLAT_ECOLX_1_b0.5.a2m'
# output_path = '/Users/madsfrandsen/Documents/BSc_project/protein_generation/BLAT_ECOLX_1_b0.5_labeled_allcaps_alldash.fasta'

# path used to train model on colab
path = '/content/drive/MyDrive/BLAT_ECOLX_1_b0.5.a2m'
output_path = 'BLAT_ECOLX_1_b0.5_labeled_allcaps_alldash.fasta'


def parse_fasta(path, output_path=None, preprocess=False):

    if preprocess:
        seq_name_to_sequence = defaultdict(str)
        seq_names = []

        name = ""
        INPUT = open(path, 'r')
        for i, line in enumerate(INPUT):
            line = line.rstrip()
            if line.startswith('>'):
                name = line
                seq_names.append(name)
            else:
                seq_name_to_sequence[name] += line
        INPUT.close()

        focus_seq_name = seq_names[0]

        focus_seq = seq_name_to_sequence[focus_seq_name]
        focus_cols = [ix for ix, s in enumerate(focus_seq) if s == s.upper()]
        focus_seq_trimmed = [focus_seq[ix] for ix in focus_cols]
        seq_len = len(focus_cols)
        alphabet_size = len(ALL_AAS)

        focus_loc = focus_seq_name.split('/')[-1]
        start, stop = focus_loc.split('-')

        for seq_name, sequence in seq_name_to_sequence.items():
            sequence = sequence.replace('.', '-')
            seq_name_to_sequence[seq_name] = ''.join([sequence[ix].upper() for ix in focus_cols])
        
        alphabet_set = set(list(ALL_AAS))
        seq_names_to_remove = []
        for seq_name, sequence in seq_name_to_sequence.items():
            for letter in sequence:
                if letter not in alphabet_set and letter != '-':
                    seq_names_to_remove.append(seq_name)
                    break
        
        for seq_name in seq_names_to_remove:
            del seq_name_to_sequence[seq_name]
        
        sequences = []
        names = []
        for seq_name, sequence in seq_name_to_sequence.items():
            sequences.append(sequence)
            names.append(seq_name)
        
    
        sequences = np.array(sequences)
        names = np.array(names)

        if output_path:
            with open(output_path, 'w') as output_file:
                for name, sequence in zip(names, sequences):
                    output_file.write(f'{name}\n')
                    output_file.write(f'{sequence}\n')

        return sequences, names, focus_seq, focus_seq_trimmed
    
    else:
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
                    sequence += line.upper().replace('.', '-').replace('Z', 'X')
            
            # Add last line to the list
            if sequence:
                sequences.append(sequence)
        
        # Convert to Numpy array
        sequences = np.array(sequences)
        names = np.array(names)

        if output_path:
            with open(output_path, 'w') as output_file:
                for name, sequence in zip(names, sequences):
                    output_file.write(f'{name}\n')
                    output_file.write(f'{sequence}\n')

        return sequences, names, None, None


data, names, focus_seq, focus_seq_trimmed = parse_fasta(path=path, output_path=output_path)

train_data, test_data, train_names, test_names = train_test_split(
    data, names, test_size=0.20, random_state=42)


class ProteinSequenceDataset(Dataset):
    def __init__(self, full_data=True, train=True):
        if full_data:
            self.data = data
            self.names = names
        else:
            self.data = train_data if train else test_data
            self.names = train_names if train else test_names

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        name = self.names[idx]

        print(len(data))
        return data, name



class ProteinSequenceDatasetRandom(Dataset):
    def __init__(self, full_data=True, train=True, preprocess_data=False):
        if full_data:
            self.data = data
            self.names = names
        else:
            self.data = train_data if train else test_data
            self.names = train_names if train else test_names
        self.reference_seq = ''.join(focus_seq) if preprocess_data else data[0]
        self.reference_name = names[0]
        self.reference_seq_trimmed = ''.join(focus_seq_trimmed) if preprocess_data else data[0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        random_indices = random.sample(range(len(self.data)), 1)
        sampled_sequences = self.data[random_indices]
        sampled_names = self.names[random_indices]

        return sampled_sequences[0], sampled_names[0]


class ProteinSequenceDatasetMSA(Dataset):
    def __init__(self, full_data=True, train=True, num_samples=64):
        if full_data:
            self.data = data
            self.names = names
        else:
            self.data = train_data if train else test_data
            self.names = train_names if train else test_names
        self.reference_seq = data[0]
        self.reference_name = names[0]
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        random_indices = random.sample(range(1, len(self.data)), self.num_samples - 1)
        sampled_sequences = [self.data[i] for i in random_indices]
        sampled_names = [self.names[i] for i in random_indices]

        sampled_sequences.insert(0, self.reference_seq)
        sampled_names.insert(0, self.reference_name)

        return sampled_sequences


class Collater(object):
    """
    Collater for generating batch data according to markov process according to Austin et al.
    inputs:
        sequences : list of sequences
        tokenizer: Tokenizer()
        num_timesteps: number of diffusion timesteps

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        Q : markov matrix
        Q_bar : cumulative prod of markov matrix
        q_x : forward transition probabilities
    """
    def __init__(self, tokenizer=Tokenizer(), num_steps=500, Q=None, Q_bar=None):
        self.tokenizer=tokenizer
        self.num_steps = num_steps
        self.K = tokenizer.K
        self.Q = Q
        self.Q_bar = Q_bar
    
    def __call__(self, data):

        sequences = [x[0] for x in data]
        names = [x[1] for x in data]

        assert len(sequences) == len(names)

        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        one_hot = torch.stack([self.tokenizer.one_hot(seq) for seq in tokenized])

        src = []
        timesteps = []
        q_x = []
        src_one_hot = []

        for i, t in enumerate(tokenized):
            x = one_hot[i]
            t = np.random.randint(1, self.num_steps)
            timesteps.append(t)

            x_t, q_x_t = sample_transition_matrix(x, self.Q_bar[t])
            src.append(x_t)
            q_x.append(q_x_t)
            src_one_hot.append(self.tokenizer.one_hot(x_t))
        
        src = torch.stack(src).to(torch.long)
        src_one_hot = torch.stack(src_one_hot).to(torch.double)
        q_x = torch.stack(q_x).to(torch.double)
        timesteps = torch.tensor(timesteps, dtype=torch.long)
        tokenized = torch.stack(tokenized).to(torch.long)

        return src, src_one_hot, timesteps, tokenized, one_hot, self.Q, self.Q_bar, q_x, names


class CollaterMSA(object):
    def __init__(self, tokenizer=Tokenizer(), num_steps=100, Q=None, Q_bar=None, num_seqs=64):
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.K = self.tokenizer.K
        self.Q = Q
        self.Q_bar = Q_bar
        self.num_seqs = num_seqs

    def __call__(self, msas):

        tokenized = []
        timesteps = []
        src = []
        src_one_hot = []
        tgt_one_hot = []
        q_x = []

        for msa in msas:
            curr_tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in msa]
            curr_tgt_one_hot = [self.tokenizer.one_hot(s) for s in curr_tokenized]
            curr_msa = torch.stack(curr_tgt_one_hot)
            length, depth, tokens = curr_msa.shape
            curr_msa = curr_msa.flatten(start_dim=0, end_dim=1)

            t = np.random.randint(1, self.num_steps)
            timesteps.append(t)

            x_t, q_x_t = sample_transition_matrix(curr_msa, self.Q_bar[t])
            x_t = x_t.reshape(length, depth)
            q_x_t = q_x_t.reshape(length, depth, tokens)
            src.append(x_t)
            curr_src_one_hot = [self.tokenizer.one_hot(t) for t in x_t]
            
            q_x.append(q_x_t)
            tokenized.append(torch.stack(curr_tokenized))
            src_one_hot.append(torch.stack(curr_src_one_hot))
            tgt_one_hot.append(torch.stack(curr_tgt_one_hot))
            q_x.append(q_x_t)
        
        src = torch.stack(src).to(torch.long)
        src_one_hot = torch.stack(src_one_hot).to(torch.double)
        timesteps = torch.tensor(timesteps)
        tokenized = torch.stack(tokenized).to(torch.long)
        tgt_one_hot = torch.stack(tgt_one_hot).to(torch.double)
        q_x = torch.stack(q_x).to(torch.double)

        return src, src_one_hot, timesteps, tokenized, tgt_one_hot, self.Q, self.Q_bar, q_x, 