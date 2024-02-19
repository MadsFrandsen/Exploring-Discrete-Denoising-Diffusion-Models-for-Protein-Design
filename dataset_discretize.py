import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from utils import *


class DiscretizeMNIST(Dataset):
    def __init__(self, num_bins=2, train=True, root='./data'):
        self.num_bins = num_bins

        self.mnist = torchvision.datasets.MNIST(
            root=root, train=train, download=True
        )

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )
    
    def discretize(self, data):
        # get min and max value of the data
        min_val = torch.min(data)
        max_val = torch.max(data)

        # get the bin size and discretize the data
        bin_size = (max_val - min_val) / self.num_bins
        data = torch.floor((data - min_val) / bin_size)
        data = torch.clamp(data, 0, self.num_bins - 1)
        
        return data

    def __getitem__(self, index):
        data, label = self.mnist[index]
        data = self.transforms(data)
        data = self.discretize(data)
        return data, label


    def __len__(self):
        return len(self.mnist)
    

class DiscretizeD3PMNIST(Dataset):
    def __init__(self, num_bins=2, time_steps=1000):
        self.num_bins = num_bins
        self.time_steps = time_steps
        self.dataset = DiscretizeMNIST(num_bins=num_bins)

        beta_t = generate_betas(time_steps)

        self.transition_matrices = compute_transition_matrix(
            beta_t, time_steps, num_bins
        )

        self.cumulated_transition = compute_acc_transition_matrices(
            transition_matrices=self.transition_matrices
        )
    
    def __len__(self):
        return len(self.dataset) * self.time_steps
    
    def __getitem__(self, idx):
        idx_data = idx // self.time_steps
        idx_step = idx % self.time_steps

        tmp_transition = torch.tensor(self.cumulated_transition[idx_step, :, :])
        tmp_transition_next = torch.tensor(
            self.transition_matrices[min(idx_step + 1, self.time_steps - 1), :, :]
        )

        data, label = self.dataset[idx_data]

        data = data.squeeze()
        data_flatten = data.view(-1).long()

        size = data.shape
        
        data_bernoulli_proba = tmp_transition[data_flatten]

        data_sample_t = torch.distributions.categorical.Categorical(
            data_bernoulli_proba
        ).sample()

        data_bernoulli_proba_next = tmp_transition_next[data_sample_t]

        data_sample_t_next = torch.distributions.categorical.Categorical(
            data_bernoulli_proba_next
        ).sample()

        data_sample_t = data_sample_t.view(size[0], size[0])
        data_sample_t_next = data_sample_t_next.view(size[0], size[0])

        return data_sample_t, data_sample_t_next, label, data, (idx_step+1) / (self.time_steps-1)
    