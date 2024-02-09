import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset


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
    
