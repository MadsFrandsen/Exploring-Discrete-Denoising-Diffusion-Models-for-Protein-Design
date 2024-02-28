import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim, 
                           max_time=1000., dtype=torch.float32):
    """Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
        timesteps: tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        max_time: float: largest time input
        dtype: data type of the generated embeddings

    Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1
    timesteps *= (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000, dtype=dtype)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = torch.concatenate([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: #zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nearest_neighbor_upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')


class Normalize(nn.Module):
    def __init__(self, num_groups=32, num_channels=None):
        super().__init__()
        assert num_channels is not None
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x):
        return self.norm(x)


class ResnetBlock(nn.Moduel):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, use_bias=True):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = Normalize(num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding='same', bias=use_bias)
        self.norm2 = Normalize(num_channels=out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding='same', bias=use_bias)
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    

    def forward(x):
        pass



        

        