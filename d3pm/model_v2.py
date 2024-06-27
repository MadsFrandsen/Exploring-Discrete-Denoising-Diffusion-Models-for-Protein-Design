import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
import torch.nn.functional as F

from labml_helpers.module import Module
from einops import rearrange
import utils



def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., dtype=torch.float32):
    """Build sinusoidal embeddings (from Fairseq)."""
    assert len(timesteps.shape) == 1
    timesteps = (timesteps * (1000. / max_time))

    half_dim = embedding_dim // 2
    emb = math.log(10_000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb)
    emb = timesteps.type(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def nearest_neighbor_upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class Normalize(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.norm1 = Normalize(32, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = Normalize(32, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        B, C, H, W = x.shape
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection using a linear layer.
        if isinstance(self.shortcut, nn.Linear):
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.shortcut(x)
            x = x.permute(0, 3, 1, 2).contiguous()

        return h + x
    
class AttnBlock(nn.Module):
    def __init__(self, num_heads, num_channels):
        super().__init__()
        self.num_heads = num_heads
        self.norm = Normalize(32, num_channels)
        self.qkv_proj = nn.Linear(num_channels, num_channels * 3)
        self.out_proj = nn.Linear(num_channels, num_channels)

    def forward(self, x, t=None):
        _ = t

        B, C, H, W = x.shape
        x_norm = self.norm(x)
        x_norm = x_norm.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        qkv = self.qkv_proj(x_norm).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b (h d) c -> b h (d c)', h=self.num_heads) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        h = self.out_proj(h)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        return x + h



class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (3, 3), (1, 1), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        x = nearest_neighbor_upsample(x)
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class DownBlock(Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_heads: int, has_attn: bool):
        super().__init__()
        self.res = ResnetBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttnBlock(n_heads, out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_heads: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResnetBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttnBlock(n_heads, out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, n_heads: int):
        super().__init__()
        self.res1 = ResnetBlock(n_channels, n_channels, time_channels)
        self.attn = AttnBlock(n_heads, n_channels)
        self.res2 = ResnetBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UNet_v2(Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 128,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, True, False, False),
                 n_blocks: int = 2, model_output: str = 'logistic_pars', 
                 num_pixel_vals: int = 256, max_time: int = 1000,
                 num_heads: int = 1):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        * `model_output` is the type of output of the model
        """
        super().__init__()

        # Output type
        self.model_output = model_output
        
        # Number of pixel values, for instance [0, ..., 255] for MNIST
        self.num_pixel_vals = num_pixel_vals

        # Image channels
        self.image_channels = image_channels

        # Number of resolutions
        n_resolutions = len(ch_mults)

        self.n_channels = n_channels

        # time-steps
        self.max_time = max_time

        self.conv_in = nn.Conv2d(image_channels, n_channels, (3, 3), (1, 1), (1, 1))


        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, num_heads, is_attn[i]))
                # down.append(ResnetBlock(in_channels, out_channels, n_channels * 4))
                # if is_attn[i]:
                #     down.append(AttnBlock(num_heads, out_channels))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        
        self.down = nn.ModuleList(down)
        
        self.mid = nn.ModuleList([
            ResnetBlock(out_channels, out_channels, n_channels * 4),
            AttnBlock(num_heads, out_channels),
            ResnetBlock(out_channels, out_channels, n_channels * 4)
        ])

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, num_heads)

        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, num_heads, is_attn[i]))
                # up.append(ResnetBlock(in_channels + out_channels, out_channels, n_channels * 4))
                # if is_attn[i]:
                #     up.append(AttnBlock(num_heads, out_channels))
                in_channels = out_channels
            if i > 0:
                out_channels = n_channels * ch_mults[i-1]
            else:
                out_channels = n_channels * ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, num_heads, is_attn[i]))
            in_channels = out_channels

            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))
        
        # Combine the set of modules
        self.up = nn.ModuleList(up)

        
        self.norm_out = Normalize(32, in_channels)
        self.final = nn.Conv2d(in_channels, 
            image_channels * 2 if model_output == 'logistic_pars' else image_channels * self.num_pixel_vals,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.time1 = nn.Linear(self.n_channels, self.n_channels * 4)
        self.time2 = nn.Linear(self.n_channels * 4, self.n_channels * 4)
        self.act = Swish()


    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        B, C, H, W = x.shape

        x_onehot = F.one_hot(x, num_classes=self.num_pixel_vals)
        # Convert to float and scale image to [-1, 1]
        x = utils.normalize_data(x.float())

        x_start = x

        # Permute (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        temb = get_timestep_embedding(t, self.n_channels, max_time=self.max_time)
        temb = self.time1(temb)
        temb = self.act(temb)
        temb = self.time2(temb)

        
        # Get image projection
        x = self.conv_in(x)
        
        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, temb)
            h.append(x)
        
        # Middle (bottom)
        x = self.middle(x, temb)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, temb)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, temb)
        
        # End
        x = self.final(self.act(self.norm_out(x)))

        # reshape back to (B, H, W, C) to fit framework.
        x = x.permute(0, 2, 3, 1)

        if self.model_output == 'logistic_pars':
            loc, log_scale = torch.chunk(x, 2, dim=-1)
            # ensure loc is between [-1, 1], just like normalized data.
            loc = torch.tanh(loc + x_start)

            return loc, log_scale
        
        elif self.model_output == 'logits':
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.image_channels, self.num_pixel_vals)
            return x_onehot + x

        else:
            raise ValueError(
                f'self.model_output = {self.model_output} but must be '
                'logits or logistic_pars')


