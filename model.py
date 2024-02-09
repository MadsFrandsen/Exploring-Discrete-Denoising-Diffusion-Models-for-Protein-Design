import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        self.same_channels = in_channels = out_channels

        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=3, stride=2, padding=1
            ), # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.GELU() # GELU activation function
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, 
                kernel_size=3, stride=2, padding=1
            ), #  3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels), # Batch normalization
            nn.GELU() # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        # If using residual connection
        if self.is_res:
            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(
                    x.shape[1], x2.shape[1], 
                    kernel_size=1, stride=1, padding=0
                ).to(x.device)
                out = shortcut(x) + x2
            
            # Normalize output tensor by dividing by sqrt(2)
            return out / 1.414
        else:
            return x2
    
    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels
    
    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels: int):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels



class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UnetUp, self).__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=2, stride=2, padding=0
            ),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)

        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UnetDown, self).__init__()

        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

class EmbedFC(nn.Moduel):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()

        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
    


class ContextUnet(nn.Moduel):
    def __init__(self, in_channels: int, n_feat=256, n_cfeat=10, img_size=28, nb_class=3):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = img_size #assume h == w. must be divisible by 4, so 28,24,20,16...
        self.nb_class = nb_class

        self.embedding = nn.Embedding(self.nb_class, in_channels)
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, nb_class, 3, 1, 1)
        )


    def forward(self, x, t, c=None):
        pass
