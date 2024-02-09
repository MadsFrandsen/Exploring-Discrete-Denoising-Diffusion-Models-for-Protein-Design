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