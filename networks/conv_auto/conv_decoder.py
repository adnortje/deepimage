# imports
import torch.nn as nn

"""
Convolutional Decoder Network

    Args:
        bnd (int) : bottleneck depth
"""


class ConvDecoder(nn.Module):

    def __init__(self, bnd):
        super(ConvDecoder, self).__init__()

        self.bnd = bnd

        self.dec = nn.Sequential(
            nn.Conv2d(
                in_channels=bnd,
                out_channels=512,
                kernel_size=1,
                stride=1,
                bias=True
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=2,
                stride=2,
                bias=True
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=2,
                stride=2,
                bias=True
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=64,
                kernel_size=2,
                stride=2,
                bias=True
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=2,
                stride=2,
                bias=True
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.dec(x)
        return x