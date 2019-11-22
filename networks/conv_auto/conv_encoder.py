# import
import torch.nn as nn

"""
Convolutional Encoder Network

"""


class ConvEncoder(nn.Module):

    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.enc(x)
        return x