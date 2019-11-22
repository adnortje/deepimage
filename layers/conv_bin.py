# imports
import torch.nn as nn
from functions import binz


"""
Convolutional Binarization Network

    Args:
        bnd    (int) : bottle-neck depth
        in_dim (int) : input channel dimension

"""


class ConvBinarizer(nn.Module):

    def __init__(self, in_dim, bnd):
        super(ConvBinarizer, self).__init__()

        self.bnd = bnd

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=self.bnd,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = binz.apply(x, self.training)
        return x
