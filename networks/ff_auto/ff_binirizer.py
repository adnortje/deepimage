# imports
import torch.nn as nn
from functions import binz
"""
Feed Forward Binarization Network

    Args:
        b_n (int) : bits in bottleneck layer
"""


class FForwardBinarizer(nn.Module):

    def __init__(self, b_n):
        super(FForwardBinarizer, self).__init__()

        self.b_n = b_n

        self.ff = nn.Sequential(
            nn.Linear(
                in_features=512,
                out_features=self.b_n,
                bias=True
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.ff(x)
        x = binz.apply(x, self.training)
        return x