# imports
import torch.nn as nn


"""
Feed Forward Decoder Network

    Args:
        p_s (int) : input image patch size
        b_n (int) : bits in bottleneck layer
"""


class FForwardDecoder(nn.Module):

    def __init__(self, p_s, b_n):
        super(FForwardDecoder, self).__init__()

        self.h, self.w = p_s
        self.opt_dim = 3 * self.h * self.w
        self.b_n = b_n

        self.dec = nn.Sequential(
            nn.Linear(
                in_features=self.b_n,
                out_features=512
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=512,
                out_features=self.opt_dim
            ),
            nn.Tanh()
        )

    def forward(self, x):
        # b_neck --> image
        x = self.dec(x)
        x = x.view(-1, 3, self.h, self.w)
        return x
