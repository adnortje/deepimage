# imports
import torch.nn as nn


"""
Feed Forward Encoder Network

    Args:
        p_s (int) : input image patch size
"""


class FForwardEncoder(nn.Module):

    def __init__(self, p_s):
        super(FForwardEncoder, self).__init__()

        self.h, self.w = p_s
        self.input_dim = 3 * self.h * self.w

        self.enc = nn.Sequential(
            nn.Linear(
                in_features=self.input_dim,
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
            nn.Tanh()
        )

    def forward(self, x):
        # Flattened image --> b_neck
        x = x.view(-1, self.input_dim)
        x = self.enc(x)
        return x