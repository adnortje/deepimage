# imports
import torch.nn as nn
from layers import ConvGruCell

"""
Convolutional GRU Encoder Network

    Args:
        p_s (int) : input patch size
"""


class ConvRnnEncoder(nn.Module):

    def __init__(self):
        super(ConvRnnEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            nn.ReLU()
        )

        self.cGRU1 = ConvGruCell(
            input_dim=64,
            hidden_dim=256,
            kernel_i=3,
            stride_i=2,
            padding_i=1,
            kernel_h=3,
            stride_h=1,
            padding_h=1,
            bias=True
        )

        self.cGRU2 = ConvGruCell(
            input_dim=256,
            hidden_dim=512,
            kernel_i=3,
            stride_i=2,
            padding_i=1,
            kernel_h=3,
            stride_h=1,
            padding_h=1,
            bias=True
        )

        self.cGRU3 = ConvGruCell(
            input_dim=512,
            hidden_dim=512,
            kernel_i=3,
            stride_i=2,
            padding_i=1,
            kernel_h=3,
            stride_h=1,
            padding_h=1,
            bias=True
        )

    def forward(self, x, h_states=None):
        if h_states is None:
            # init hidden states
            h_states = self._init_h_states(x.size())

        x = self.conv(x)
        h1 = self.cGRU1(x, h_states[0])
        h2 = self.cGRU2(h1, h_states[1])
        h3 = self.cGRU3(h2, h_states[2])
        h_states = [h1, h2, h3]

        return h3, h_states

    def _init_h_states(self, size):
        b_s, _, h, w = size

        h_states = [
            self.cGRU1.init_hidden(b_s, h // 4, w // 4),
            self.cGRU2.init_hidden(b_s, h // 8, w // 8),
            self.cGRU3.init_hidden(b_s, h // 16, w // 16)
        ]
        return h_states
