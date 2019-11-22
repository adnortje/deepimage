# imports
import torch.nn as nn
from layers import ConvGruCell


"""
Recurrent Convolutional Decoder Network

    Args:
        p_s (int) : patch size
        bnd (int) : bottle-neck depth
"""


class ConvRnnDecoder(nn.Module):

    def __init__(self, bnd):
        super(ConvRnnDecoder, self).__init__()

        # bottleneck depth
        self.bnd = bnd

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.bnd,
                out_channels=512,
                kernel_size=1,
                stride=1,
                bias=True
            ),
            nn.ReLU()
        )

        self.cGRU1 = ConvGruCell(
            input_dim=512,
            hidden_dim=512,
            kernel_i=3,
            stride_i=1,
            padding_i=1,
            kernel_h=1,
            stride_h=1,
            padding_h=0,
            bias=True
        )

        self.cGRU2 = ConvGruCell(
            input_dim=128,
            hidden_dim=512,
            kernel_i=3,
            stride_i=1,
            padding_i=1,
            kernel_h=1,
            stride_h=1,
            padding_h=0,
            bias=True
        )

        self.cGRU3 = ConvGruCell(
            input_dim=128,
            hidden_dim=256,
            kernel_i=3,
            stride_i=1,
            padding_i=1,
            kernel_h=3,
            stride_h=1,
            padding_h=1,
            bias=True
        )

        self.cGRU4 = ConvGruCell(
            input_dim=64,
            hidden_dim=128,
            kernel_i=3,
            stride_i=1,
            padding_i=1,
            kernel_h=3,
            stride_h=1,
            padding_h=1,
            bias=True
        )

        self.convRGB = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.Tanh()
        )

        # Depth-to-Space unit
        self.depth2space = nn.PixelShuffle(
            upscale_factor=2
        )

    def forward(self, x, h_states=None):
        if h_states is None:
            # init states
            h_states = self._init_hidden_states(x.size())

        x = self.conv(x)

        h1 = self.cGRU1(x, h_states[0])
        x = self.depth2space(h1)

        h2 = self.cGRU2(x, h_states[1])
        x = self.depth2space(h2)

        h3 = self.cGRU3(x, h_states[2])
        x = self.depth2space(h3)

        h4 = self.cGRU4(x, h_states[3])
        x = self.depth2space(h4)

        x = self.convRGB(x)
        h_states = [h1, h2, h3, h4]

        return x, h_states

    def _init_hidden_states(self, size):
        b_s, _, h, w = size

        h_states = [
            self.cGRU1.init_hidden(b_s, h, w),
            self.cGRU2.init_hidden(b_s, h * 2, w * 2),
            self.cGRU3.init_hidden(b_s, h * 4, w * 4),
            self.cGRU4.init_hidden(b_s, h * 8, w * 8)
        ]
        return h_states
