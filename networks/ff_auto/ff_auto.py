# imports
import torch
import torch.nn as nn
from numbers import Number
import networks.functional as f
from .ff_encoder import FForwardEncoder
from .ff_decoder import FForwardDecoder
from .ff_binirizer import FForwardBinarizer

"""
Class FF_Autoencoder:
    
    Method:
        create(self, itrs, ptch, bn):
            used to create a Feed-Forward Residual Autoencoder.
            Args:
                itrs (int): number of autoencoder iterations
                p_s  (int): input patch size
                b_n  (int):  bits in bottleneck layer
                
            Return:
                Feed-Forward Residual Autoencoder (nn.Module)
    
"""


class FForwardAutoencoder(nn.Module):
    
    def __init__(self, itrs, p_s, b_n):
        super(FForwardAutoencoder, self).__init__()
        
        # def model name
        self.display_name = "FeedForwardAR"
        self.name = "FForward"
        
        # def p_s
        if isinstance(p_s, Number):
            self.p_s = (int(p_s), int(p_s))
        else:
            self.p_s = p_s
        self.p_h, self.p_w = self.p_s
        
        # def b_n
        self.b_n = int(b_n)
        
        # Cell of Feed Forward AR system
        class _AutoencoderCell(nn.Module):
            
            def __init__(self, patch_size, bits):
                super(_AutoencoderCell, self).__init__()
                
                self.p_s = patch_size
                self.b_n = bits
                
                # def autoencoder cell components
                self.enc = FForwardEncoder(
                    p_s=self.p_s
                )
                self.bin = FForwardBinarizer(
                    b_n=self.b_n
                )
                self.dec = FForwardDecoder(
                    p_s=self.p_s,
                    b_n=self.b_n
                )
            
            def forward(self, x):
                # forward
                x = self.dec(self.bin(self.enc(x)))
                return x
        
        # ModuleList of Autoencoder Cells so weights aren't shared
        self.itrs = itrs
        self.ae_sys = nn.ModuleList(
            [
                _AutoencoderCell(self.p_s, self.b_n) for _ in range(self.itrs)
            ]
        )
        
    def forward(self, r):
        
        losses = []
        
        for i in range(self.itrs):
            
            dec = self.ae_sys[i](r)
            
            # calculate L1 loss
            r = r - dec
            losses.append(r.abs().mean())

            # detach r after after each iteration
            r = r.detach()
        
        # sum  & normalize residual loss
        loss = sum(losses) / self.itrs

        return loss
            
    def encode_decode(self, r, itrs):
        
        assert(self.itrs >= itrs), "itrs > Training Iterations"

        # run in inference mode
        with torch.no_grad():

            # extract original image dimensions
            img_size = r.size()[2:]

            # convert images to patches
            r = f.sliding_window(
                input=r,
                kernel_size=self.p_s,
                stride=self.p_s
            )

            # init output
            output = 0.0

            for i in range(itrs):
                # encode & decode patches
                dec = self.ae_sys[i](r)
                r = r - dec
                # sum residual predictions
                output += dec

            # clamp patches [-1, 1]
            output = output.clamp(-1, 1)

            # reshape patches to images
            images = f.refactor_windows(
                windows=output,
                output_size=img_size
            )

        return images
