# imports
import torch
import warnings
import torch.nn as nn
from numbers import Number
import networks.functional as f
from layers import ConvBinarizer
from .conv_rnn_encoder import ConvRnnEncoder
from .conv_rnn_decoder import ConvRnnDecoder

"""
Implementation of Google's Recurrent Convolutional Neural Network Autoencoder
    
            
        Args:
            itrs (int): number of autoencoder  iterations 
            p_s  (int): input patch size
            b_n  (int): bits in bottle-neck layer
            
        Ref:
            https://arxiv.org/abs/1608.05148
"""


class ConvRnnAutoencoder(nn.Module):
    
    def __init__(self, itrs, p_s, b_n):
        super(ConvRnnAutoencoder, self).__init__()
        
        # model name
        self.name = "ConvRNN"
        self.display_name = "ConvGRU-OSR"
        
        # def p_s
        if isinstance(p_s, Number):
            self.p_s = (int(p_s), int(p_s))
        else:
            self.p_s = p_s
        self.p_h, self.p_w = self.p_s
        assert(self.p_w % 2 == self.p_h % 2 == 0), "p_s % 2 != 0"
        
        # def bnD
        self.b_n = int(b_n)
        self.bnd = self._calc_bnd()
        assert(self.bnd != 0), "Too Few Bits"
        
        # def model itrs
        self.itrs = itrs

        # Model Components
        self.encoder = ConvRnnEncoder()
        
        self.binarizer = ConvBinarizer(
            in_dim=512,
            bnd=self.bnd
        )
        self.decoder = ConvRnnDecoder(
            bnd=self.bnd
        )

    def forward(self, r0, h_e=None, h_d=None):
  
        r = r0
        losses = []
        
        for i in range(self.itrs):
            
            # forward 
            enc, h_e = self.encoder(r, h_e)
            b = self.binarizer(enc)
            dec, h_d = self.decoder(b, h_d)

            # residual error
            r = r0 - dec
            losses.append(nn.L1Loss()(dec, target=r0))
            
        loss = sum(losses) / self.itrs

        return loss   
    
    def encode_decode(self, r0, itrs, h_e=None, h_d=None):
 
        if self.itrs < itrs: 
            warnings.warn('itrs > Training Iterations')

        with torch.no_grad():

            # init dec
            dec = None

            # extract original image dimensions
            img_size = r0.size()[2:]

            # covert images to patches
            r0 = f.sliding_window(
                input=r0,
                kernel_size=self.p_s,
                stride=self.p_s
            )

            r = r0
            
            for i in range(itrs):

                # encode & decode
                enc, h_e = self.encoder(r, h_e)
                b = self.binarizer(enc)
                dec, h_d = self.decoder(b, h_d)

                # calculate residual error
                r = r0 - dec

            # reshape patches to images
            dec = f.refactor_windows(
                windows=dec,
                output_size=img_size
            )
           
        return dec  
    
    def _calc_bnd(self):
        s = (self.p_w//16)*(self.p_h//16)
        return self.b_n//s
