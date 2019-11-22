# imports
from torch.autograd import Function

"""
Autograd Function : binz

    forward and backward methods for binarization function
    
    Ref (math) : https://arxiv.org/pdf/1511.06085.pdf Stochastic Binarization
    Ref (code) : https://github.com/1zb/pytorch-image-comp-rnn/blob/master/functions/sign.py
    
"""


class binz(Function):
    
    @staticmethod
    def forward(ctx, inpt, is_train=True):
        # forward: (x), a value between [-1,1] to be converted to {-1,1}
        if is_train:
            # during training
            p = inpt.new(inpt.size()).uniform_()
            x = inpt.clone()
            x[(1 - inpt)/2 <= p] = 1
            x[(1 - inpt)/2 > p] = -1
        else:
            # inference mode
            x = inpt.sign()
        return x
    
    @staticmethod
    def backward(ctx, grad_out):
        # backward: (grad_out), output gradient is passed through unchanged (STE)
        grad_in = grad_out.clone()
        return grad_in, None
