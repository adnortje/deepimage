# imports
import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


# ----------------------------------------------------------------------------------------------------------------------
# Functions used by Inpainting Network Modules
# ----------------------------------------------------------------------------------------------------------------------

"""
Removes 1-layer of windows from window sequence
    
    Args:
        input (torch.Tensor) : input sequence of windows (B, C, H, W)
    
    Return:
        windows (torch.Tensor) : output sequence with border windows removed (B-n, C, H, W)
        batch_size (int)       : initial batch size 

"""


def extract_center_windows(windows, batch_size):

    # scaling factor
    scale = int(math.sqrt(windows.size(0)/batch_size))

    # extract window dimensions
    win_size = windows.size()[2:]
    w_h, w_w = win_size

    # make grid
    grid = refactor_windows(
        windows=windows,
        output_size=(scale*w_h, scale*w_w)
    )

    # remove outer states
    grid = grid[:, :, w_h:-w_h, w_w:-w_w]

    # grid -> windows
    windows = sliding_window(
        input=grid,
        kernel_size=win_size,
        stride=win_size
    )

    return windows


"""
Implements a Sliding Window Over Input  

    Args:
        input       (torch.Tensor) : input with dimensions (B, C, H, W)
        kernel_size (int, tuple)   : desired window height & width 
        stride      (int, tuple)   : step width & height for sliding window

    Return:
        window (torch.Tensor) : batch of windows (B*N, C, kernel_size), where N is the number of extracted windows 

"""


def sliding_window(input, kernel_size, stride):

    # extract input channel dimension
    _, input_c, _, _ = input.size()

    # def kernel
    kernel_size = _pair(kernel_size)
    k_h, k_w = kernel_size

    # def stride
    stride = _pair(stride)
    s_h, s_w = stride

    # convert images to grids
    windows = input.unfold(
        dimension=1,
        size=input_c,
        step=input_c
    ).unfold(
        dimension=2,
        size=k_h,
        step=s_h
    ).unfold(
        dimension=3,
        size=k_w,
        step=s_w
    ).contiguous().view(-1, input_c, *kernel_size)

    return windows


"""
Refactor a sequence of windows into the original tensor shape

    Args:
        windows  (torch.Tensor) : sequence of windows with shape (B, C, H, W)
        output_size (int tuple) : desired output spatial dimensions (h, w)

    Return:
        windows (torch.Tensor)  : refactored windows with shape (B, C, h, w)

"""


def refactor_windows(windows, output_size):

    # extract output dimensions
    h, w = output_size

    # extract windows channels
    c = windows.size(1)

    # number of patches per row
    n_row = w // windows.size(3)

    if windows.size(0) % n_row != 0:
        raise ValueError("Windows cannot be refactored into specified output size!")

    # cluster windows into rows
    windows = torch.stack(
        torch.split(
            tensor=windows,
            split_size_or_sections=n_row,
            dim=0
        ),
        dim=0
    )

    # refactor windows
    windows = windows.permute(0, 3, 1, 4, 2).contiguous()
    windows = windows.view(-1, h, w, c)
    windows = windows.permute(0, 3, 1, 2)

    return windows


"""
Pad with Tensor Values

    pads a 2D tensor with a specified tensor value.
    
    Args: 
        input_tensor (torch.Tensor) : input tensor of shape (B, C, H, W)
        padding      (torch.Tensor) : tensor containing padding values (B, C, h, w)
    
    Return:
        pad_tensor (torch.Tensor) : padded tensor with shape (B, C, H + h, W + w)

"""


def tensor_pad2d(input_tensor, padding):

    if input_tensor.size()[0:2] != padding.size()[0:2]:
        raise ValueError("Disagreement in batch and channel size between input and padding tensors!")

    b, c, h, w = input_tensor.size()
    pad_h, pad_w = padding.size()[2:]

    if w % pad_w != 0 or h % pad_h != 0:
        raise ValueError("Padding tensor width or height does not agree with input dimensions!")

    pad_tensor = nn.ZeroPad2d(
        padding=(pad_w, pad_w, pad_h, pad_h)
    )(input_tensor)

    horizontal_pad = torch.cat(
        [padding for _ in range(w//pad_w + 2)],
        dim=3
    )

    vertical_pad = torch.cat(
        [padding for _ in range(h//pad_h + 2)],
        dim=2
    )

    # pad bottom
    pad_tensor[:, :, h+pad_h:, :] = horizontal_pad

    # pad top
    pad_tensor[:, :, 0:pad_h, :] = horizontal_pad

    # pad left
    pad_tensor[:, :, :, 0:pad_w] = vertical_pad

    # pad right
    pad_tensor[:, :, :, w+pad_w:] = vertical_pad

    return pad_tensor
