"""
Script: calc_cc:

    used to calculate Codec Compression Curve values & save values to numpy file
    
"""

# imports
import os
import numpy as np
import argparse as arg
import sys
sys.path.append("..")
from image_codec import ImageCodec

# Argument Parser
parser = arg.ArgumentParser(
    prog='Calculate Codec Compression Curve Values:',
    description='Calculates & Stores Metric vs Bpp values'
)
parser.add_argument(
    '--codec',
    '-c',
    metavar='CODEC',
    type=str,
    choices=['JPEG', 'WEBP', 'JPEG2000'],
    required=True,
    help='specify lossy image codec'
)
parser.add_argument(
    '--metric',
    '-m',
    metavar='METRIC',
    type=str,
    choices=['SSIM', 'PSNR'],
    required=True,
    help='specify image evaluation metric'
)
parser.add_argument(
    '--ImgDir',
    '-id',
    metavar='IMG_DIR',
    type=str,
    required=True,
    help='specify image directory'
)
parser.add_argument(
    '--save_loc',
    '-sl',
    metavar='SAVE_LOC',
    type=str,
    required=True,
    help='specify save directory'
)
args = parser.parse_args() 

# check save_loc & Img dir exists
ImgDir = os.path.expanduser(
    args.ImgDir
)
assert(os.path.isdir(ImgDir))

save_loc = os.path.expanduser(
    args.save_loc
)
assert(os.path.isdir(save_loc))


# calc Compression Curve values 
codec = ImageCodec(
    img_dir=ImgDir,
    codec=args.codec
)

# Save Values to numpy File
cc = codec.get_cc(args.metric)

# Save Values to numpy File
save_loc = save_loc + args.codec + '_' + args.metric
np.save(save_loc, cc)
