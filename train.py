"""
Python Training Script

    used to train Image Compression Models

"""

# imports 

import os
import time
import torch
import torch.nn as nn
import argparse as arg
import torch.optim as optim
import networks as img_auto
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from img_tools import TrainImageDataLoaders

# --------------------------------------------------------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------------------------------------------------------


# create Argument Parser
parser = arg.ArgumentParser(
    prog='Train: Image Compression System:',
    description='Facilitates the training of an image compression system'
)
parser.add_argument(
    '--sys',
    '-s',
    metavar='SYSTEM',
    type=str,
    required=True,
    choices=['FForward', 'Conv', 'ConvRNN'],
    help='Name of image compression system'
)
parser.add_argument(
    '--epochs',
    '-e',
    metavar='EPOCHS',
    type=int,
    default=30,
    help='No. of training epochs'
)
parser.add_argument(
    '--learn_rate',
    '-lr',
    metavar='LEARN_RATE',
    type=float,
    default=0.0001,
    help='learning rate'
)
parser.add_argument(
    '--gamma',
    '-g',
    metavar='GAMMA',
    type=float,
    default=0.1,
    help='learning rate decay rate'
)
parser.add_argument(
    '--log',
    '-l',
    metavar='LOG_DIR',
    type=str,
    default='./',
    help='Directory to store training logs'
)
parser.add_argument(
    '--trainDir',
    '-td',
    metavar='TRAIN_DIR',
    type=str,
    default='./',
    help='Directory to fetch training data'
)
parser.add_argument(
    '--save_loc',
    '-sv',
    metavar='SAVE_LOC',
    type=str,
    default='./',
    help='Directory to save trained model'
)
parser.add_argument(
    '--img_ext',
    '-ie',
    metavar='IMG_EXT',
    type=str,
    default='.png',
    help='image extension'
)
parser.add_argument(
    '--patch_size',
    '-ps',
    metavar='PATCH_SIZE',
    type=int,
    default=32,
    help='size of training image patches'
)
parser.add_argument(
    '--batch_size',
    '-bs',
    metavar='BATCH_SIZE',
    type=int,
    default=32,
    help='batch size'
)
parser.add_argument(
    '--itrs',
    metavar='ITRS',
    type=int,
    default=16,
    help='No. of image encoder iterations'
)
parser.add_argument(
    '--bits',
    '-bn',
    metavar='BITS',
    type=int,
    default=128,
    help='Bits in bottleneck layer of autoencoder'
)
parser.add_argument(
    '--verbose',
    '-v',
    action='store_true'
)
parser.add_argument(
    '--checkpoint',
    '-chkp',
    action='store_true'
)
args = parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------------------------------------------------

# GPU or CPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define System

# init variables
sys = None
writer = None

if args.sys == 'FForward':

    sys = img_auto.FForwardAutoencoder(
        itrs=args.itrs,
        p_s=args.patch_size,
        b_n=args.bits
    )

elif args.sys == 'Conv':

    sys = img_auto.ConvAutoencoder(
        itrs=args.itrs,
        p_s=args.patch_size,
        b_n=args.bits
    )

elif args.sys == 'ConvRNN':

    sys = img_auto.ConvRnnAutoencoder(
        itrs=args.itrs,
        p_s=args.patch_size,
        b_n=args.bits,
    )


if torch.cuda.device_count() > 1:
    # Run on Parallel GPUs if available
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    sys = nn.DataParallel(sys)

sys.to(device)

# def optimizer
optm = optim.Adam(
    sys.parameters(),
    args.learn_rate
)

# def scheduler
scheduler = lr_scheduler.MultiStepLR(
    optimizer=optm, 
    milestones=[3000, 10000, 14000, 18000, 28000, 34000], 
    gamma=args.gamma
)

# check train, log and save locations
log_loc = os.path.expanduser(args.log)
assert(os.path.isdir(log_loc))

save_loc = os.path.expanduser(args.save_loc)
assert(os.path.isdir(save_loc))

train_dir = os.path.expanduser(args.trainDir)
assert(os.path.isdir(train_dir))

# def patch dataLoader
dataloaders = TrainImageDataLoaders(
    b_s=args.batch_size,
    p_s=args.patch_size,
    root_dir=train_dir
).get_train_dls()

# def start epoch, t_prev, best_Loss
t_prev = 0.0
best_Loss = 1000
curr_epoch = 1

# def state file
state_file = ''.join([
    save_loc,
    '/',
    sys.name,
    '_',
    'chkp.pt'
])

if os.path.isfile(state_file):

    print("Continue Training from m.r.c : ")

    # try load checkpoint
    chkp = torch.load(state_file)

    # load start epoch, best Loss & prev train time
    t_prev = chkp['time']
    best_Loss = chkp['Loss']
    curr_epoch = chkp['epoch']

    # load model, scheduler & optimizer state dict
    sys.load_state_dict(
        chkp['sys']
    )
    optm.load_state_dict(
        chkp['optim']
    )
    scheduler.load_state_dict(
        chkp['sched']
    )

    del chkp

else:
    print("Training New System : ")

# write to log file
if args.verbose:
    writer = SummaryWriter(log_loc)
    print("START TRAINING")

# start timing
train_strt = time.time()

for epoch in range(curr_epoch, args.epochs + 1, 1):

    # start epoch
    if args.verbose:
        print("Epoch {}/{}".format(epoch, args.epochs))
        print("--------------------------------------")

    epoch_strt = time.time()

    for phase in ['train', 'valid']:

        # running Loss
        run_Loss = 0.0

        if phase is 'train':
            # set to train
            sys.train(True)
            # step scheduler
            scheduler.step()

        elif phase is 'valid':
            # set to train
            sys.train(False)

        for i, data in enumerate(dataloaders[phase], 0):

            # place data on GPU
            r = data.to(device)

            # zero model gradients
            optm.zero_grad()

            # forward & loss
            loss = sys(r)

            run_Loss += loss.item()

            # backward
            if phase is 'train':
                loss.backward()
                optm.step()

            # del loss to save memory
            del loss

        # Loss averaged over no. btches
        epoch_Loss = run_Loss / (i+1)

        if args.verbose:
            # display Phase Loss and write to log file
            print("Phase: {} Loss : {}".format(phase, epoch_Loss))
            writer.add_scalar('{}/Loss'.format(phase), epoch_Loss, epoch)
        
        if phase is 'valid' and epoch_Loss < best_Loss:
            # save best system
            best_Loss = epoch_Loss
            fn = save_loc + '/' + sys.name + '.pt'
            torch.save(sys.state_dict(), fn)

    # end of epoch
    epoch_time = (time.time() - epoch_strt) / 60

    if args.verbose:
        print("Epoch time: {} min".format(epoch_time))
        print("-------------------------------------")

    if args.checkpoint:
        # save checkpoint
        chkp = {
            'epoch': epoch + 1,
            'sys': sys.state_dict(),
            'optim': optm.state_dict(),
            'sched': scheduler.state_dict(),
            'Loss': best_Loss,
            'time': time.time() - train_strt + t_prev
        }
        torch.save(chkp, state_file)

# END of Training
train_time = (time.time() - train_strt + t_prev) / 60

if args.verbose:
    print('Total Training Time {} min'.format(train_time))
    print('Best Loss : {}'.format(best_Loss))
    print('FIN TRAINING')

# close writer
writer.close()

