# imports

import os
import torch
import numbers
import img_tools.ImgTools as im_t
import torchvision.transforms as tf
import img_tools.ImgTransforms as ctf
from torch.utils.data import DataLoader
from img_tools.ImgDataset import ImageDataset

# ----------------------------------------------------------------------------------------------------------------------
# DataLoaders for Training & Validation Patches
# ----------------------------------------------------------------------------------------------------------------------

"""
Train_Img_DataLoaders

    facilitates the creation of dataLoaders for system training

    Args:
        b_s (int)        : batch size
        p_s (int)        : patch size
        rootDir (string) : data directory containing train & valid sub-folders     
"""


class TrainImageDataLoaders:

    def __init__(self, b_s, p_s, root_dir=None):

        self.b_s = b_s

        if isinstance(p_s, numbers.Number):
            self.p_s = (int(p_s), int(p_s))
        else:
            self.p_s = p_s

        self.p_h, self.p_w = self.p_s

        if root_dir is None:
            root_dir = '~/Pictures/Clic/Professional'

        self.root_dir = os.path.expanduser(
            root_dir
        )

        # def root to valid & train dir
        self.data_rt = {
            'train': self.root_dir + '/train',
            'valid': self.root_dir + '/valid'
        }

        # check train and valid directories exist
        if not os.path.isdir(self.data_rt['train']):
            raise NotADirectoryError('train directory d.n.e!')
        if not os.path.isdir(self.data_rt['valid']):
            raise NotADirectoryError('valid directory d.n.e!')

    def get_train_dls(self):

        # Image to Tensor with shape (C, h, w)
        train_transform = tf.Compose([

            # fetch random patch from image
            tf.RandomCrop(self.p_s),
            tf.ToTensor(),
            tf.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        valid_transform = tf.Compose([

            # fetch random patch from image
            tf.CenterCrop((self.p_h, self.p_w)),
            tf.ToTensor(),
            tf.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        # def train set
        train_set = ImageDataset(
            root_dir=self.data_rt['train'],
            transform=train_transform
        )

        # def train dl
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.b_s,
            shuffle=True,
            num_workers=2
        )

        # def valid set
        valid_set = ImageDataset(
            root_dir=self.data_rt['valid'],
            transform=valid_transform
        )

        # def valid dl
        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=self.b_s,
            shuffle=True,
            num_workers=2
        )

        # def dict dl's
        dataloaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        return dataloaders

    def display_data(self, dataset, n_row=5, inpaint=False):

        # display data from dataLoaders
        patches = iter(self.get_train_dls()[dataset]).next()

        # display image patches
        im_t.disp_patches(
            patches,
            nrow=n_row,
            norm=True
        )

        return


# ----------------------------------------------------------------------------------------------------------------------
# DataLoaders for Images, Patches given an Image Directory
# ----------------------------------------------------------------------------------------------------------------------

"""
EvaluationImageDataLoaders

    facilitates the creation of img & ptch dataloaders from an Image directory for eval purposes.

    Args:
        b_s     (int, tuple) : batch size
        p_s     (int, tuple) : patch size
        img_dir (string)     : directory containg image files
        img_s   (int, tuple) : size to resize images to bfr processing     
"""


class EvaluationImageDataLoaders:

    def __init__(self, img_dir, img_s, p_s, b_s=1):

        self.b_s = b_s

        # def patch size
        if isinstance(p_s, numbers.Number):
            self.p_s = (int(p_s), int(p_s))
        else:
            self.p_s = p_s

        self.p_h, self.p_w = self.p_s

        # def img size
        if isinstance(img_s, numbers.Number):
            self.img_s = (int(img_s), int(img_s))
        else:
            self.img_s = img_s

        self.img_h, self.img_w = self.img_s

        self.img_dir = os.path.expanduser(
            img_dir
        )
        # check that image directory exists
        assert (os.path.isdir(self.img_dir))

        # number of patches in img width
        self.npw = self.img_w // self.p_w

    def get_img_dl(self):

        # define image transform
        img_trans = tf.Compose([

            # resize image
            tf.Resize(
                self.img_s
            ),
            # convert to tensor
            tf.ToTensor(),
            # Normalize image
            tf.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        # define image dataset
        img_set = ImageDataset(
            root_dir=self.img_dir,
            transform=img_trans
        )

        # def dataLoader
        img_dl = DataLoader(
            dataset=img_set,
            batch_size=self.b_s,
            shuffle=False
        )

        return img_dl

    def get_patch_dl(self):

        # define image to random patch transform
        patch_trans = tf.Compose([

            # randomly crop img
            tf.RandomCrop(
                size=max(self.p_s)
            ),

            # convert PIL Image to Tensor
            tf.ToTensor(),

            # Normalize Image [0,1] -> [-1,-1]
            tf.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        # define patch dataset
        patch_set = ImageDataset(
            root_dir=self.img_dir,
            transform=patch_trans
        )

        # def dataLoader
        patch_dl = DataLoader(
            dataset=patch_set,
            batch_size=self.b_s,
            shuffle=False,
        )

        return patch_dl

    def display_patches(self):

        # display data from dataLoaders

        patches = iter(
            self.get_patch_dl()
        ).next()

        # [-1, 1] -> [0, 1]
        for i in range(patches.size(0)):
            patches[i] = ctf.InvNormalization(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )(patches[i])

        # display patches
        im_t.disp_patches(
            patches,
            n=5
        )

        return

    def display_img(self):

        # display data from image dataloader

        img = iter(
            self.get_img_dl()
        ).next()[0]

        #  Inverse Normalization [-1,1] -> [0,1]
        img = ctf.InvNormalization(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )(img)

        # display image
        im_t.imshow(img)

        return
