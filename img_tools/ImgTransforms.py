# imports
import random
import numbers
from PIL import ImageOps
import torchvision.transforms as tf

"""
Image2Grids
    
    crops a given PIL Image into grid blocks made up of patches
    
    Args:
        size (int) : patch size cropped from image
        n_p  (int) : number of patches per grid row & col
        
    Returns:
        patches (list[]) : list of PIL Image patches in the same order as the grid sampling 
         
"""


class Image2Grids(object):

    def __init__(self, size, n_p, pad=0):

        # define patch size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.p_h, self.p_w = self.size

        # define grid size
        self.n_p = int(n_p)

        self.grid_size = (
            self.n_p*self.p_h,
            self.n_p*self.p_w
        )
        self.grid_h, self.grid_w = self.grid_size

        self.pad = pad

        self.transform = Image2Patches(
            size=self.p_s
        )

    def __call__(self, img):

        # returns grid of Images list[list[PIL Image]]

        patches = []

        # check grid size <= image size
        img_w, img_h = img.size
        assert(
            (self.grid_h <= img_h) and (self.grid_w <= img_w)
        )

        # add padding
        if self.pad > 0:
            img = ImageOps.expand(img, border=self.pad, fill=0)

        # separate into grids of patches
        for h in range(0, img_h, self.grid_h):
            for w in range(0, img_w, self.grid_w):
                grid_block = img.crop(
                    (w, h, w+self.grid_w, h+self.grid_h)
                )
                patches.extend(
                    self.transform(grid_block)
                )

        return patches


"""
Image2Patches 
    
    crops a given PIL Image into a smaller set of patches
    
    Args:
        size (int or tuple): desired output size of the cropped patches.
        pad  (int or tuple): amount of padding to place around Image bfr crop.
    
    Returns:
        patches (list[Image]): List of Image patches that make up an image
"""


class Image2Patches(object):
    
    def __init__(self, size, pad=0):

        # define patch size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.pad = pad
        
    def __call__(self, img):

        # returns list of PIL images

        patches = []

        if self.pad > 0:
            img = ImageOps.expand(img, border=self.pad, fill=0)

        # image width & height
        img_w, img_h = img.size
        # patch width & height
        p_w, p_h = self.size

        # separate into patches
        for h in range(0, img_h, p_h):
            for w in range(0, img_w, p_w):
                patch = img.crop((w, h, w+p_w, h+p_h))
                patches.append(patch)

        return patches
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


"""
RandomPatchSeq
    
    crops a sequence of patches from a PIL Image at random
    
    Args:
        size  (int or tuple): size of the cropped patches.
        pad   (int or tuple): padding to place around Image bfr crop.
        n_seq (int)         : no. patches in seq
        
    Returns:
        patches (list[Image]) : list of PIL Image patches
"""


class RandomPatchSequence(object):
    
    def __init__(self, n_seq, size, pad=0):
        
        # define patch size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        # define padding & number of patches
        self.pad = pad
        self.n_seq = int(n_seq)
        
    def __call__(self, img):

        # get random patch sequence

        patches = []
        
        # add padding
        if self.pad > 0:
            img = ImageOps.expand(img, border=self.pad, fill=0)
        
        # get image & patch width and height
        p_w, p_h = self.size
        img_w, img_h = img.size

        # check img is large enough for sequence extraction
        assert(
                (self.n_seq * p_w <= img_w) and (self.n_seq * p_h <= img_h)
        )
        
        # get random starting row & col
        row = random.sample(
            range(0, (img_h+1)-p_h),
            1
        )[0]
        
        col = random.sample(
            range(0, (img_w+1)-self.n_seq*p_w),
            1
        )[0]
        
        for i in range(0, self.n_seq):
            # increment col by patch width
            c = i*p_w + col
            patch = img.crop((c, row, c+p_w, row+p_h))
            patches.append(patch)
            
        return patches
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1}, n_seq={2})'.format(self.size, self.pad, self.n_seq)


"""
RandomPatches
    
    crops a selected number of patches from an image at random locations
    
    Args:
        n_patches  (int)            : number of random patches to be cropped
        patch_size (sequence or int): desired size of cropped output patches
        padding    (int or sequence): amount of padding to place around Image bfr crop.
    Returns:
        patches    (PIL Image)      : list of Image patches
    
"""


class RandomPatches(object):
    
    def __init__(self, n, size, pad=0):

        # define patch size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        # define number of patches
        self.n = int(n)

        self.pad = pad

        self.transform = tf.RandomCrop(
            self.size, 
            self.pad
        )
        
    def __call__(self, img):

        # get n random patches from img
        patches = []

        # randomly crop n patches
        for i in range(self.n):
            patch = self.transform(img)
            patches.append(patch)

        return patches
    
    def __repr__(self):
        return self.__class__.__name__ + '(no. patches = {0}, size={1}, padding={2})'.format(
            self.n,
            self.size,
            self.pad
        )


"""
InvNormalization
    
    inverse of normalization process
    
    Args:
        mean (int) : per channel mean
        std  (int) : per channel std
        
    Returns:
        Tensor (torch.Tensor)       : InvNormalized Tensor
    
"""


class InvNormalization(object):

    def __init__(self, mean, std):
        
        if isinstance(mean, numbers.Number):
            self.mean = (mean, mean, mean)
        else:
            self.mean = mean
            
        if isinstance(std, numbers.Number):
            self.std = (std, std, std)
        else:
            self.std = std
            
        self.transform = tf.Compose([
            tf.Normalize(
                (0, 0, 0),
                (1/self.std[0], 1/self.std[1], 1/self.std[2])
            ), 
            tf.Normalize(
                (-self.mean[0], -self.mean[1], -self.mean[2]), 
                (1, 1, 1)
            )
        ])
        
    def __call__(self, img_tensor):
        # inverse normalisation
        img = self.transform(img_tensor)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean = {0}, std = {1})'.format(self.mean, self.std)

"""
RandomPatchGrid

    randomly crops a grid of adjacent patches from an image

    Args:
        size (int) : patch size cropped from image
        n_p (int)  : number of patches per grid row & col

"""


class RandomPatchGrid(object):

    def __init__(self, size, n_p):

        # define patch size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.p_h, self.p_w = self.size

        # define grid size
        self.n_p = int(n_p)
        self.grid_size = (
            self.n_p * self.p_h,
            self.n_p * self.p_w
        )
        self.grid_h, self.grid_w = self.grid_size

        # def transform
        self.transform = tf.Compose([
            # randomly crop image
            CustomRandomCrop(
                size=self.grid_size,
            ),
            # convert grid to patches
            Image2Patches(
                size=self.size,
                pad=0
            )
        ])

    def __call__(self, img):

        # check that image > grid size
        img_w, img_h = img.size
        assert (self.grid_h <= img_h and self.grid_w <= img_w)

        # apply img to patches transform
        patches = self.transform(
            img
        )

        return patches

"""
CustomRandomCrop

    randomly crops a patch from the image but takes size under consideration when choosing random row and column values

    Args:
        size (int) : patch size cropped from image
        n_p (int)  : number of patches per grid row & col

"""


class CustomRandomCrop(object):

    def __init__(self, size, pad=0):

        # define patch size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.p_h, self.p_w = self.size

        self.pad = pad

    def __call__(self, img):

        # add padding
        if self.pad > 0:
            img = ImageOps.expand(img, border=self.pad, fill=0)

        # check that image > patch size
        img_w, img_h = img.size

        assert (self.p_h <= img_h and self.p_w <= img_w)

        # randomly select center vs edge transform
        i = random.sample(range(0, img_h, self.p_h), 1)[0]
        j = random.sample(range(0, img_w, self.p_w), 1)[0]

        # crop img
        img = img.crop((j, i, j + self.p_w, i + self.p_h))

        return img
