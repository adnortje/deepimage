import os
import glob
from PIL import Image
from torch.utils.data import Dataset

"""
Class: ImageDataset

    Extends the torch Dataset class. Facilitates the creation of an iterable 
    dataset from an image folder.
    
    Args:
        rootDir   (string)             : path to directory containing images
        transform (callable, optional) : optional transforms to be aplied to a sample image
        img_ext   (string)             : image file extension, default is png
        
    Returns:
        leng (int)             : total no. of image samples in dataset
        img  (Image or Tensor) : Transformed Image 
"""


class ImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, img_ext='png'):
        
        self.root_dir = os.path.expanduser(
            root_dir
        )

        # check directory exists
        assert(os.path.isdir(self.root_dir))
        
        self.transform = transform

        # get list of image names
        self.image_names = glob.glob(
            self.root_dir+'/*.'+img_ext
        )
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        # fetch image name
        img = self.image_names[idx]

        # open image & convert to RGB colour space
        img = Image.open(img).convert('RGB')

        # apply transform
        if self.transform is not None:
            img = self.transform(img)

        return img
