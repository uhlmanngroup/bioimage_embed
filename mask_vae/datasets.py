#  %%
import sys
from torch.utils.data import random_split, DataLoader
import glob
# Note - you must have torchvision installed for this example
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from scipy.ndimage import convolve,sobel 

from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


class DSB2018(Dataset):
    def __init__(self, path_glob, transform=None):
        self.image_paths = glob.glob(path_glob, recursive=True)
        self.transform = transform
        
    # def make_subset(self, index):
    #     self.getitem(index)

    def is_image_cropped(image):
        if (np.sum(
            np.sum(image[:,0]),
            np.sum(image[:,-1]),
            np.sum(image[0,:]),
            np.sum(image[-1,:])
        ) == 0):
            return False
        else:
            return True
    
    def getitem(self, index):
        try:
            x = Image.open(self.image_paths[index])
            # if self.transform is not None:
            x = self.transform(x)
            return x
        except:
            return None
        
    
    def __getitem__(self, index):
        # x = self.getitem(index)
        # if self.is_image_cropped(x):
        # return index
        # if isinstance(index, slice):
            # return [self.getitem(i) for i in range(*index.indices(10))]
        # TODO implement indexing/slicing
        # return self.getitem(index)
        dummy_list = np.arange(0,self.__len__())
        return [self.getitem(i) for i in dummy_list[index]]

        # else:
        #     return self.getitem(index+x)

    def __len__(self):
        return len(self.image_paths)