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
from urllib.error import URLError

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive, verify_str_arg



class DatasetGlob(Dataset):
    def __init__(self, path_glob, transform=None,**kwargs):
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
            if self.transform is not None:
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

        # return self.getitem(index)
        dummy_list = np.arange(0,self.__len__())
        loop = np.array(dummy_list[index])
        if isinstance(index, slice):
            return [self.getitem(i) for i in loop]
        return self.getitem(index)

        # else:
        #     return self.getitem(index+x)

    def __len__(self):
        return len(self.image_paths)
    
class BBBC010(DatasetGlob):
    url = "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip"
    md5 = "ae73910ed293397b4ea5082883d278b1"
    dataset = "bbbc010"
    # folder_name = "BBBC010_v1_foreground_eachworm"
    images_file = "BBBC010_v1_foreground_eachworm.zip"
    filename = images_file
    
    def __init__(self, data_folder="data", download=True, transform=None, **kwargs):

        self.raw_folder = f"{data_folder}/{self.dataset}"
        path_glob = f"{self.raw_folder}/**/*.png"
        
        # self.image_paths = glob.glob(path_glob, recursive=True)
        # self.transform = transform
        if download:
            self.download()
        super(BBBC010, self).__init__(path_glob, transform, **kwargs)

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file))

    def download(self) -> None:
        
        """Download the BBBC010 data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files

        try:
            print(f"Downloading {self.url}")
            download_and_extract_archive(self.url, download_root=self.raw_folder, filename=self.filename, md5=self.md5)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")