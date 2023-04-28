#  %%
import sys
from torch.utils.data import random_split, DataLoader
import glob
import random

# Note - you must have torchvision installed for this example
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from scipy.ndimage import convolve, sobel

from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from urllib.error import URLError

from bio_vae.idr import IDRDataSet
from PIL import Image, ImageSequence

from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
)


class DatasetGlob(Dataset):
    def __init__(self, path_glob, transform=None, samples=-1, shuffle=True, **kwargs):
        self.image_paths = glob.glob(path_glob, recursive=True)
        if shuffle:
            random.shuffle(self.image_paths)
        if samples > 0 and samples < len(self.image_paths):
            self.image_paths = self.image_paths[0:samples]
        self.transform = transform
        self.samples = samples
        assert len(self.image_paths) > 0

    def __len__(self):
        return len(self.image_paths)

    def getitem(self, index):
        try:
            x = Image.open(self.image_paths[index])
            # x = np.array(x)
            x = ImageSequence.Iterator(x)
            if self.transform is not None:
                x = self.transform(image=np.array(x[0]))["image"]
            return x
        except:
            return None

    # def make_subset(self, index):
    #     self.getitem(index)

    def is_image_cropped(self,image):
        if (
            np.sum(
                np.sum(image[:, 0]),
                np.sum(image[:, -1]),
                np.sum(image[0, :]),
                np.sum(image[-1, :]),
            )
            == 0
        ):
            return False
        else:
            return True

        # return self.transform(Image.open(self.image_paths[index]))

    def __getitem__(self, index):
        # x = self.getitem(index)
        # if self.is_image_cropped(x):
        # return index
        # if isinstance(index, slice):
        # return [self.getitem(i) for i in range(*index.indices(10))]
        # TODO implement indexing/slicing
        # return self.getitem(index)

        # return self.getitem(index)
        dummy_list = np.arange(0, self.__len__())
        loop = np.array(dummy_list[index])
        if isinstance(index, slice):
            return [self.getitem(i) for i in loop]
        return self.getitem(index)

        # else:
        #     return self.getitem(index+x)


class WebArchiveDataset(DatasetGlob):
    def __init__(
        self,
        url,
        md5,
        filename,
        dataset,
        data_folder="data",
        download=True,
        transform=None,
        **kwargs,
    ):
        self.url = url
        self.md5 = md5
        self.dataset = dataset
        self.raw_folder = f"{data_folder}/{self.dataset}"
        self.filename = filename
        self.images_file = self.filename
        path_glob = f"{self.raw_folder}/**/*.png"

        # self.image_paths = glob.glob(path_glob, recursive=True)
        # self.transform = transform
        if download:
            self.download()
        super(WebArchiveDataset, self).__init__(path_glob, transform, **kwargs)

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file))

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files

        try:
            print(f"Downloading {self.url}")
            download_and_extract_archive(
                self.url,
                download_root=self.raw_folder,
                filename=self.filename,
                md5=self.md5,
            )
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")


class BroadDataset(WebArchiveDataset):
    lookup_info = {
        "BBBC010": {
            "url": "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip",
            "md5": "ae73910ed293397b4ea5082883d278b1",
            "dataset": "bbbc010",
            "filename": "BBBC010_v1_foreground_eachworm.zip",
        }
    }

    def __init__(
        self,
        image_set="BBBC010",
        **kwargs,
    ):
        super(BroadDataset, self).__init__(
            **self.lookup_info[image_set],
            **kwargs,
        )


class BBBC010(BroadDataset):
    def __init__(self, *args, **kwargs):
        super(BBBC010).__init__(image_set="BBBC010", *args, **kwargs)
