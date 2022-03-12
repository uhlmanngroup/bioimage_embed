

import sys
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import glob
# Note - you must have torchvision installed for this example
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from skimage.measure import regionprops
from torchvision.transforms.functional import crop
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pytorch_lightning import loggers as pl_loggers
import torchvision
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import convolve, sobel
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


class cropCentroid(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        return self.crop_centroid(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def crop_centroid(self, image, size):
        np_image = np.array(image)
        im_height, im_width = np_image.shape

        properties = regionprops(np_image.astype(int),
                                 np_image.astype(int))
        center_of_mass = properties[0].centroid
        # weighted_center_of_mass = properties[0].weighted_centroid
        top = int(center_of_mass[0]-size/2)
        left = int(center_of_mass[1]-size/2)
        height, width = size, size
        # TODO find bad croppings
        # if ((top <= 0)  or (top+height >= im_height)  or (left <= 0) or (left+width >= 0) ):
        # return Image.eval(crop(image,top,left,height,width), (lambda x: 0))
        return crop(image, top, left, height, width)


class DistogramtoImage(torch.nn.Module):
    def __init__(self, size=256+128):
        super().__init__()
        self.size = size

    def forward(self, image):
        return(self.get_points_from_dist_C(image))

    def __repr__(self):
        return self.__class__.__name__

    def get_points_from_dist(self, image):
        return MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=0).fit_transform(image)

    def get_points_from_dist_C(self, tensor):
        dist_list = []
        np_tensor = np.array(tensor)
        for i in range(np_tensor.shape[0]):
            image = np_tensor[0, :, :]
            dist_list.append(self.get_points_from_dist(image))
        return torch.tensor(np.array(dist_list))


class ImagetoDistogram(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        # return self.get_distogram(img, self.size)
        return self.get_distogram_C(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"

    def get_distogram_C(self, tensor, size):
        dist_list = []
        np_tensor = np.array(tensor)
        for i in range(np_tensor.shape[0]):
            image = np_tensor[0, :, :]
            dist_list.append(self.get_distogram(image, size))
        return torch.tensor(np.array(dist_list))

    def get_distogram(self, image, size):
        distograms = []
        np_image = np.array(image)
        scaling = np.linalg.norm(np_image.shape)
        # for i in range(np_image_full.shape[0]):
        # np_image = np_image_full[i]
        # im_height, im_width = np_image.shape

        contour = find_contours(np_image)
        contour_x, contour_y = contour[0][:, 0], contour[0][:, 1]
        # plt.scatter(contour_x,contour_y)
        # plt.show()
        #  %%
        rho, phi = self.cart2pol(contour_x, contour_y)

        rho_interp = interp1d(np.linspace(0, 1, len(rho)),
                              rho, kind='cubic')(np.linspace(0, 1, size))
        phi_interp = interp1d(np.linspace(0, 1, len(phi)),
                              phi, kind='cubic')(np.linspace(0, 1, size))

        xii, yii = np.divide(self.pol2cart(rho_interp, phi_interp), scaling)
        # distograms.append(euclidean_distances(np.array([xii,yii]).T))
        return euclidean_distances(np.array([xii, yii]).T)

    def cart2pol(self, x, y):
        return(np.sqrt(x**2 + y**2), np.arctan2(y, x))

    def pol2cart(self, rho, phi):
        return(rho * np.cos(phi), rho * np.sin(phi))


    