
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
from scipy.ndimage import convolve,sobel 
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from mask_vae.transforms import DistogramToMaskPipeline

class Mask_VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, model):
        super(Mask_VAE, self).__init__()
        # self.obj = model
        self.model = model
        # self.model.__init__()

    # def __getattr__(self, attr):
    #     return getattr(self.obj, attr)
    
    def forward(self,x):
        return self.model(x)
    
    def decoder(self,z):
        return self.model.decoder(z)
    
    def encoder(self,img):
        return self.model.encoder(img)
    
    def mask_from_latent(self,z,window_size):
        # This should be class-method based
        # I.e. self.decoder(z)
        dist = self.decoder(z).detach().numpy()
        mask = DistogramToMaskPipeline(window_size)(dist)
        return mask
        
        