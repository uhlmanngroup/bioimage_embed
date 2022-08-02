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
from mask_vae.transforms import DistogramToMaskPipeline
from .utils import BaseVAE
from mask_vae.models import VQ_VAE, VAE


class Mask_VAE(BaseVAE):
    model_lookup = {
        "vq_vae": VQ_VAE,
        "vae": VAE,
    }
    # model_defaults = {VQ_VAE:{"channels":1},
    #                   VAE: {}}
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, model="VQ_VAE", *args, **kwargs):
        super(Mask_VAE, self).__init__()
        if type(model) is str:
            self.model = self.model_lookup[model.lower()](*args, **kwargs)
        else:
            self.model = model

    # def __getattr__(self, attr):
    #     return getattr(self.obj, attr)

    def forward(self, x):
        return self.model(x)

    def decoder(self, z):
        return self.model.decoder(z)

    def encoder(self, img):
        return self.model.encoder(img)

    def decode(self, z):
        return self.model.decode(z)

    def encode(self, img):
        return self.model.encode(img)

    def recon(self, img):
        return self.model.recon(img)

    def mask_from_latent(self, z, window_size):
        # This should be class-method based
        # I.e. self.decoder(z)
        dist = self.decoder(z).detach().numpy()
        mask = DistogramToMaskPipeline(window_size)(dist)
        return mask

    def get_embedding(self):
        return self.model.get_embedding()

    def loss_function(self, *args, recons, input, distance_matrix_loss=True, **kwargs):

        # decode_z, input, mu, log_var = kwargs
        # # Check to see if distance matrix creates a shape without intersecting edges
        # x_diff = torch.diff(recons,1,-1)-torch.diff(recons,2,-1)
        # y_diff = torch.diff(recons,1,-2)
        
        # Need to invent metric for ensuring that the final shape is a simple polygon

        diag_loss = F.mse_loss(
            torch.diagonal(recons), torch.zeros_like(torch.diagonal(recons))
        )
        symmetry_loss = F.mse_loss(recons, recons.transpose(3, 2))
        vae_loss = self.model.loss_function(*args, recons=recons, input=input, **kwargs)
        if distance_matrix_loss:
            vae_loss["loss"] = (
                8 / 10 * vae_loss["loss"] + 1 / 10 * diag_loss + 1 / 10 * symmetry_loss
            )

        return vae_loss

    def output_from_results(self, *args, **kwargs):
        return self.model.output_from_results(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)
