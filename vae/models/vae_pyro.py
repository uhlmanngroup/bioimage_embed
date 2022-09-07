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


class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, h_dim=(1, 5, 5), z_dim=(1, 5, 5), use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.autoencoder = AutoEncoder(1, 1)
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder

        self.z_dim = torch.tensor(z_dim)
        self.x_dim = (1, window_size, window_size)
        self.h_dim = torch.tensor(h_dim)

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(torch.prod(self.h_dim), torch.prod(self.z_dim))
        self.fc21 = nn.Linear(torch.prod(self.h_dim), torch.prod(self.z_dim))
        self.fc22 = nn.Linear(torch.prod(self.h_dim), torch.prod(self.z_dim))

        pyro.module("decoder", self.autoencoder)

        # self.fc3 = nn.Linear(torch.prod(self.z_dim), torch.prod(self.h_dim))
        self.softplus = nn.Softplus()

    def encode(self, x):
        h = self.encoder(x)
        # h = self.softplus(h)
        # h = self.flatten(h)
        # z = self.sigmoid(h)

        # No clue if this is actually mu
        z = self.fc21(self.flatten(h))
        mu = torch.exp(self.fc22(self.flatten(h)))
        # z, mu = self.bottleneck(h)
        return z.reshape(h.shape), mu.reshape(h.shape)

    def decode(self, z):
        # z = self.fc3(z).reshape(-1,*tuple(self.h_dim))
        z = z.reshape(-1, *tuple(self.h_dim))
        return self.decoder(z)

    def forward(self, x):
        z, mu = self.encode(x)
        return self.decode(z)

    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], *tuple(self.h_dim))))
            z_scale = x.new_ones(torch.Size((x.shape[0], *tuple(self.h_dim))))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(3))
            # decode the latent code z
            img = self.decode(z)
            loc_img = torch.sigmoid(img)
            scale = torch.ones_like(loc_img)
            # score against actual images
            pyro.sample("obs", dist.ContinuousBernoulli(logits=img).to_event(3), obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encode(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(3))

    def construct_from_z(self, z):
        return torch.sigmoid(self.decode(z))

    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encode(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        # loc_img = torch.sigmoid(self.decode(z))
        return self.construct_from_z(z)
