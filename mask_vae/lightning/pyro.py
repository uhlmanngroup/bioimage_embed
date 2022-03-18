

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


class LitAutoEncoderPyro(pl.LightningModule):
    def __init__(self, model, batch_size=1, learning_rate=1e-3):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.autoencoder = model
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    def forward(self, x):
        return self.autoencoder.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def torch_training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.forward(inputs)
        loss = self.loss_fn(output, inputs)
        self.log("train_loss", loss)
        # tensorboard = self.logger.experiment
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        # torchvision.utils.make_grid(output)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx)
        self.logger.experiment.add_image(
            "output", torchvision.utils.make_grid(torch.sigmoid(output)), batch_idx)

    def pyro_training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.vae.reconstruct_img(inputs)
        loss = self.loss_fn(self.vae.model, self.vae.guide, inputs)
        self.log("train_loss", loss)
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx)
        self.logger.experiment.add_image(
            "output", torchvision.utils.make_grid(torch.sigmoid(output)), batch_idx)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.torch_training_step(train_batch, batch_idx)

    def training_step(self, train_batch, batch_idx):
        return self.pyro_training_step(train_batch, batch_idx)