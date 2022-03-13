

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


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model, batch_size=1, learning_rate=1e-3):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        # self.loss_fn = torch.nn.BCELoss()
        
    def decoder(self,z):
        return self.model.decoder(z)
    
    def encoder(self,img):
        return self.model.encoder(img)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)

    def training_step(self, inputs, batch_idx):
        vq_loss, output, perplexity = self.forward(inputs)
        # output = x_recon
        # loss = self.loss_fn(output, inputs)

        # vq_loss, data_recon, perplexity = model(inputs)
        # recon_error = F.mse_loss(output, inputs)
        recon_error = self.loss_fn(output, inputs)
        loss = recon_error + vq_loss  # Missing variance bit
        self.log("train_loss", loss)
        # tensorboard = self.logger.experiment
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

        # torchvision.utils.make_grid(output)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx)
        # self.logger.experiment.add_embedding(
        #     "input_image", torchvision.utils.make_grid(transformer_image(inputs)), batch_idx)
        self.logger.experiment.add_image(
            "output", torchvision.utils.make_grid(output), batch_idx)
        # self.logger.experiment.add_embedding(
        #     "output_image", torchvision.utils.make_grid(transformer_image(output)), batch_idx)

        # tensorboard.add_image("input", transforms.ToPILImage()(output[batch_idx]), batch_idx)
        # tensorboard.add_image("output", transforms.ToPILImage()(output[batch_idx]), batch_idx)
        return loss

    def get_embedding(self):
        return self.model.get_embedding()