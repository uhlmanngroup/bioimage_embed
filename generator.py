import pytest
import os
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#  %%
from ast import excepthandler
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

from mask_vae.datasets import DSB2018
from mask_vae.transforms import ImagetoDistogram, cropCentroid, DistogramToCoords, CropCentroidPipeline, DistogramToCoords, MaskToDistogramPipeline, DistogramToMaskPipeline
from mask_vae.models import AutoEncoder, VAE, VQ_VAE, Mask_VAE
from mask_vae.lightning import LitAutoEncoder, LitVariationalAutoEncoder

window_size = 96

# torch_model = VQ_VAE(channels=1)
# model = LitAutoEncoder(VQ_VAE(channels=1))
trainer = pl.Trainer()
ckpt_file = "checkpoints/last.ckpt"
model = LitAutoEncoder.load_from_checkpoint(
    ckpt_file, model=Mask_VAE(VQ_VAE(channels=1)))
# print(model)
# trainer.fit(model,
#             ckpt_path="checkpoints/last.ckpt")
# model = lit_model.load_from_checkpoint(checkpoint_path="checkpoints/last.ckpt")
test_img = torch.tensor(np.zeros((1, 96, 96)), dtype=torch.float32)
print(test_img)
z = model.model.encoder(test_img)
# print(z)
#  %%
for i in range(1):
    z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_random = torch.ones_like(z)
    # z_random = z
    generated_image_dist = model.model.decoder(z_random).detach().numpy()
    # dist_stack = np.stack(
    #     [generated_image_dist[0], np.transpose(generated_image_dist[0],)], axis=0)
    dist_stack = np.stack(
        [generated_image_dist, generated_image_dist.transpose(0, 2, 1)], axis=0)

    symmetric_generated_image_dist = np.max(dist_stack, axis=0)
    # print(symmetric_generated_image_dist.shape)
    plt.imshow(symmetric_generated_image_dist[0])
    plt.savefig("test_dist.png")
    plt.show()
    out = DistogramToMaskPipeline(window_size)(
        symmetric_generated_image_dist)
    # generated_image = model.model.mask_from_latent(z_random,window_size)
    plt.imshow((out[0]))
    plt.savefig("test_dist_mask.png")
    plt.show()

# %%

# %%
