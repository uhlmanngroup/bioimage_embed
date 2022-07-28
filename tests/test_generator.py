#  %%
import pytest
import os
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from mpl_toolkits.axes_grid1 import ImageGrid

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
from mask_vae.transforms import ImagetoDistogram, cropCentroid, DistogramToCoords,CropCentroidPipeline
from mask_vae.transforms import DistogramToCoords, MaskToDistogramPipeline,AsymmetricDistogramToMaskPipeline
from mask_vae.transforms import DistogramToMaskPipeline

from mask_vae.models import AutoEncoder, VAE, VQ_VAE, Mask_VAE
from mask_vae.lightning import LitAutoEncoderPyro, LitAutoEncoderTorch

interp_size = 128*4

window_size = 128-32
batch_size = 32
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

train_dataset_glob = "data/stage1_train/*/masks/*.png"
train_dataset_glob = "data/BBBC010_v1_foreground_eachworm/*.png"

# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
transformer_coords = DistogramToCoords(window_size)

# train_dataset_raw = DSB2018(train_dataset_glob)
# train_dataset_crop = DSB2018(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))
train_dataset = DSB2018(train_dataset_glob, transform=transformer_dist)
train_dataset_crop = DSB2018(train_dataset_glob, transform=transformer_crop)

# img_squeeze = train_dataset_crop[0].unsqueeze(0)
# img_crop = train_dataset_crop[0]

dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

ckpt_file = "checkpoints/last.ckpt"


model = Mask_VAE(VQ_VAE(channels=1))
model = LitAutoEncoderTorch(model).load_from_checkpoint(
    ckpt_file, model=model)

test_img = train_dataset[1].unsqueeze(0)
# img_squeeze = train_dataset_crop[1].unsqueeze(0)
img_crop = train_dataset_crop[1].unsqueeze(0)

# train_dataset = train_dataset_dist

def test_img_test():
    plt.close()
    plt.imshow(test_img[0][0])
    plt.show()
   
def test_forward_test():
    plt.close()
    loss,y_prime,_, = model.forward(test_img)
    y_prime = y_prime.detach().numpy()
    plt.imshow(y_prime[0][0])
    plt.show()
def test_encoder_decoder():
    plt.close()
    z = model.encoder(test_img)
    y_prime = model.decoder(z).detach().numpy()
    plt.imshow(y_prime[0][0])
    plt.show()
    
def test_generate():
    z = model.encoder(test_img)
    z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_random = torch.ones_like(z)
    z_random = z + torch.normal(torch.zeros_like(z), torch.ones_like(z))/20
    generated_image_dist = model.decoder(z_random).detach().numpy()
    # dist_stack = np.stack(
    #     [generated_image_dist[0], np.transpose(generated_image_dist[0],)], axis=0)
    # dist_stack = np.stack(
    #     [generated_image_dist, generated_image_dist.transpose(0,1, 3, 2)], axis=0)

    # symmetric_generated_image_dist = np.max(dist_stack, axis=0)
    # print(symmetric_generated_image_dist.shape)
    # plt.imshow(symmetric_generated_image_dist[0][0])
    # plt.savefig("test_dist.png")
    # plt.show()

    # plt.imshow(symmetric_generated_image_dist[0][0])
    # plt.savefig("test_dist.png")
    # plt.show()

    # coords = DistogramToCoords(window_size)(symmetric_generated_image_dist[0])
    # plt.scatter(coords[0, :, 0], coords[0, :, 1])
    # plt.show()
    mask = AsymmetricDistogramToMaskPipeline(window_size)(
        generated_image_dist)


    # generated_image = model.model.mask_from_latent(z_random,window_size)
    # plt.imshow((mask[0]))
    # plt.savefig("test_dist_mask.png")
    # plt.show()

    # plt.imshow(train_dataset_crop[0][0])
    # plt.show()
    plt.close()
    plt.imshow(img_crop[0][0])
    plt.savefig("tests/test_generate_img_crop.png")
    plt.show()
    plt.imshow(mask[0][0])
    plt.savefig("tests/test_generate_mask.png")
    plt.show()
# %%
