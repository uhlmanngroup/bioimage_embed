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
from mask_vae.models import AutoEncoder, VAE, VQ_VAE
from mask_vae.lightning import LitAutoEncoder, LitVariationalAutoEncoder

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

train_dataset_glob = os.path.join(os.path.expanduser("~"),
                                  "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size,interp_size)
transformer_coords = DistogramToCoords(window_size)

train_dataset_raw = DSB2018(train_dataset_glob)
train_dataset_crop = DSB2018(train_dataset_glob, transform=CropCentroidPipeline(window_size))
train_dataset_dist = DSB2018(train_dataset_glob, transform=transformer_dist)

img_squeeze = train_dataset_crop[0].unsqueeze(0)
img_crop = train_dataset_crop[0]

train_dataset = train_dataset_dist

dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)


# def test_transforms():
#     dist = np.array(train_dataset_crop[1][0]).astype(float)
#     plt.imshow(dist)
#     plt.show()


def test_dist_to_coord():
    # dist = transformer_dist(train_dataset[0][0])
    plt.close()
    coords = DistogramToCoords(window_size)(train_dataset_dist[0])
    plt.scatter(coords[0][:, 0], coords[0][:, 1])
    plt.savefig("testss/test_dist_to_coord.png")
    plt.show()
    
def test_pipeline_forward():
    # dist = MaskToDistogramPipeline(window_size)(train_dataset_raw[0])
    # plt.imshow(dist)
    # plt.savefig("tests/test_mask_to_dist.png")
    # plt.show()
    # plt.close()
    dist = train_dataset_dist[0]
    plt.imshow(dist.squeeze())
    plt.savefig("tests/test_pipeline_forward.png")
    plt.show()
    mask = DistogramToMaskPipeline(window_size)(dist)    
    plt.imshow(mask.squeeze())
    plt.savefig("tests/test_dist_to_mask.png")
    plt.show()
    
def test_dist_to_coord():
    # dist = transformer_dist(train_dataset[0][0])
    coords = transformer_coords(train_dataset_dist[0])
    plt.scatter(coords[0][:, 0], coords[0][:, 1])
    plt.savefig("tests/test_dist_to_coord.png")
    plt.show()


def test_models():
    # vae = AutoEncoder(1, 1)
    vae = VQ_VAE(channels=1)
    img = train_dataset_crop[0].unsqueeze(0)
    loss, x_recon, perplexity = vae(img)
    z = vae.encoder(img)
    y_prime = vae.decoder(z)
    # print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
    print(
        f"img_dims:{img.shape} x_recon:_dims:{x_recon.shape} z:_dims:{z.shape}")

def test_training():
    model = LitAutoEncoder(VQ_VAE(channels=1))
    trainer = pl.Trainer(
        max_steps=1,
        # limit_train_batches=1,
        # limit_val_batches=1
        # logger=tb_logger,
        # enable_checkpointing=True,
        # gpus=1,
        # accumulate_grad_batches=1,
        # callbacks=[checkpoint_callback],
        # min_epochs=1,
        max_epochs=1,
    )  # .from_argparse_args(args)
    trainer.fit(model, dataloader) 