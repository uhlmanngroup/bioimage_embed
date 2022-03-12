import pytest
import os
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

from mask_vae.datasets import DSB2018
from mask_vae.transforms import ImagetoDistogram, cropCentroid, DistogramToCoords, CropCentroidPipeline, DistogramToCoords, MaskToDistogramPipeline, DistogramToMaskPipeline
from mask_vae.models import AutoEncoder, VAE, VQ_VAE
window_size = 96
train_dataset_glob = os.path.join(os.path.expanduser("~"),
                                  "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size)
transformer_coords = DistogramToCoords(window_size)

train_dataset_crop = DSB2018(train_dataset_glob, transform=transformer_crop)
train_dataset_dist = DSB2018(train_dataset_glob, transform=transformer_dist)

img_squeeze = train_dataset_crop[0].unsqueeze(0)
img = train_dataset_dist[0]

# def test_transforms():
#     dist = np.array(train_dataset_crop[1][0]).astype(float)
#     plt.imshow(dist)
#     plt.show()


def test_dist_to_coord():
    # dist = transformer_dist(train_dataset[0][0])
    coords = transformer_coords(train_dataset_dist[0])
    plt.scatter(coords[0][:, 0], coords[0][:, 1])
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
