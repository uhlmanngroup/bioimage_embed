#  %%
from tkinter import N
import pytest
import os
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import umap 
import umap.plot
from sklearn.decomposition import PCA

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
from mask_vae.transforms import DistogramToCoords, MaskToDistogramPipeline, AsymmetricDistogramToMaskPipeline
from mask_vae.transforms import DistogramToMaskPipeline,AsymmetricDistogramToSymmetricDistogram

from mask_vae.models import AutoEncoder, VAE, VQ_VAE, Mask_VAE
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

# def my_collate(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

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
# train_dataset_crop_filtered = [x for x in train_dataset_crop if x is not None]
# img_squeeze = train_dataset_crop[0].unsqueeze(0)
# img_crop = train_dataset_crop[0]
#  %%
# dataloader = DataLoader(train_dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=8, pin_memory=True,collate_fn=my_collate)

ckpt_file = "checkpoints/last.ckpt"


model = Mask_VAE(VQ_VAE(channels=1))
model = LitAutoEncoder(model).load_from_checkpoint(
    ckpt_file, model=model)


# model = LitAutoEncoder.load_from_checkpoint(
#     ckpt_file, model=Mask_VAE(VQ_VAE(channels=1)))
# transformer_dist = MaskToDistogramPipeline(window_size, interp_size)

# print(model)
# trainer.fit(model,
#             ckpt_path="checkpoints/last.ckpt")
# model = lit_model.load_from_checkpoint(checkpoint_path="checkpoints/last.ckpt")
# test_img = torch.tensor(np.zeros((1, 96, 96)), dtype=torch.float32)
# train_dataset = DSB2018(train_dataset_glob, transform=transformer_dist)

test_mask = train_dataset_crop[1].unsqueeze(0)
test_img = train_dataset[1].unsqueeze(0)
plt.imshow(test_img[0][0])
plt.show()
loss,y_prime,_, = model.forward(test_img)
y_prime = y_prime.detach().numpy()
plt.imshow(y_prime[0][0])
plt.show()
#  %%
z = model.encoder(test_img)
y_prime = model.decoder(z).detach().numpy()
plt.imshow(y_prime[0][0])
plt.show()
# print(z)
# z = model.model.model._encoder(test_img)
# z = model.model.model._pre_vq_conv(z)
#  %%
from tqdm import tqdm
z_list = []
for data in tqdm(train_dataset):
    if data is not None:
        z_list.append(model.encoder(data))
    if (len(z_list) >= 1000):
        break
latent = torch.stack(z_list).detach().numpy()
#  %%
latent_umap = latent.reshape(latent.shape[0],-1)
unfit_umap = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine')
unfit_umap = umap.UMAP()

fit_umap = unfit_umap.fit(latent_umap)
proj = fit_umap.transform(latent_umap)

umap.plot.points(fit_umap)
plt.show()
plt.scatter(proj[:,0],proj[:,1])
plt.show()
#  %%
pc_1 = fit_umap.inverse_transform(np.array([[1,0]]))
pc_2 = fit_umap.inverse_transform(np.array([[0,1]]))

plt.plot(pc_1.T)
plt.show()
plt.plot(pc_2.T)
plt.show()
# %%
coord_1 = np.array([[-1,7]])
coord_2 = np.array([[0,-4]])

cluster_1 = fit_umap.inverse_transform(coord_1)
cluster_2 = fit_umap.inverse_transform(coord_2)

cluster_1_z = cluster_1.reshape((z.shape))
cluster_2_z = cluster_2.reshape((z.shape))

z_cluster_1 = model.decoder(torch.tensor(cluster_1_z)).detach().numpy()
z_cluster_2 = model.decoder(torch.tensor(cluster_2_z)).detach().numpy()

mask_1 = AsymmetricDistogramToMaskPipeline(window_size)(z_cluster_1)
mask_2 = AsymmetricDistogramToMaskPipeline(window_size)(z_cluster_2)

plt.title("coord_1")
plt.imshow(mask_1[0][0])
plt.show()
plt.title("coord_2")
plt.imshow(mask_2[0][0])
plt.show()

plt.scatter(proj[:,0],proj[:,1])
plt.scatter(coord_1[:,0],coord_1[:,1],label="coord_1")
plt.scatter(coord_2[:,0],coord_2[:,1],label="coord_2")
plt.legend()
plt.show()
#  %%

from scipy.spatial import ConvexHull, convex_hull_plot_2d

hull = ConvexHull(proj)
plt.plot(proj[hull.vertices,0], proj[hull.vertices,1], 'r--', lw=2)
#  %%

from scipy.stats import gaussian_kde

kde = gaussian_kde(proj)

#  %%

# %%
embed = model.get_embedding()

proj = PCA().fit_transform(embed)
# proj = umap.UMAP(n_neighbors=3,
#                  min_dist=0.1,
#                  metric='cosine').fit_transform(embed)
fit_umap = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine').fit(embed)
proj = fit_umap.transform(embed)

pc_1 = fit_umap.inverse_transform(np.array([[0,10]]))
pc_2 = fit_umap.inverse_transform(np.array([[0,1]]))

plt.scatter(proj[:,0],proj[:,1])
plt.show()
#  %%
# fig = plt.figure())
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(10,2),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )
fig, ax = plt.subplots(10,2,figsize=(4, 20))

for i in range(10):
    z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_random = torch.ones_like(z)
    z_random = z + torch.normal(torch.zeros_like(z), torch.ones_like(z))/2
    generated_image_dist = model.decoder(z_random).detach().numpy()
    # dist_stack = np.stack(
    #     [generated_image_dist[0], np.transpose(generated_image_dist[0],)], axis=0)
    dist_stack = np.stack(
        [generated_image_dist, generated_image_dist.transpose(0,1, 3, 2)], axis=0)

    symmetric_generated_image_dist = np.max(dist_stack, axis=0)
    # print(symmetric_generated_image_dist.shape)
    # plt.imshow(symmetric_generated_image_dist[0][0])
    # plt.savefig("test_dist.png")
    # plt.show()

    # plt.imshow(symmetric_generated_image_dist[0][0])
    # plt.savefig("test_dist.png")
    # plt.show()

    coords = DistogramToCoords(window_size)(symmetric_generated_image_dist)
    # plt.scatter(coords[0, :, 0], coords[0, :, 1])
    # plt.show()
    # mask = DistogramToMaskPipeline(window_size)(
    #     symmetric_generated_image_dist[0])
    mask = AsymmetricDistogramToMaskPipeline(window_size)(
        generated_image_dist)
    # generated_image = model.model.mask_from_latent(z_random,window_size)
    # plt.imshow((mask[0]))
    # plt.savefig("test_dist_mask.png")
    # plt.show()

    # plt.imshow(train_dataset_crop[0][0])
    # plt.show()
    
    ax[i][0].imshow(test_mask[0][0])
    ax[i][1].imshow(mask[0][0])
plt.show()


# %%
